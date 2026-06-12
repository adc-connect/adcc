#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import glob
import os

import numpy as np
from numpy.testing import assert_allclose
import pytest

import adcc
import adcc.backends
import adcc.block as b
from adcc.backends import have_backend
from adcc.MoSpaces import split_spaces
from adcc.gradients.TwoParticleDensityMatrix import (
    TwoParticleDensityMatrix, ao_pair_indices,
)

from .backends.testing import cached_backend_hf


pytestmark = pytest.mark.skipif(
    not have_backend("pyscf"), reason="pyscf not found."
)

# Spin cases reproduced by ``to_ao_pair_density`` / ``to_ao_basis``.  The
# direct (``g2_ao_1``) cases enter with ``+`` sign, the exchange (``g2_ao_2``)
# cases with ``-`` sign and a swapped ket pair.
_DIRECT_SPIN_CASES = [("a", "a", "a", "a"), ("b", "b", "b", "b"),
                      ("a", "b", "a", "b"), ("b", "a", "b", "a")]
_EXCHANGE_SPIN_CASES = [("a", "a", "a", "a"), ("b", "b", "b", "b"),
                        ("a", "b", "b", "a"), ("b", "a", "a", "b")]


def _random_ao_inputs(nao):
    rng = np.random.default_rng(20240611)
    g1_ao = rng.standard_normal((nao, nao))
    g1_ao = 0.5 * (g1_ao + g1_ao.T)
    w_ao = rng.standard_normal((nao, nao))
    w_ao = 0.5 * (w_ao + w_ao.T)
    g2_ao_1 = rng.standard_normal((nao, nao, nao, nao))
    g2_ao_2 = rng.standard_normal((nao, nao, nao, nao))
    return g1_ao, w_ao, g2_ao_1, g2_ao_2


def _random_tpdm(refstate, blocks=(b.oooo, b.oovv, b.ovov)):
    """
    Build a ``TwoParticleDensityMatrix`` with deterministically filled blocks.

    Using an explicit ``TwoParticleDensityMatrix`` (instead of a physical MP2
    density) gives dense, non-degenerate data in every block so the full spin
    expansion (``aaaa``/``bbbb`` and the mixed ``abab``/``baba``/``abba``/
    ``baab`` cases) is exercised with generic numbers rather than values that
    might accidentally cancel.
    """
    g2 = TwoParticleDensityMatrix(refstate)
    g2.reference_state = refstate
    for blk in blocks:
        tensor = g2[blk]
        tensor.set_random()
        g2[blk] = tensor
    return g2


def _direct_gradient_inputs(mp):
    """Inputs for ``correlated_gradient_direct`` from an ``LazyMp``.

    Returns ``(g1_ao, w_ao, g2_total)`` where the one-electron AO matrices are
    zero (the storage/validation tests only exercise the two-electron packed
    density path) and ``g2_total`` is the real MP2 two-particle density matrix
    with its block prefactors applied, as built by ``nuclear_gradient``.
    """
    grad = adcc.nuclear_gradient(mp, tei_contraction="full_ao")
    nao = mp.reference_state.gradient_provider.mol.nao_nr()
    g1_ao = np.zeros((nao, nao))
    w_ao = np.zeros((nao, nao))
    return g1_ao, w_ao, grad.g2


def _two_stage_packed_density(g2, refstate):
    """
    Dense two-stage reference for the packed AO-pair effective density.

    This is intentionally written as an explicit
    ``(p, r) -> (mu, nu)`` half-transform followed by a
    ``(q, s) -> (lambda, sigma)`` transform + ``s2kl`` pack, so it pins the
    *staging contract* that the libtensor-based refactor (milestones m2/m3)
    must reproduce, independently of the current fused implementation.
    """
    cc = g2._ao_coefficient_map(refstate)
    nao = next(iter(cc.values())).shape[1]
    npair = nao * (nao + 1) // 2
    qall, sall = ao_pair_indices(nao)

    def accumulate(out, qidx, sidx):
        for block in g2.blocks_nonzero:
            spaces = split_spaces(block)
            tensor = np.asarray(g2[block].to_ndarray())
            for spins in _DIRECT_SPIN_CASES:
                c1, c2, c3, c4 = (cc[f"{sp}_{spin}"]
                                  for sp, spin in zip(spaces, spins))
                # stage 1: bra pair (p, r) -> (mu, nu); keep ket pair in MO
                half = np.einsum("ip,kr,ijkl->prjl", c1, c3, tensor,
                                 optimize=True)
                # stage 2: ket pair (q, s) -> packed (lambda, sigma)
                right = c2[:, qidx][:, None, :] * c4[:, sidx][None, :, :]
                out += np.einsum("prjl,jlm->prm", half, right, optimize=True)
            for spins in _EXCHANGE_SPIN_CASES:
                c1, c2, c3, c4 = (cc[f"{sp}_{spin}"]
                                  for sp, spin in zip(spaces, spins))
                half = np.einsum("ip,lr,ijkl->prjk", c1, c4, tensor,
                                 optimize=True)
                right = c2[:, qidx][:, None, :] * c3[:, sidx][None, :, :]
                out -= np.einsum("prjk,jkm->prm", half, right, optimize=True)

    out = np.zeros((nao, nao, npair))
    accumulate(out, qall, sall)
    # complete off-diagonal ket pairs (s2kl stores both orders summed)
    offdiag = qall != sall
    if np.any(offdiag):
        swapped = np.zeros((nao, nao, int(np.count_nonzero(offdiag))))
        accumulate(swapped, sall[offdiag], qall[offdiag])
        out[:, :, offdiag] += swapped
    return out


def test_packed_ao_pair_contraction_matches_full_pyscf_reference():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    provider = hf.gradient_provider
    inputs = _random_ao_inputs(hf.n_bas)
    _, _, g2_ao_1, g2_ao_2 = inputs

    full = provider.correlated_gradient(*inputs)
    pair_density = TwoParticleDensityMatrix.ao_pair_density_from_dense(
        g2_ao_1, g2_ao_2
    )
    packed_tei = provider._contract_tei_with_packed_density(
        pair_density, shell_chunk_size=2
    )

    assert_allclose(packed_tei, full.two_electron, atol=1e-10)


def test_direct_mp2_pair_density_matches_dense_ao_transform():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))
    gradient = adcc.nuclear_gradient(mp, tei_contraction="full_ao")

    g2_ao_1, g2_ao_2 = gradient.g2.to_ao_basis()
    dense_pair_density = TwoParticleDensityMatrix.ao_pair_density_from_dense(
        g2_ao_1.to_ndarray(), g2_ao_2.to_ndarray()
    )
    direct_pair_density = gradient.g2.to_ao_pair_density(
        gradient.reference_state, pair_chunk_size=5
    )

    assert_allclose(direct_pair_density, dense_pair_density, atol=1e-10)


def test_direct_mp2_gradient_matches_full_ao_fallback():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))

    full = adcc.nuclear_gradient(mp, tei_contraction="full_ao")
    direct = adcc.nuclear_gradient(
        mp, tei_contraction="direct", tei_shell_chunk_size=2,
        tei_pair_chunk_size=5
    )

    assert_allclose(direct.total, full.total, atol=1e-10)
    assert_allclose(direct.components.two_electron,
                    full.components.two_electron, atol=1e-10)


# ---------------------------------------------------------------------------
# Guardrail tests (milestone m0)
#
# These pin the libtensor -> dense handover *at the intermediate boundary*
# that the upcoming refactor will move, so a correct relocation of that
# boundary can be told apart from a subtly wrong one.  They include an
# open-shell reference and an explicit two-stage staging reference.
# ---------------------------------------------------------------------------


def test_open_shell_pair_density_matches_dense_ao_transform():
    """Direct packed density must match the dense AO transform for UHF.

    An open-shell reference (CN doublet, n_alpha != n_beta, distinct alpha and
    beta spatial orbitals) is the strongest available check that the mixed-spin
    cases of the transform are handled correctly when the refactor moves the
    handover boundary.
    """
    hf = cached_backend_hf("pyscf", "cn_sto3g", conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)
    assert not refstate.restricted

    g2 = _random_tpdm(refstate)
    g2_ao_1, g2_ao_2 = g2.to_ao_basis()
    dense_pair_density = TwoParticleDensityMatrix.ao_pair_density_from_dense(
        g2_ao_1.to_ndarray(), g2_ao_2.to_ndarray()
    )
    direct_pair_density = g2.to_ao_pair_density(refstate, pair_chunk_size=3)

    assert_allclose(direct_pair_density, dense_pair_density, atol=1e-10)


@pytest.mark.parametrize("system", ["h2o_sto3g", "cn_sto3g"])
def test_packed_density_matches_two_stage_staging_contract(system):
    """Packed density must equal an explicit bra/ket two-stage transform.

    This pins the staging contract (bra half-transform ``(p,r)->(mu,nu)`` then
    ket transform + ``s2kl`` pack) that milestones m2/m3 must reproduce when
    the transform is lifted into libtensor, independently of the current fused
    einsum implementation.
    """
    hf = cached_backend_hf("pyscf", system, conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)

    g2 = _random_tpdm(refstate)
    reference = _two_stage_packed_density(g2, refstate)
    produced = g2.to_ao_pair_density(refstate, pair_chunk_size=3)

    assert_allclose(produced, reference, atol=1e-10)


@pytest.mark.parametrize("pair_chunk_size", [1, 2, 3, 1000])
def test_packed_density_independent_of_pair_chunk_size(pair_chunk_size):
    """The packed density must not depend on the pair-chunk batching.

    The refactor changes how AO pairs are batched; this guards against a
    chunk-boundary bug that would only appear for particular chunk sizes.
    """
    hf = cached_backend_hf("pyscf", "cn_sto3g", conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)

    g2 = _random_tpdm(refstate)
    reference = g2.to_ao_pair_density(refstate, pair_chunk_size=None)
    chunked = g2.to_ao_pair_density(refstate, pair_chunk_size=pair_chunk_size)

    assert_allclose(chunked, reference, atol=1e-12)


def test_open_shell_packed_contraction_matches_dense_reference():
    """Full packed-density TEI contraction must match the dense path (UHF).

    This anchors the end-to-end two-electron gradient for an open-shell
    reference, downstream of the packed density, so the ERI-side contraction
    is pinned together with the spin expansion.
    """
    hf = cached_backend_hf("pyscf", "cn_sto3g", conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)
    provider = hf.gradient_provider

    g2 = _random_tpdm(refstate)
    g2_ao_1, g2_ao_2 = g2.to_ao_basis()
    g2_ao_1, g2_ao_2 = g2_ao_1.to_ndarray(), g2_ao_2.to_ndarray()

    nao = g2_ao_1.shape[0]
    dummy = np.zeros((nao, nao))
    full = provider.correlated_gradient(dummy, dummy, g2_ao_1, g2_ao_2)

    pair_density = TwoParticleDensityMatrix.ao_pair_density_from_dense(
        g2_ao_1, g2_ao_2
    )
    packed_tei = provider._contract_tei_with_packed_density(
        pair_density, shell_chunk_size=2
    )

    assert_allclose(packed_tei, full.two_electron, atol=1e-10)


# ---------------------------------------------------------------------------
# export_block: dense sub-block / spin-subblock tensor view (C++)
#
# These pin the new libadcc Tensor.export_block primitive that the direct
# gradient transform will consume.  export_block must agree with slicing the
# full dense to_ndarray() export for arbitrary ranges and, in particular, for
# spin sub-blocks (the alpha/beta partition along each axis).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system", ["h2o_sto3g", "cn_sto3g"])
@pytest.mark.parametrize("block", [b.oooo, b.oovv, b.ovov])
def test_export_block_matches_to_ndarray_slices(system, block):
    hf = cached_backend_hf("pyscf", system, conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)

    g2 = _random_tpdm(refstate, blocks=(block,))
    tensor = g2[block]
    full = tensor.to_ndarray()
    shape = full.shape

    # Full range must reproduce to_ndarray exactly.
    eb_full = tensor.export_block([0] * len(shape), list(shape))
    assert_allclose(eb_full, full, atol=1e-12)

    # A handful of random sub-ranges.
    rng = np.random.default_rng(42)
    for _ in range(8):
        start = [int(rng.integers(0, s)) for s in shape]
        end = [int(rng.integers(st + 1, s + 1)) for st, s in zip(start, shape)]
        eb = tensor.export_block(start, end)
        ref_slice = full[tuple(slice(a, b_) for a, b_ in zip(start, end))]
        assert_allclose(eb, ref_slice, atol=1e-12)


def test_export_block_spin_subblocks_match():
    """Every (alpha/beta)^4 spin sub-block must match the to_ndarray slice.

    This is the key use case for the direct gradient transform: extracting a
    single spin sub-block (e.g. abab) of a 4-index TPDM block without
    materialising the full dense tensor.  Uses an open-shell reference so the
    alpha and beta ranges differ in size.
    """
    from itertools import product
    from adcc.MoSpaces import split_spaces

    hf = cached_backend_hf("pyscf", "cn_sto3g", conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)
    ms = refstate.mospaces

    for block in (b.oooo, b.oovv, b.ovov):
        g2 = _random_tpdm(refstate, blocks=(block,))
        tensor = g2[block]
        full = tensor.to_ndarray()
        spaces = split_spaces(block)
        splits = [(ms.n_orbs_alpha(sp), ms.n_orbs(sp)) for sp in spaces]

        for combo in product([0, 1], repeat=4):
            start, end = [], []
            for (na, ntot), spin in zip(splits, combo):
                start.append(0 if spin == 0 else na)
                end.append(na if spin == 0 else ntot)
            eb = tensor.export_block(start, end)
            ref_slice = full[tuple(slice(a, b_) for a, b_ in zip(start, end))]
            assert_allclose(eb, ref_slice, atol=1e-12,
                            err_msg=f"{block} spin {combo} mismatch")


def test_export_block_validates_arguments():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)
    g2 = _random_tpdm(refstate, blocks=(b.oovv,))
    tensor = g2[b.oovv]
    shape = tensor.to_ndarray().shape

    with pytest.raises(Exception):
        tensor.export_block([0, 0, 0], list(shape))  # wrong ndim
    with pytest.raises(Exception):
        # end exceeds shape
        tensor.export_block([0, 0, 0, 0],
                            [shape[0] + 1, shape[1], shape[2], shape[3]])


# ---------------------------------------------------------------------------
# Out-of-core (HDF5) packed-density storage and API validation
#
# These pin the new tei_pair_density_storage="hdf5" path (equivalence with the
# in-memory path and scratch-file cleanup) and the user-facing validation
# contracts of the new keyword arguments.
# ---------------------------------------------------------------------------


def test_direct_gradient_hdf5_matches_memory(tmp_path):
    """The out-of-core HDF5 storage path must match the in-memory path.

    Exercises the h5py-dataset writes inside ``to_ao_pair_density`` (the
    ``out[...] = 0`` reset and the ``out[:, :, start:stop] += chunk``
    read-modify-write), which behave differently for an HDF5 dataset than for
    a plain ndarray.
    """
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))

    memory = adcc.nuclear_gradient(
        mp, tei_contraction="direct", tei_pair_density_storage="memory"
    )
    hdf5 = adcc.nuclear_gradient(
        mp, tei_contraction="direct", tei_pair_density_storage="hdf5",
        tei_pair_chunk_size=3
    )

    assert_allclose(hdf5.total, memory.total, atol=1e-10)
    assert_allclose(hdf5.components.two_electron,
                    memory.components.two_electron, atol=1e-10)


def test_direct_gradient_hdf5_removes_scratch_file(tmp_path):
    """The temporary HDF5 scratch file must be removed after contraction."""
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))
    provider = hf.gradient_provider

    g1_ao, w_ao, g2_total = _direct_gradient_inputs(mp)
    before = set(glob.glob(os.path.join(str(tmp_path), "adcc_pyscf_gradient_*")))
    provider.correlated_gradient_direct(
        g1_ao, w_ao, g2_total, refstate=adcc.ReferenceState(hf),
        pair_density_storage="hdf5", scratch_directory=str(tmp_path)
    )
    after = set(glob.glob(os.path.join(str(tmp_path), "adcc_pyscf_gradient_*")))
    assert before == after, "scratch HDF5 file was not cleaned up"


def test_direct_gradient_hdf5_default_pair_chunk_is_bounded():
    """The hdf5 path must not silently use the full npair working buffer.

    Out-of-core storage only offloads the output density; if the in-memory
    working chunk defaulted to the full ``npair`` it would defeat the memory
    bound.  The hdf5 path therefore selects a bounded default chunk size.
    """
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))
    provider = hf.gradient_provider
    g1_ao, w_ao, g2_total = _direct_gradient_inputs(mp)

    nao = provider.mol.nao_nr()
    npair = nao * (nao + 1) // 2

    captured = {}
    original = g2_total.to_ao_pair_density

    def spy(refstate_or_coefficients=None, pair_chunk_size=None, out=None):
        captured["pair_chunk_size"] = pair_chunk_size
        return original(refstate_or_coefficients,
                        pair_chunk_size=pair_chunk_size, out=out)

    g2_total.to_ao_pair_density = spy
    provider.correlated_gradient_direct(
        g1_ao, w_ao, g2_total, refstate=adcc.ReferenceState(hf),
        pair_density_storage="hdf5"
    )
    assert captured["pair_chunk_size"] is not None
    assert captured["pair_chunk_size"] <= npair


def test_nuclear_gradient_rejects_invalid_tei_contraction():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))
    with pytest.raises(ValueError, match="Invalid tei_contraction"):
        adcc.nuclear_gradient(mp, tei_contraction="bogus")


def test_correlated_gradient_direct_rejects_invalid_storage():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))
    provider = hf.gradient_provider
    g1_ao, w_ao, g2_total = _direct_gradient_inputs(mp)
    with pytest.raises(ValueError, match="pair_density_storage"):
        provider.correlated_gradient_direct(
            g1_ao, w_ao, g2_total, refstate=adcc.ReferenceState(hf),
            pair_density_storage="bogus"
        )


def test_contract_tei_rejects_non_positive_shell_chunk():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    provider = hf.gradient_provider
    nao = provider.mol.nao_nr()
    npair = nao * (nao + 1) // 2
    pair_density = np.zeros((nao, nao, npair))
    with pytest.raises(ValueError, match="shell_chunk_size"):
        provider._contract_tei_with_packed_density(
            pair_density, shell_chunk_size=0
        )


def test_to_ao_pair_density_rejects_non_positive_pair_chunk():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)
    g2 = _random_tpdm(refstate)
    with pytest.raises(ValueError, match="pair_chunk_size"):
        g2.to_ao_pair_density(refstate, pair_chunk_size=0)


def test_to_ao_pair_density_rejects_bad_output_shape():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    refstate = adcc.ReferenceState(hf)
    g2 = _random_tpdm(refstate)
    with pytest.raises(ValueError, match="Invalid output shape"):
        g2.to_ao_pair_density(refstate, out=np.zeros((1, 1, 1)))
