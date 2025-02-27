#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import adcc
import pytest
import numpy as np
from numpy.testing import assert_allclose

from adcc.AdcMatrix import AdcExtraTerm, AdcMatrixProjected, AdcMatrixShifted
from adcc.Intermediates import Intermediates
from adcc.adc_pp.matrix import AdcBlock

from .projection_test import (
    assert_nonzero_blocks, construct_nonzero_blocks, assert_equal_symmetry
)
from .testdata_cache import testdata_cache
from . import testcases


h2o_sto3g = testcases.get_by_filename("h2o_sto3g").pop()
methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]


# Distinct implementations of the matrix equations only exist for the cases
# "gen" and "cvs".
@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("case", ["gen", "cvs"])
@pytest.mark.parametrize("system", ["h2o_sto3g", "cn_sto3g"])
class TestAdcMatrix:
    def load_matrix_data(self, system: str, case: str, method: str) -> dict:
        refdata = testdata_cache.adcc_data(
            system=system, method=method, case=case
        )
        return refdata["matrix"]

    def construct_matrix(self, system: str, case: str,
                         method: str) -> adcc.AdcMatrix:
        # build a matrix from the cached reference state
        refstate = testdata_cache.refstate(system, case)
        if "cvs" in case and "cvs" not in method:
            method = f"cvs-{method}"
        return adcc.AdcMatrix(method, refstate)

    def construct_trial_vec(self, system: str, case: str, method: str, kind: str):
        matdata = self.load_matrix_data(system, case, method)
        states = testdata_cache.adcc_states(
            system, method=method, kind=kind, case=case
        )
        blocks = states.matrix.axis_blocks
        out = states.excitation_vector[0].copy()
        out[blocks[0]].set_from_ndarray(matdata["random_singles"])
        if len(blocks) > 1:
            out[blocks[1]].set_from_ndarray(matdata["random_doubles"])
        return out

    def test_diagonal(self, system: str, case: str, method: str):
        matdata = self.load_matrix_data(system, case, method)
        matrix = self.construct_matrix(system, case, method)
        blocks = matrix.axis_blocks

        diag_s = matrix.diagonal()[blocks[0]]
        assert_allclose(matdata["diagonal_singles"], diag_s.to_ndarray(),
                        rtol=1e-10, atol=1e-12)

        if len(blocks) > 1:
            diag_d = matrix.diagonal()[blocks[1]]
            assert_allclose(matdata["diagonal_doubles"], diag_d.to_ndarray(),
                            rtol=1e-10, atol=1e-12)

    def test_matvec(self, system: str, case: str, method: str):
        matdata = self.load_matrix_data(system, case, method)
        matrix = self.construct_matrix(system, case, method)
        # the matrix data is only dumped once and not per kind
        # -> singlet/any for PP-ADC
        if matrix.reference_state.restricted:
            if matrix.method.adc_type == "pp":
                kind = "singlet"
            else:
                raise ValueError(f"Unknown adc type {matrix.method.adc_type}.")
        else:
            kind = "any"  # we don't do the test for spin flip
        trial_vec = self.construct_trial_vec(system, case, method, kind)
        result = matrix @ trial_vec
        assert_allclose(matdata["matvec_singles"], result.ph.to_ndarray(),
                        rtol=1e-10, atol=1e-12)
        if "matvec_doubles" in matdata:
            assert_allclose(matdata["matvec_doubles"], result.pphh.to_ndarray(),
                            rtol=1e-10, atol=1e-12)

    def test_compute_block(self, system: str, case: str, method: str):
        matdata = self.load_matrix_data(system, case, method)
        matrix = self.construct_matrix(system, case, method)
        # matrix data is only dumped once and not per kind
        # -> singlet/any for PP-ADC
        if matrix.reference_state.restricted:
            if matrix.method.adc_type == "pp":
                kind = "singlet"
            else:
                raise ValueError(f"Unknwon adc type {matrix.method.adc_type}.")
        else:
            kind = "any"  # we don't do the test for spin flip
        trial_vec = self.construct_trial_vec(
            system, case=case, method=method, kind=kind
        )
        blocks = matrix.axis_blocks
        for b1, i1 in [("s", 0), ("d", 1)][:len(blocks)]:
            for b2, i2 in [("s", 0), ("d", 1)][:len(blocks)]:
                res = matrix.block_apply(
                    f"{blocks[i1]}_{blocks[i2]}", trial_vec[blocks[i2]]
                )
                assert_allclose(
                    matdata[f"result_{b1}{b2}"], res.to_ndarray(), rtol=1e-10,
                    atol=1e-12
                )


class TestAdcMatrixInterface:
    @pytest.mark.parametrize("method", methods)
    @pytest.mark.parametrize("case", h2o_sto3g.cases)
    @pytest.mark.parametrize("system", ["h2o_sto3g"])
    def test_properties(self, system: str, case: str, method: str):
        reference_state = testdata_cache.refstate(system=system, case=case)
        ground_state = adcc.LazyMp(reference_state)
        if "cvs" in case and "cvs" not in method:
            method = f"cvs-{method}"
        matrix = adcc.AdcMatrix(method, ground_state)

        assert matrix.ndim == 2
        assert matrix.is_core_valence_separated == ("cvs" in case)
        # check that the blocks are correct
        blocks = matrix.axis_blocks
        if matrix.method.adc_type == "pp":
            assert blocks == ["ph", "pphh", "ppphhh"][:matrix.method.level // 2 + 1]
        else:
            raise NotImplementedError(f"Unknown adc type {matrix.method.adc_type}.")
        assert sorted(matrix.axis_spaces.keys(), key=len) == blocks
        assert sorted(matrix.axis_lengths.keys(), key=len) == blocks
        # check that the spaces for each block are correct.
        # -> build the spaces for each block
        spaces = []
        for block in blocks:
            assert all(sp in "ph" for sp in block)
            block_spaces = ["o1" if sp == "h" else "v1" for sp in block]
            if "cvs" in case:  # replace last o1 -> o2
                block_spaces.reverse()
                block_spaces[block_spaces.index("o1")] = "o2"
                block_spaces.reverse()
            spaces.append(sorted(block_spaces))
        for block, space in zip(blocks, spaces):
            assert matrix.axis_spaces[block] == space
        # check that the sizes of the matrix and for each block are correct
        # -> compute the sizes for the case starting from the "gen" sizes
        sizes = {"o1": 10, "v1": 4}
        if "cvs" in case:
            sizes["o2"] = 2 * h2o_sto3g.core_orbitals  # alpha and beta
            sizes["o1"] -= 2 * h2o_sto3g.core_orbitals  # alpha and beta
        if "fc" in case:
            sizes["o1"] -= 2 * h2o_sto3g.frozen_core  # alpha and beta
        if "fv" in case:
            sizes["v1"] -= 2 * h2o_sto3g.frozen_virtual  # alpha and beta
        matrix_size = 0  # the sum of all blocks
        for block, space in zip(blocks, spaces):
            size = 1
            for sp in space:
                size *= sizes[sp]
            assert matrix.axis_lengths[block] == size
            matrix_size += size
        assert matrix.shape == (matrix_size, matrix_size)
        assert len(matrix) == matrix_size

        assert matrix.reference_state == reference_state
        assert matrix.mospaces == reference_state.mospaces
        assert isinstance(matrix.timer, adcc.timings.Timer)

    def test_intermediates_adc2(self):
        ground_state = adcc.LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        matrix = adcc.AdcMatrix("adc2", ground_state)
        assert isinstance(matrix.intermediates, Intermediates)
        intermediates = Intermediates(ground_state)
        matrix.intermediates = intermediates
        assert matrix.intermediates == intermediates

    def test_matvec_adc2(self):
        ground_state = adcc.LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        matrix = adcc.AdcMatrix("adc2", ground_state)

        vectors = [adcc.guess_zero(matrix) for _ in range(3)]
        for vec in vectors:
            vec.set_random()
        v, w, x = vectors

        # Compute references:
        refv = matrix.matvec(v)
        refw = matrix.matvec(w)
        refx = matrix.matvec(x)

        # @ operator (1 vector)
        resv = matrix @ v
        diffv = refv - resv
        assert diffv.ph.dot(diffv.ph) < 1e-12
        assert diffv.pphh.dot(diffv.pphh) < 1e-12

        # @ operator (multiple vectors)
        resv, resw, resx = matrix @ [v, w, x]
        diffs = [refv - resv, refw - resw, refx - resx]
        for i in range(3):
            assert diffs[i].ph.dot(diffs[i].ph) < 1e-12
            assert diffs[i].pphh.dot(diffs[i].pphh) < 1e-12

        # compute matvec
        resv = matrix.matvec(v)
        diffv = refv - resv
        assert diffv.ph.dot(diffv.ph) < 1e-12
        assert diffv.pphh.dot(diffv.pphh) < 1e-12

        resv = matrix.rmatvec(v)
        diffv = refv - resv
        assert diffv.ph.dot(diffv.ph) < 1e-12
        assert diffv.pphh.dot(diffv.pphh) < 1e-12

        # Test apply
        resv.ph = matrix.block_apply("ph_ph", v.ph)
        resv.ph += matrix.block_apply("ph_pphh", v.pphh)
        refv = matrix.matvec(v)
        diffv = resv.ph - refv.ph
        assert diffv.dot(diffv) < 1e-12

    def test_extra_term(self):
        ground_state = adcc.LazyMp(testdata_cache.refstate("h2o_sto3g", "gen"))
        matrix_adc1 = adcc.AdcMatrix("adc1", ground_state)
        with pytest.raises(TypeError):
            matrix_adc1 += 42
        matrix = adcc.AdcMatrix("adc2", ground_state)

        with pytest.raises(TypeError):
            adcc.AdcMatrix("adc2", ground_state,
                           diagonal_precomputed=42)
        with pytest.raises(ValueError):
            adcc.AdcMatrix("adc2", ground_state,
                           diagonal_precomputed=matrix.diagonal() + 42)
        with pytest.raises(TypeError):
            AdcExtraTerm(matrix, "fail")
        with pytest.raises(TypeError):
            AdcExtraTerm(matrix, {"fail": "not_callable"})

        shift = -0.3
        shifted = AdcMatrixShifted(matrix, shift)
        # TODO: need to use AmplitudeVector to differentiate between
        # diagonals for ph and pphh
        # if we just pass numbers, i.e., shift
        # we get 2*shift on the diagonal
        ones = matrix.diagonal().ones_like()

        def __shift_ph(hf, mp, intermediates):
            def apply(invec):
                return adcc.AmplitudeVector(ph=shift * invec.ph)
            diag = adcc.AmplitudeVector(ph=shift * ones.ph)
            return AdcBlock(apply, diag)

        def __shift_pphh(hf, mp, intermediates):
            def apply(invec):
                return adcc.AmplitudeVector(pphh=shift * invec.pphh)
            diag = adcc.AmplitudeVector(pphh=shift * ones.pphh)
            return AdcBlock(apply, diag)
        extra = AdcExtraTerm(
            matrix, {'ph_ph': __shift_ph, 'pphh_pphh': __shift_pphh}
        )
        # cannot add to 'pphh_pphh' in ADC(1) matrix
        with pytest.raises(ValueError):
            matrix_adc1 += extra

        shifted_2 = matrix + extra
        shifted_3 = extra + matrix
        for manual in [shifted_2, shifted_3]:
            assert_allclose(
                shifted.diagonal().ph.to_ndarray(),
                manual.diagonal().ph.to_ndarray(),
                atol=1e-12
            )
            assert_allclose(
                shifted.diagonal().pphh.to_ndarray(),
                manual.diagonal().pphh.to_ndarray(),
                atol=1e-12
            )
            vec = adcc.guess_zero(matrix)
            vec.set_random()
            ref = shifted @ vec
            ret = manual @ vec
            diff_s = ref.ph - ret.ph
            diff_d = ref.pphh - ret.pphh
            assert np.max(np.abs(diff_s.to_ndarray())) < 1e-12
            assert np.max(np.abs(diff_d.to_ndarray())) < 1e-12


@pytest.mark.parametrize("system", ["h2o_sto3g", "cn_sto3g"])
class TestAdcMatrixShifted:
    def construct_matrices(self, system, shift):
        reference_state = testdata_cache.refstate(system, case="gen")
        ground_state = adcc.LazyMp(reference_state)
        matrix = adcc.AdcMatrix("adc3", ground_state)
        shifted = AdcMatrixShifted(matrix, shift)
        return matrix, shifted

    def test_diagonal(self, system: str):
        shift = -0.3
        matrix, shifted = self.construct_matrices(system, shift)

        for block in ("ph", "pphh"):
            odiag = matrix.diagonal()[block].to_ndarray()
            sdiag = shifted.diagonal()[block].to_ndarray()
            assert np.max(np.abs(sdiag - shift - odiag)) < 1e-12

    def test_matmul(self, system: str):
        shift = -0.3
        matrix, shifted = self.construct_matrices(system, shift)

        vec = adcc.guess_zero(matrix)
        vec.set_random()

        ores = matrix @ vec
        sres = shifted @ vec

        assert ores.ph.describe_symmetry() == sres.ph.describe_symmetry()
        assert ores.pphh.describe_symmetry() == sres.pphh.describe_symmetry()

        diff_s = sres.ph - ores.ph - shift * vec.ph
        diff_d = sres.pphh - ores.pphh - shift * vec.pphh
        assert np.max(np.abs(diff_s.to_ndarray())) < 1e-12
        assert np.max(np.abs(diff_d.to_ndarray())) < 1e-12

    # TODO Test block_view, block_apply


@pytest.mark.parametrize("system", ["h2o_sto3g", "cn_sto3g"])
class TestAdcMatrixProjected:
    def construct_matrices(self, system, n_core: int, n_virt: int):
        reference_state = testdata_cache.refstate(system, "gen")
        ground_state = adcc.LazyMp(reference_state)
        matrix = adcc.AdcMatrix("adc3", ground_state)

        out = construct_nonzero_blocks(reference_state.mospaces, n_core, n_virt)
        spaces, nonzero_blocks = out

        excitation_blocks = spaces["ph"] + spaces["pphh"]
        projected = AdcMatrixProjected(matrix, excitation_blocks,
                                       core_orbitals=n_core,
                                       outer_virtuals=n_virt)
        return matrix, projected, nonzero_blocks

    def test_diagonal(self, system: str):
        out = self.construct_matrices(system, n_core=2, n_virt=1)
        matrix, projected, nonzeros = out

        for block in ("ph", "pphh"):
            odiag = matrix.diagonal()[block]
            pdiag = projected.diagonal()[block]
            assert_nonzero_blocks(odiag, pdiag, nonzeros[block], zero_value=100000)
            # TODO Manually verified to be identical, however, string parsing
            #      of the describe_symmetry output is not super reliable and so this
            #      test does not pass in CI.
            # assert_equal_symmetry(odiag, pdiag)

    def test_matmul(self, system):
        out = self.construct_matrices(system, n_core=1, n_virt=1)
        matrix, projected, nonzeros = out

        spin_block_symmetrisation = "none"
        if "h2o" in system:
            spin_block_symmetrisation = "symmetric"
        vec = adcc.guess_zero(matrix,
                              spin_block_symmetrisation=spin_block_symmetrisation)
        vec.set_random()
        pvec = projected.apply_projection(vec.copy())  # only apply projection

        pres = projected @ vec
        ores = matrix @ pvec
        res_for_sym = matrix @ vec

        assert_equal_symmetry(res_for_sym.ph, pres.ph)
        assert_equal_symmetry(res_for_sym.pphh, pres.pphh)
        assert_nonzero_blocks(ores.ph, pres.ph, nonzeros["ph"], tol=1e-14)
        assert_nonzero_blocks(ores.pphh, pres.pphh, nonzeros["pphh"], tol=1e-14)

    # TODO Test block_view, block_apply
