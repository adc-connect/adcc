#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
"""Fast unit tests for the paired-state MECP/MECI scanner and penalty objective.

These mirror the gating style of ``geomopt_scanner_test.py`` (PySCF-only, no
geomeTRIC dependency) and cover: the pure penalty math (finite-difference
gradient consistency, the Levine-Coe-Martinez formula, degeneracy limits and
parameter validation), the :class:`MECPObjective` geomeTRIC bridge contract, the
paired-engine mechanics (two distinct surfaces from one SCF + one ADC/MMP), the
distinctness guard, and paired-target normalisation/validation.
"""
import types

import numpy as np
import pytest
from numpy.testing import assert_allclose

import adcc
import adcc.backends
from adcc.gradients.mecp import mecp_penalty, MECPObjective, _N_STATES2
from adcc.gradients.paired_scanner import (
    PairedExcitedStateTarget,
    PairedGroundExcitedStateTarget,
)
from adcc.gradients.scanner import _TrackingDescriptor


pytestmark = pytest.mark.skipif(
    "pyscf" not in adcc.backends.available(), reason="PySCF not found."
)


def _h2o_scf():
    from pyscf import gto, scf
    mol = gto.M(
        atom="""
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
        """,
        basis="sto-3g", unit="Bohr", symmetry=False,
        verbose=0, parse_arg=False,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-9
    return mf


def _mecp_scf():
    """A small twisted ethylene-ish scratch SCF with >=2 singlet roots."""
    from pyscf import gto, scf
    mol = gto.M(
        atom="""
        C 0.0  0.0  0.0
        C 1.3  0.0  0.0
        H 0.0  1.0  0.0
        H 0.0 -1.0  0.0
        H 1.3  1.0  0.0
        H 1.3 -1.0  0.0
        """,
        basis="sto-3g", unit="Angstrom", symmetry=False,
        verbose=0, parse_arg=False,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    mf.conv_tol_grad = 1e-9
    return mf


# ---------------------------------------------------------------------------
# Penalty math (controlled energies/gradients) -- no SCF needed.
# ---------------------------------------------------------------------------

def test_n_states2_is_one_for_two_state_pair():
    assert _N_STATES2 == 1


def test_penalty_gradient_matches_finite_difference():
    rng = np.random.default_rng(42)
    e_lo, e_hi = -1.3, -1.1
    g_lo = rng.normal(size=(2, 3))
    g_hi = rng.normal(size=(2, 3))

    energy, gradient = mecp_penalty(e_lo, g_lo, e_hi, g_hi,
                                    sigma=3.5, alpha=0.025)

    h = 1e-7

    def efun(el, eh):
        return mecp_penalty(el, g_lo, eh, g_hi, sigma=3.5, alpha=0.025)[0]

    d_lo = (efun(e_lo + h, e_hi) - efun(e_lo - h, e_hi)) / (2 * h)
    d_hi = (efun(e_lo, e_hi + h) - efun(e_lo, e_hi - h)) / (2 * h)
    gradient_fd = d_lo * g_lo + d_hi * g_hi
    assert gradient.shape == g_lo.shape
    assert_allclose(gradient, gradient_fd, atol=1e-6)


def test_penalty_energy_matches_hand_computed_lcm_formula():
    # EAvg = (-1.3 + -1.1)/2 = -1.2; EDif = 0.2; with sigma=3.5, alpha=0.025,
    # n_states2=1: EPen = 3.5 * 0.04 / (0.225) = 0.6222..., E = EAvg + EPen.
    e_lo, e_hi = -1.3, -1.1
    g_lo = np.zeros((1, 3))
    g_hi = np.zeros((1, 3))
    energy, _ = mecp_penalty(e_lo, g_lo, e_hi, g_hi, sigma=3.5, alpha=0.025)
    e_avg = 0.5 * (e_lo + e_hi)
    e_dif = e_hi - e_lo
    e_pen = 3.5 * e_dif ** 2 / ((e_dif + 0.025) * 1)
    assert energy == pytest.approx(e_avg + e_pen)


def test_penalty_penalty_vanishes_at_exact_degeneracy():
    g_lo = np.array([[1.0, 0.0, 0.0]])
    g_hi = np.array([[0.0, 2.0, 0.0]])
    e = -1.2
    energy, gradient = mecp_penalty(e, g_lo, e, g_hi, sigma=3.5, alpha=0.025)
    # At EDif == 0 the penalty and its gradient vanish, so the objective reduces
    # to the average surface.
    assert energy == pytest.approx(e)
    assert_allclose(gradient, 0.5 * (g_lo + g_hi))


def test_penalty_objective_is_flat_when_both_gradients_zero_at_degeneracy():
    e = -1.0
    zero = np.zeros((3, 3))
    energy, gradient = mecp_penalty(e, zero, e, zero)
    assert energy == pytest.approx(e)
    assert_allclose(gradient, zero)


def test_penalty_alpha_zero_is_raw_squared_difference_mode():
    e_lo, e_hi = -1.3, -1.1
    g_lo = np.zeros((1, 3))
    g_hi = np.array([[1.0, 0.0, 0.0]])
    # alpha == 0: the penalty becomes sigma * EDif (energy-difference objective)
    # so the penalty's energy contribution is sigma * EDif and the gradient
    # contribution is sigma * GDif about the upper-lower split.
    energy, gradient = mecp_penalty(e_lo, g_lo, e_hi, g_hi, sigma=2.0, alpha=0.0)
    e_dif = e_hi - e_lo
    assert energy == pytest.approx(0.5 * (e_lo + e_hi) + 2.0 * e_dif)
    assert_allclose(gradient, 0.5 * (g_lo + g_hi) + 2.0 * (g_hi - g_lo))


def test_penalty_parameters_are_validated():
    with pytest.raises(ValueError, match="sigma"):
        mecp_penalty(0.0, np.zeros(3), 1.0, np.zeros(3), sigma=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        mecp_penalty(0.0, np.zeros(3), 1.0, np.zeros(3), alpha=-0.1)


def test_penalty_tolerates_unsorted_input_without_sort_inversion():
    # Passing e_upper < e_lower keeps e_dif = e_upper - e_lower (negative); the
    # squared terms stay real and finite, the objective stays finite.  This only
    # documents that the function tolerates unsorted input; callers (the paired
    # scanner) always feed energy-sorted surfaces.
    energy, gradient = mecp_penalty(-1.1, np.zeros(3), -1.3, np.zeros(3))
    assert np.isfinite(energy)
    assert_allclose(gradient, np.zeros(3))


# ---------------------------------------------------------------------------
# MECPObjective bridge contract.
# ---------------------------------------------------------------------------

def test_mecp_objective_returns_combined_energy_and_gradient():
    g_lo = np.ones((2, 3))
    g_hi = 2 * np.ones((2, 3))
    energy_direct, gradient_direct = mecp_penalty(
        -1.3, g_lo, -1.1, g_hi, sigma=3.5, alpha=0.025,
    )

    class _FakeScanner:
        def __call__(self, coords):
            return ((-1.3, g_lo), (-1.1, g_hi))

    obj = MECPObjective(_FakeScanner(), sigma=3.5, alpha=0.025)
    energy, gradient = obj("coords-ignored")
    assert energy == pytest.approx(energy_direct)
    assert_allclose(gradient, gradient_direct)
    assert obj.last_pair[0][0] == pytest.approx(-1.3)
    assert obj.last_pair[1][0] == pytest.approx(-1.1)
    assert obj.last_energy == pytest.approx(energy_direct)


def test_mecp_objective_calc_new_honours_geometric_dict_contract():
    class _FakeScanner:
        def __call__(self, coords):
            return ((-1.3, np.ones((3, 3))), (-1.1, np.zeros((3, 3))))

    obj = MECPObjective(_FakeScanner())
    result = obj.calc_new(np.zeros(9))
    assert set(result) == {"energy", "gradient"}
    assert isinstance(result["energy"], float)
    assert result["gradient"].shape == (9,)
    assert result["gradient"].ndim == 1


def test_mecp_objective_accepts_custom_penalty():
    calls = []

    def custom(e0, g0, e1, g1, *, sigma, alpha):
        calls.append((e0, e1, sigma, alpha))
        return e0 + e1, g0 + g1

    class _FakeScanner:
        def __call__(self, coords):
            return ((-1.0, np.zeros(2)), (-2.0, np.ones(2)))

    obj = MECPObjective(_FakeScanner(), sigma=1.0, alpha=0.0, penalty=custom)
    e, g = obj(None)
    assert e == pytest.approx(-3.0)
    assert_allclose(g, np.ones(2))
    assert calls and calls[0] == (-1.0, -2.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Paired-engine mechanics (one SCF + one ADC, two surfaces).
# ---------------------------------------------------------------------------

def _assert_independent_match(scanner):
    states = adcc.run_adc(scanner.last_scf,
                          **scanner.paired_target.kwargs())
    return states


def test_paired_excited_scanner_returns_distinct_energies_and_gradients():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="index", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )

    (e_lo, g_lo), (e_hi, g_hi) = scanner(scanner.initial_coords)

    assert e_lo <= e_hi
    assert g_lo.shape == (3, 3)
    assert g_hi.shape == (3, 3)
    assert np.all(np.isfinite(g_lo)) and np.all(np.isfinite(g_hi))

    # Each channel matches an independent nuclear_gradient eval on the selected
    # excitation.  The paired call selected roots (0, 1) (follow=="index" first
    # call); energies/gradients must agree elementwise.
    states = adcc.run_adc(scanner.last_scf, method="adc2", n_singlets=3,
                          conv_tol=1e-9)
    energies_lo = states.excitations[0].total_energy
    energies_hi = states.excitations[1].total_energy
    assert e_lo == pytest.approx(energies_lo, abs=1e-7)
    assert e_hi == pytest.approx(energies_hi, abs=1e-7)
    grad_lo = adcc.nuclear_gradient(states.excitations[0],
                                    eri_contraction="full_ao").total
    grad_hi = adcc.nuclear_gradient(states.excitations[1],
                                    eri_contraction="full_ao").total
    assert_allclose(g_lo, grad_lo, atol=1e-7)
    assert_allclose(g_hi, grad_hi, atol=1e-7)


def test_paired_calc_new_returns_both_surfaces():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="index", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )

    result = scanner.calc_new(scanner.initial_coords.ravel())
    assert set(result) >= {"energies", "gradients", "energy_lower",
                           "energy_upper", "gradient_lower", "gradient_upper"}
    assert result["energies"].shape == (2,)
    assert result["gradients"].shape == (2, 3, 3)
    assert result["energy_lower"] <= result["energy_upper"]
    assert result["gradient_lower"].shape == (9,)
    assert result["gradient_upper"].shape == (9,)


def test_paired_excited_overlap_tracking_seeds_and_tracks_both_slots():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="overlap", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )

    # First call seeds both slots from positional indices; no tracking yet.
    scanner(scanner.initial_coords)
    assert scanner.last_trackings == (None, None)
    assert scanner.previous_indices == (0, 1)

    # Second call at the same geometry: both slots track via overlap.
    scanner(scanner.initial_coords)
    slot0, slot1 = scanner.last_trackings
    assert slot0 is not None and slot1 is not None
    assert slot0.index != slot1.index  # distinctness guard
    assert slot0.best_score == pytest.approx(1.0, abs=1e-3)
    assert slot1.best_score == pytest.approx(1.0, abs=1e-3)
    assert scanner.previous_indices == (slot0.index, slot1.index)


def test_paired_mecp_scanner_returns_ground_and_excited_surfaces():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", lower="mp2", upper=0, n_singlets=3,
        follow="index", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )

    (e_lo, g_lo), (e_hi, g_hi) = scanner(scanner.initial_coords)

    assert e_lo <= e_hi
    assert g_lo.shape == (3, 3) and g_hi.shape == (3, 3)

    # Lower channel is the MP2 ground state: matches LazyMp + nuclear_gradient.
    mp = adcc.LazyMp(adcc.ReferenceState(scanner.last_scf))
    assert e_lo == pytest.approx(mp.energy(2), abs=1e-7)
    exp = adcc.nuclear_gradient(mp, eri_contraction="full_ao").total
    assert_allclose(g_lo, exp, atol=1e-7)

    # Upper channel is the excited root: only the excited side is tracked.
    assert scanner.last_excitations[0] is None
    assert scanner.last_excitations[1] is not None
    assert scanner.last_trackings == (None, None)  # index mode, first call


# ---------------------------------------------------------------------------
# Distinctness guard with synthetic near-degenerate candidates.
# ---------------------------------------------------------------------------

class _FakeAoOperator:
    def __init__(self, matrix):
        self._matrix = matrix

    def to_ndarray(self):
        return self._matrix


class _FakeExcitation:
    def __init__(self, index, transition_dm, state_diffdm, omega):
        self.index = index
        self.transition_dm_ao = _FakeAoOperator(transition_dm)
        self.state_diffdm_ao = _FakeAoOperator(state_diffdm)
        self.excitation_energy = omega


def test_distinctness_guard_keeps_slots_apart_under_collapse():
    # Both previous descriptors would, independently, best-match candidate 0.
    # The joint selection must keep the two roots distinct: one slot takes 0,
    # the other takes the next-best candidate.
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 0),  # same seed is fine here
        follow="overlap", conv_tol=1e-9,
    )
    nao = scanner.base_mol.nao

    def projector(i):
        m = np.zeros((nao, nao))
        m[i, i] = 1.0
        return m

    # Both slots previously followed the *same* character (a collapse scenario).
    seed_transition = projector(0)
    seed_diffdm = projector(1)
    shared_descriptor = _TrackingDescriptor(
        mol=scanner.base_mol.copy(),
        transition_dm=seed_transition, state_diffdm=seed_diffdm,
    )

    # Three candidates: index 0 carries the seeded character, indices 1 and 2
    # carry orthogonal characters.  The guard must not let both slots pick 0.
    candidates = [
        _FakeExcitation(0, seed_transition.copy(), seed_diffdm.copy(), 0.20),
        _FakeExcitation(1, projector(2), projector(3), 0.21),
        _FakeExcitation(2, projector(4), projector(5), 0.22),
    ]
    states = types.SimpleNamespace(excitations=candidates)

    scanner.previous_descriptors = (shared_descriptor, shared_descriptor)
    scanner.previous_indices = (0, 1)

    chosen, _descriptors, _trackings = scanner._select_excitations(
        states, scanner.base_mol.copy()
    )

    assert len(set(chosen)) == 2  # distinct
    assert 0 in chosen  # the best-matching candidate is taken once
    assert scanner.last_trackings is not None


def test_distinctness_guard_raises_when_too_few_states():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), follow="index",
    )
    states = types.SimpleNamespace(excitations=[object()])  # only one root
    with pytest.raises(RuntimeError, match="at least two excited states"):
        scanner._select_excitations(states, scanner.base_mol.copy())


def test_paired_scanner_no_excitations_raises():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), follow="index",
    )
    states = types.SimpleNamespace(excitations=[])
    with pytest.raises(RuntimeError, match="did not return any excited states"):
        scanner._select_excitations(states, scanner.base_mol.copy())


# ---------------------------------------------------------------------------
# Paired-target normalisation / validation.
# ---------------------------------------------------------------------------

def test_paired_target_dataclasses_validate_index_arity():
    PairedExcitedStateTarget(method="adc2", state_indices=(0, 1))  # ok
    with pytest.raises(ValueError, match="exactly two"):
        PairedExcitedStateTarget(method="adc2", state_indices=(0, 1, 2))


def test_paired_ground_target_enforces_mp2():
    PairedGroundExcitedStateTarget(method="adc2", state_index=0)  # ok, level=2
    with pytest.raises(NotImplementedError, match="MP2 ground-state"):
        PairedGroundExcitedStateTarget(method="adc2", state_index=0, level=3)


def test_paired_mecp_lower_mp3_raises():
    with pytest.raises(NotImplementedError, match="MP2 ground-state"):
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", lower="mp3", upper=0,
        )


def test_paired_mecp_upper_must_be_int():
    with pytest.raises(TypeError, match="upper must be an int"):
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", lower="mp2", upper="0",
        )


def test_paired_meci_states_wrong_length_raises():
    with pytest.raises(ValueError, match="exactly two"):
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", states=(0, 1, 2),
        )


def test_paired_meci_missing_method_raises():
    with pytest.raises(ValueError, match="ADC method is required"):
        adcc.PairedStateGradientScanner(_h2o_scf(), states=(0, 1))


def test_paired_mecp_requires_method():
    with pytest.raises(ValueError, match="ADC method is required"):
        adcc.PairedStateGradientScanner(_h2o_scf(), lower="mp2", upper=0)


def test_paired_missing_target_raises():
    with pytest.raises(ValueError, match="paired target"):
        adcc.PairedStateGradientScanner(_h2o_scf())


def test_paired_run_adc_kwargs_forwarded_with_native_names():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=2, conv_tol=1e-7,
        follow="index",
    )
    kwargs = scanner.paired_target.kwargs()
    assert kwargs["method"] == "adc2"
    assert kwargs["n_singlets"] == 2
    assert kwargs["conv_tol"] == pytest.approx(1e-7)
    assert "output" not in kwargs


def test_paired_state_index_out_of_range_raises():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 5), n_singlets=2,
        follow="index", conv_tol=1e-9,
    )
    with pytest.raises(ValueError, match="out of range"):
        scanner(scanner.initial_coords)


def test_paired_invalid_follow_raises_eagerly():
    with pytest.raises(ValueError, match="overlap.*index"):
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", states=(0, 1), follow="bogus",
        )


def test_paired_validates_coordinate_shape():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="index", conv_tol=1e-9,
    )
    with pytest.raises(ValueError, match="Expected coordinates"):
        scanner(np.zeros((2, 3)))


def test_paired_requires_pyscf_scf_object():
    with pytest.raises(TypeError, match="PySCF SCF object"):
        adcc.PairedStateGradientScanner(
            _h2o_scf().mol, method="adc2", states=(0, 1),
        )


def test_paired_unconverged_scf_raises():
    from pyscf import scf
    mf = scf.RHF(_h2o_scf().mol)
    mf.max_cycle = 1
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    scanner = adcc.PairedStateGradientScanner(
        mf, method="adc2", states=(0, 1), n_singlets=3, follow="index",
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        scanner(scanner.initial_coords)


# ---------------------------------------------------------------------------
# geomeTRIC oracle cross-check (only when geometric is importable).
#
# This calls geomeTRIC's *real* :class:`geometric.engine.ConicalIntersection`
# engine with fake single-point sub-engines returning controlled energies and
# gradients, then asserts :func:`mecp_penalty` reproduces geomeTRIC's own
# penalty objective and gradient to machine precision.  This is a direct
# cross-check against the implementation that already exists upstream, not a
# re-port of its formula.
# ---------------------------------------------------------------------------

def _geom_conical_intersection(energies, grads, sigma, alpha):
    """Build a live geomeTRIC ConicalIntersection over fake single-point engines.

    Returns the ``(energy, gradient)`` geomeTRIC computes for the given
    controlled per-state energies and gradients.  The fake sub-engines subclass
    :class:`geometric.engine.Engine` and return canned single-point dicts from
    ``calc_new`` so the real penalty machinery in ``ConicalIntersection.calc_new``
    runs end-to-end.
    """
    import tempfile
    from geometric.engine import ConicalIntersection, Engine

    class _FakeMol:
        def __len__(self):
            return 1

    class _FakeEngine(Engine):
        def __init__(self, molecule, energy, gradient):
            self._energy = float(energy)
            self._gradient = np.asarray(gradient, dtype=float)
            super().__init__(molecule)

        def calc_new(self, coords, dirname):
            return {"energy": self._energy,
                    "gradient": self._gradient.copy()}

    mol = _FakeMol()
    engines = [_FakeEngine(mol, e, g) for e, g in zip(energies, grads)]
    ci = ConicalIntersection(mol, engines, sigma, alpha)
    coords = np.zeros(np.asarray(grads[0]).size)
    with tempfile.TemporaryDirectory() as dnm:
        out = ci.calc(coords, dnm)
    return out["energy"], np.asarray(out["gradient"])


def test_penalty_matches_geometric_oracle():
    import importlib.util
    if importlib.util.find_spec("geometric") is None:
        pytest.skip("geometric not installed")
    rng = np.random.default_rng(123)
    energies = [-1.3, -1.08]
    grads = [rng.normal(size=6), rng.normal(size=6)]
    for sigma, alpha in [(3.5, 0.025), (5.0, 0.1), (1.0, 0.0)]:
        e_oracle, g_oracle = _geom_conical_intersection(
            energies, grads, sigma, alpha,
        )
        e_adcc, g_adcc = mecp_penalty(
            energies[0], grads[0], energies[1], grads[1],
            sigma=sigma, alpha=alpha,
        )
        assert e_adcc == pytest.approx(e_oracle, abs=1e-13)
        assert_allclose(g_adcc, g_oracle, atol=1e-13)


def test_penalty_matches_geometric_oracle_at_degeneracy():
    # At exact degeneracy the penalty and its gradient vanish; the oracle must
    # reduce to the average surface, matching mecp_penalty exactly.
    import importlib.util
    if importlib.util.find_spec("geometric") is None:
        pytest.skip("geometric not installed")
    rng = np.random.default_rng(7)
    energies = [-1.2, -1.2]
    grads = [rng.normal(size=6), rng.normal(size=6)]
    e_oracle, g_oracle = _geom_conical_intersection(energies, grads, 3.5, 0.025)
    e_adcc, g_adcc = mecp_penalty(
        energies[0], grads[0], energies[1], grads[1], sigma=3.5, alpha=0.025,
    )
    assert e_adcc == pytest.approx(e_oracle, abs=1e-13)
    assert_allclose(g_adcc, g_oracle, atol=1e-13)
    # Sanity: the oracle itself collapsed to the average at the seam.
    assert e_oracle == pytest.approx(-1.2, abs=1e-13)
    assert_allclose(g_oracle, 0.5 * (np.asarray(grads[0]) + np.asarray(grads[1])))
