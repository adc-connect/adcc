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


@pytest.mark.filterwarnings("ignore::UserWarning")
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


@pytest.mark.filterwarnings("ignore::UserWarning")
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


def test_paired_meci_overlap_follow_emits_warning():
    # follow="overlap" fights the adiabatic energy-ordering that defines a MECI
    # seam and can flip a slot onto a higher root near degeneracy.  The scanner
    # warns (it does not raise -- the user may know what they are doing for
    # well-separated states) and points at follow="index".
    with pytest.warns(UserWarning, match="follow='overlap' is not recommended"):
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
            follow="overlap",
        )


def test_paired_mecp_overlap_follow_does_not_warn():
    # The warning is specific to the excited/excited MECI pair.  A ground/excited
    # MECP only tracks one (excited) root, so follow="overlap" is a legitimate
    # choice there and must not warn.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", lower="mp2", upper=0, n_singlets=3,
            follow="overlap",
        )


def test_paired_meci_index_follow_does_not_warn():
    # The recommended setting for a MECI pair must not warn.
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
            follow="index",
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


# ---------------------------------------------------------------------------
# Exact-degeneracy guard at alpha == 0 (findings 1 & 5) and finiteness guards.
# ---------------------------------------------------------------------------

def test_penalty_alpha_zero_exact_degeneracy_keeps_penalty_gradient():
    # At (alpha == 0, e_dif == 0) the raw energy-difference mode is evaluated in
    # closed form (no division): the penalty energy vanishes, but its gradient
    # tends to the constant sigma * G_dif -- the force that pins the optimiser
    # to the crossing -- which must NOT be dropped at the seam.  The objective
    # gradient is therefore G_avg + sigma * G_dif, the genuine continuous
    # extension, not the bare average surface.
    g_lo = np.array([[1.0, 0.0, 0.0]])
    g_hi = np.array([[0.0, 2.0, 0.0]])
    e = -1.2
    sigma = 2.0
    energy, gradient = mecp_penalty(e, g_lo, e, g_hi, sigma=sigma, alpha=0.0)
    assert energy == pytest.approx(e)  # penalty energy vanishes at the seam
    assert_allclose(gradient, 0.5 * (g_lo + g_hi) + sigma * (g_hi - g_lo))


def test_penalty_alpha_zero_near_degeneracy_is_finite():
    # The alpha == 0 raw mode carries no division, so the whole sub-DBL_MIN
    # underflow regime (E_dif**2 flushing to zero in float64) -- not just the
    # exact E_dif == 0 point -- is well-defined rather than 0.0 / 0.0.
    tiny = 1e-310  # E_dif ** 2 flushes to zero in float64
    energy, gradient = mecp_penalty(
        -1.2, np.zeros(3), -1.2 + tiny, np.zeros(3), sigma=2.0, alpha=0.0,
    )
    assert np.isfinite(energy)
    assert_allclose(gradient, np.zeros(3))

    def efun(off):
        return mecp_penalty(-1.2, np.zeros(3), -1.2 + off, np.zeros(3),
                            sigma=2.0, alpha=0.0)[0]

    # The limit from outside the exact point (alpha == 0) is the average as
    # off -> 0 (the penalty energy sigma * off vanishes).
    assert efun(tiny) == pytest.approx(-1.2, abs=1e-6)


def test_penalty_alpha_zero_underflow_regime_is_finite():
    # Regression for the former ZeroDivisionError gap: with the old short-
    # circuit (abs(E_dif) < 1e-300) a value of E_dif in [1e-300, ~7e-162) still
    # made denom ** 2 underflow to zero while alpha == 0, raising 0.0 / 0.0.
    # The no-division raw mode handles the whole regime.  1e-200 sits squarely
    # in the former gap (E_dif ** 2 flushes to zero, but no division happens).
    tiny = 1e-200
    energy, gradient = mecp_penalty(
        -1.2, np.zeros(3), -1.2 + tiny, np.zeros(3), sigma=2.0, alpha=0.0,
    )
    assert np.isfinite(energy)
    assert_allclose(gradient, np.zeros(3))


def test_penalty_rejects_non_finite_energies():
    g = np.zeros(3)
    with pytest.raises(RuntimeError, match="non-finite"):
        mecp_penalty(float("nan"), g, -1.0, g)
    with pytest.raises(RuntimeError, match="non-finite"):
        mecp_penalty(-1.0, g, float("inf"), g)


def test_penalty_rejects_non_finite_gradients():
    with pytest.raises(RuntimeError, match="non-finite"):
        mecp_penalty(-1.3, np.array([float("nan"), 0.0, 0.0]), -1.1,
                     np.zeros(3))


# ---------------------------------------------------------------------------
# Paired threshold enforcement (finding 3) for MECI and MECP overlap tracking.
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_paired_overlap_tracking_below_min_score_raises():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="overlap", tracking_min_score=2.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    with pytest.raises(RuntimeError, match="below threshold"):
        scanner(scanner.initial_coords)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_paired_overlap_tracking_ambiguous_gap_raises():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="overlap", tracking_min_gap=10.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    with pytest.raises(RuntimeError, match="ambiguous"):
        scanner(scanner.initial_coords)


def test_mecp_overlap_tracking_below_min_score_raises():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", lower="mp2", upper=0, n_singlets=3,
        follow="overlap", tracking_min_score=2.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed (ground never tracked)
    with pytest.raises(RuntimeError, match="below threshold"):
        scanner(scanner.initial_coords)


def test_mecp_overlap_tracking_ambiguous_gap_raises():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", lower="mp2", upper=0, n_singlets=3,
        follow="overlap", tracking_min_gap=10.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    with pytest.raises(RuntimeError, match="ambiguous"):
        scanner(scanner.initial_coords)


# ---------------------------------------------------------------------------
# tracking_min_gap with only two states (finding 4).
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tracking_min_gap_raises_when_only_two_excited_states():
    # n_singlets == 2 leaves no per-channel alternative to compare a gap
    # against; requesting ambiguity detection must surface that clearly.
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=2,
        follow="overlap", tracking_min_gap=0.5, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed (no tracking yet)
    with pytest.raises(RuntimeError, match="at least three computed excited"):
        scanner(scanner.initial_coords)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tracking_min_gap_inert_at_two_states_with_default():
    # With the default tracking_min_gap == 0 the two-state case does not raise;
    # the per-channel gap is reported as inf (no measurable alternative).
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=2,
        follow="overlap", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)
    scanner(scanner.initial_coords)
    slot0, slot1 = scanner.last_trackings
    assert slot0 is not None and slot1 is not None
    assert slot0.gap == float("inf")
    assert slot1.gap == float("inf")
    assert slot0.index != slot1.index


# ---------------------------------------------------------------------------
# MECP excited-root overlap tracking (finding 2) and per-channel diagnostics.
# ---------------------------------------------------------------------------

def test_mecp_overlap_tracking_seeds_and_tracks_excited_root():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", lower="mp2", upper=0, n_singlets=3,
        follow="overlap", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    # Seed call: ground is never tracked; the excited side falls back to its
    # positional seed index because there is no previous descriptor yet.
    scanner(scanner.initial_coords)
    assert scanner.last_trackings == (None, None)
    assert scanner.last_excitations[0] is None
    assert scanner.last_excitations[1] is not None

    # Second call at the same geometry: the excited side tracks via overlap.
    scanner(scanner.initial_coords)
    ground_tr, excited_tr = scanner.last_trackings
    assert ground_tr is None             # ground remains untracked
    assert excited_tr is not None
    assert excited_tr.index == 0         # the seeded S1 character
    assert excited_tr.best_score == pytest.approx(1.0, abs=1e-3)
    assert excited_tr.previous_index == 0
    assert excited_tr.switched is False


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_paired_overlap_tracking_records_per_channel_diagnostics():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="overlap", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    scanner(scanner.initial_coords)  # track both slots via overlap
    slot0, slot1 = scanner.last_trackings
    assert slot0 is not None and slot1 is not None
    assert slot0.index != slot1.index
    # Per-channel diagnostics that catch a silent root flip.
    for slot in (slot0, slot1):
        assert np.isfinite(slot.gap)
        assert slot.gap >= 0.0
    assert slot0.previous_index == 0
    assert slot1.previous_index == 1
    assert slot0.switched is False
    assert slot1.switched is False


# ---------------------------------------------------------------------------
# Energy-sorting reorder branch (finding 8) via synthetic surface injection.
# ---------------------------------------------------------------------------

class _FakeSurface:
    """A non-LazyMp target with a scalar total_energy and (ignored) extras."""

    def __init__(self, total_energy):
        self.total_energy = total_energy


class _FakeGradResult:
    def __init__(self, total):
        self.total = np.asarray(total, dtype=float)


def _patched_paired_scanner(monkeypatch, energies):
    """Build a paired scanner whose __call__ uses fake, injected surfaces.

    ``_run_scf`` and ``_build_target`` are stubbed so the energy path runs on
    controlled ``total_energy`` values, and the gradient eval is stubbed to a
    zero gradient.  This isolates the energy-sorting / finiteness logic of
    ``PairedStateGradientScanner.__call__`` from any SCF/ADC machinery.
    """
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3,
        follow="index", conv_tol=1e-9,
    )
    monkeypatch.setattr(scanner, "_run_scf", lambda coords: object())
    targets = tuple(_FakeSurface(e) for e in energies)
    monkeypatch.setattr(scanner, "_build_target", lambda scfres: targets)
    import adcc.gradients as gradients_mod
    monkeypatch.setattr(
        gradients_mod, "nuclear_gradient",
        lambda t, **kwargs: _FakeGradResult(np.zeros((3, 3))),
    )
    return scanner


def test_paired_call_energy_sorts_and_records_last_pair_order(monkeypatch):
    # Slot 1 carries the lower energy; __call__ must return it as the lower
    # surface while keeping last_energies in slot order.
    e_slot0, e_slot1 = -1.1, -1.3  # slot 1 is lower
    scanner = _patched_paired_scanner(monkeypatch, (e_slot0, e_slot1))
    (e_lo, g_lo), (e_hi, g_hi) = scanner(scanner.initial_coords)
    assert e_lo == pytest.approx(e_slot1)
    assert e_hi == pytest.approx(e_slot0)
    assert scanner.last_pair_order == (1, 0)         # energy-sorted insertion
    assert scanner.last_energies == (e_slot0, e_slot1)  # slot order preserved


def test_paired_call_default_slot_order_when_already_sorted(monkeypatch):
    scanner = _patched_paired_scanner(monkeypatch, (-1.3, -1.1))  # slot0 lower
    (e_lo, _), (e_hi, _) = scanner(scanner.initial_coords)
    assert e_lo == pytest.approx(-1.3)
    assert e_hi == pytest.approx(-1.1)
    assert scanner.last_pair_order == (0, 1)


# ---------------------------------------------------------------------------
# Non-finite surface guard in the scanner (finding 7, paired-side).
# ---------------------------------------------------------------------------

def test_paired_scanner_rejects_non_finite_energy(monkeypatch):
    import adcc.gradients as gradients_mod
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3, follow="index",
    )
    monkeypatch.setattr(scanner, "_run_scf", lambda coords: object())
    monkeypatch.setattr(
        scanner, "_build_target",
        lambda scfres: (_FakeSurface(float("inf")), _FakeSurface(-1.0)),
    )
    monkeypatch.setattr(
        gradients_mod, "nuclear_gradient",
        lambda t, **kwargs: _FakeGradResult(np.zeros((3, 3))),
    )
    with pytest.raises(RuntimeError, match="Non-finite surface result"):
        scanner(scanner.initial_coords)


def test_paired_scanner_rejects_non_finite_gradient(monkeypatch):
    import adcc.gradients as gradients_mod
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=3, follow="index",
    )
    monkeypatch.setattr(scanner, "_run_scf", lambda coords: object())
    monkeypatch.setattr(
        scanner, "_build_target",
        lambda scfres: (_FakeSurface(-1.0), _FakeSurface(-2.0)),
    )
    monkeypatch.setattr(
        gradients_mod, "nuclear_gradient",
        lambda t, **kwargs: _FakeGradResult(np.full((3, 3), float("nan"))),
    )
    with pytest.raises(RuntimeError, match="Non-finite surface result"):
        scanner(scanner.initial_coords)


# ---------------------------------------------------------------------------
# MECP lower normalisation branches and out-of-range upper (finding 10).
# ---------------------------------------------------------------------------

def test_paired_mecp_lower_as_ground_state_target():
    from adcc.gradients import GroundStateTarget
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", lower=GroundStateTarget(level=2), upper=0,
    )
    assert isinstance(scanner.paired_target, PairedGroundExcitedStateTarget)
    assert scanner.paired_target.level == 2


def test_paired_mecp_lower_as_int_level():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", lower=2, upper=0,
    )
    assert isinstance(scanner.paired_target, PairedGroundExcitedStateTarget)
    assert scanner.paired_target.level == 2


def test_paired_mecp_lower_bad_type_raises():
    with pytest.raises(TypeError, match="lower must be"):
        adcc.PairedStateGradientScanner(
            _h2o_scf(), method="adc2", lower=1.5, upper=0,
        )


def test_paired_mecp_upper_out_of_range_raises():
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", lower="mp2", upper=5, n_singlets=2,
        follow="index", conv_tol=1e-9,
    )
    with pytest.raises(ValueError, match="out of range"):
        scanner(scanner.initial_coords)


# ---------------------------------------------------------------------------
# Distinctness guard on duplicate positional seeds (finding 11).
# ---------------------------------------------------------------------------

def test_paired_meci_duplicate_index_mode_raises_distinctness():
    # In pure index mode with states=(0, 0) both slots want the same root;
    # the distinctness guard must raise rather than silently collapse.
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 0), n_singlets=3,
        follow="index", conv_tol=1e-9,
    )
    with pytest.raises(RuntimeError, match="collapsed onto state"):
        scanner(scanner.initial_coords)


# ---------------------------------------------------------------------------
# Joint tie-break toward the smaller excitation-energy gap (finding 13).
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_joint_selection_tie_break_prefers_smaller_excitation_gap(monkeypatch):
    # Both slots track via overlap with two distinct ordered pairs tied on the
    # summed overlap score but differing in excitation-energy gap; the joint
    # selection must prefer the pair closer to the seam (smaller gap) even
    # though it is neither the first-iterated nor the lowest-index pair.
    scanner = adcc.PairedStateGradientScanner(
        _h2o_scf(), method="adc2", states=(0, 1), n_singlets=4,
        follow="overlap", conv_tol=1e-9,
    )
    nao = scanner.base_mol.nao

    # Score table keyed by the previous descriptor identity + candidate index:
    # slot0 likes {0, 2}, slot1 likes {1, 3} -> four ordered pairs tie at 1.0.
    sentinel0, sentinel1 = object(), object()
    score_map = {
        id(sentinel0): [1.0, 0.0, 1.0, 0.0],
        id(sentinel1): [0.0, 1.0, 0.0, 1.0],
    }
    omegas = [0.10, 0.20, 0.11, 0.19]
    # Paired gap analysis: (0,1)=0.10 (0,3)=0.09 (2,1)=0.09 (2,3)=0.08 smallest.

    # _descriptor returns the candidate index so _tracking_score can map it
    # back to a controlled score without fabricating density matrices.
    monkeypatch.setattr(
        scanner, "_descriptor", lambda excitation, mol: excitation.index,
    )
    monkeypatch.setattr(
        scanner, "_tracking_score",
        lambda prev, current, mol, unavailable:
            float(score_map[id(prev)][current]),
    )
    # Ensure score_vectors are built (follow == "overlap" AND prev is not None).
    scanner.previous_descriptors = (sentinel0, sentinel1)
    scanner.previous_indices = (0, 1)

    candidates = [
        _FakeExcitation(i, np.zeros((nao, nao)), np.zeros((nao, nao)), omegas[i])
        for i in range(4)
    ]
    states = types.SimpleNamespace(excitations=candidates)
    chosen, _descriptors, _trackings = scanner._select_excitations(
        states, scanner.base_mol.copy()
    )
    assert tuple(chosen) == (2, 3)  # smallest excitation-energy-gap pair
