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
import types

import numpy as np
import pytest
from numpy.testing import assert_allclose

import adcc
import adcc.backends
from adcc.gradients.scanner import (
    GroundStateTarget,
    _TrackingDescriptor,
    _safe_ao_ndarray,
    density_overlap_score,
)


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


def test_density_overlap_score_uses_absolute_transition_phase():
    dm = np.eye(2)
    s = np.eye(2)
    assert density_overlap_score(dm, dm, s) == pytest.approx(1.0)
    assert density_overlap_score(dm, -dm, s) == pytest.approx(-1.0)
    assert density_overlap_score(dm, -dm, s, use_abs=True) == pytest.approx(1.0)


def test_scanner_requires_pyscf_scf_object():
    with pytest.raises(TypeError, match="PySCF SCF object"):
        adcc.NuclearGradientScanner(_h2o_scf().mol, method="mp2")


def test_scanner_validates_coordinate_shape():
    scanner = adcc.NuclearGradientScanner(_h2o_scf(), method="mp2")
    with pytest.raises(ValueError, match="Expected coordinates"):
        scanner(np.zeros((2, 3)))


def test_ground_state_scanner_matches_explicit_gradient_loop():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="mp2", gradient_kwargs={"eri_contraction": "full_ao"},
    )
    coords = scanner.initial_coords

    energy, gradient = scanner(coords)

    mp = adcc.LazyMp(adcc.ReferenceState(scanner.last_scf))
    explicit_gradient = adcc.nuclear_gradient(mp, eri_contraction="full_ao")
    assert energy == pytest.approx(mp.energy(2))
    assert gradient.shape == (3, 3)
    assert_allclose(gradient, explicit_gradient.total)


def test_calc_new_returns_geometric_engine_shape():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="mp2", gradient_kwargs={"eri_contraction": "full_ao"},
    )
    result = scanner.calc_new(scanner.initial_coords.ravel())
    assert set(result) == {"energy", "gradient"}
    assert isinstance(result["energy"], float)
    assert result["gradient"].shape == (9,)


def test_scanner_accepts_pyscf_mole_directly():
    # The scanner reads atom_coords(unit="Bohr") on a PySCF Mole, so it can be
    # passed straight to as_pyscf_method without a hand-rolled wrapper.
    scfres = _h2o_scf()
    scanner = adcc.NuclearGradientScanner(
        scfres, method="mp2", gradient_kwargs={"eri_contraction": "full_ao"},
    )
    e_via_mole, g_via_mole = scanner(scfres.mol)
    e_via_coords, g_via_coords = scanner(scanner.initial_coords)
    assert e_via_mole == pytest.approx(e_via_coords, abs=1e-10)
    assert_allclose(g_via_mole, g_via_coords, atol=1e-10)


def test_scanner_step_callback_is_invoked_with_energy_and_gradient():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="mp2", gradient_kwargs={"eri_contraction": "full_ao"},
    )
    seen = []
    scanner.step_callback = lambda e, g: seen.append((e, float(np.asarray(g).sum())))
    e, g = scanner(scanner.initial_coords)
    assert len(seen) == 1
    assert seen[0][0] == pytest.approx(e, abs=1e-12)
    assert seen[0][1] == pytest.approx(float(np.asarray(g).sum()), abs=1e-10)


def test_run_adc_kwargs_are_forwarded_with_native_names():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=2, conv_tol=1e-7,
        follow="index", gradient_kwargs={"eri_contraction": "full_ao"},
    )
    assert scanner.target.kwargs()["conv_tol"] == pytest.approx(1e-7)
    assert scanner.target.kwargs()["n_singlets"] == 2
    assert "output" not in scanner.target.kwargs()


def test_tracking_descriptor_keeps_independent_mol_snapshot():
    class FakeAoOperator:
        def to_ndarray(self):
            return np.eye(7)

    class FakeExcitation:
        transition_dm_ao = FakeAoOperator()
        state_diffdm_ao = FakeAoOperator()

    scanner = adcc.NuclearGradientScanner(_h2o_scf(), method="adc2")
    mol = scanner.base_mol.copy()
    descriptor = scanner._descriptor(FakeExcitation(), mol)
    old_coords = descriptor.mol.atom_coords(unit="Bohr").copy()

    mol.set_geom_(old_coords + 0.1, unit="Bohr")

    assert_allclose(descriptor.mol.atom_coords(unit="Bohr"), old_coords)


def test_unconverged_scf_raises():
    from pyscf import scf
    mf = scf.RHF(_h2o_scf().mol)
    mf.max_cycle = 1
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    scanner = adcc.NuclearGradientScanner(mf, method="mp2")
    with pytest.raises(RuntimeError, match="did not converge"):
        scanner(scanner.initial_coords)


def test_excited_state_scanner_matches_explicit_gradient_loop():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=0,
        follow="index", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )

    energy, gradient = scanner(scanner.initial_coords)

    # Fixed-index selection must pick exactly the requested positional state.
    assert scanner.last_excitation.index == 0
    assert scanner.last_tracking is None

    states = adcc.run_adc(scanner.last_scf, method="adc2", n_singlets=3,
                          conv_tol=1e-9)
    explicit = adcc.nuclear_gradient(states.excitations[0],
                                     eri_contraction="full_ao")
    assert energy == pytest.approx(states.excitations[0].total_energy, abs=1e-7)
    assert gradient.shape == (3, 3)
    assert_allclose(gradient, explicit.total, atol=1e-7)


def test_overlap_tracking_follows_same_state_character():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=1,
        follow="overlap", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )

    # First call seeds the tracker from the positional index.
    scanner(scanner.initial_coords)
    assert scanner.last_excitation.index == 1
    assert scanner.last_tracking is None

    # Second call at the same geometry must follow the seeded state via density
    # overlap.  At identical geometry the matching state has self-overlap ~1.
    scanner(scanner.initial_coords)
    assert scanner.last_tracking is not None
    assert scanner.last_tracking.index == 1
    scores = scanner.last_tracking.scores
    assert np.argmax(scores) == 1
    assert scores[1] == pytest.approx(1.0, abs=1e-3)


def test_overlap_tracking_continuous_under_small_displacement():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=0,
        follow="overlap", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )

    scanner(scanner.initial_coords)
    omega_initial = scanner.last_excitation.excitation_energy

    displaced = scanner.initial_coords.copy()
    displaced[0, 2] += 0.02
    scanner(displaced)

    # The followed state should remain the same physical state, i.e. its
    # excitation energy must not jump discontinuously between candidates.
    assert scanner.last_tracking is not None
    assert abs(scanner.last_excitation.excitation_energy - omega_initial) < 0.05


def test_overlap_tracking_below_min_score_raises():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=0,
        follow="overlap", tracking_min_score=2.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    with pytest.raises(RuntimeError, match="below threshold"):
        scanner(scanner.initial_coords)


def test_overlap_tracking_ambiguous_gap_raises():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=0,
        follow="overlap", tracking_min_gap=10.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    with pytest.raises(RuntimeError, match="ambiguous"):
        scanner(scanner.initial_coords)


def test_scf_guess_continuity_updates_previous_state():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="mp2", gradient_kwargs={"eri_contraction": "full_ao"},
    )

    energy_first, _ = scanner(scanner.initial_coords)
    assert scanner.previous_scf is scanner.last_scf

    # Move away and return: the scanner reuses one PySCF scanner object across
    # geometries (orbital/guess continuity), and the surface stays consistent so
    # the energy at the original geometry is reproduced.
    displaced = scanner.initial_coords.copy()
    displaced[0, 2] += 0.05
    scanner(displaced)
    energy_back, _ = scanner(scanner.initial_coords)
    assert energy_back == pytest.approx(energy_first, abs=1e-9)


# ---------------------------------------------------------------------------
# Root-reorder tracking (FINDING 4)
# ---------------------------------------------------------------------------

class _FakeAoOperator:
    def __init__(self, matrix):
        self._matrix = matrix

    def to_ndarray(self):
        return self._matrix


class _FakeExcitation:
    def __init__(self, index, transition_dm, state_diffdm):
        self.index = index
        self.transition_dm_ao = _FakeAoOperator(transition_dm)
        self.state_diffdm_ao = _FakeAoOperator(state_diffdm)


def test_overlap_tracking_follows_reordered_root_by_overlap():
    # Prove that overlap tracking follows a physical state whose *positional*
    # index differs from the originally requested ``state_index``: the seeded
    # descriptor matches candidate #1, while the requested state_index is 0.
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=0,
        follow="overlap",
    )
    nao = scanner.base_mol.nao

    def projector(i):
        m = np.zeros((nao, nao))
        m[i, i] = 1.0
        return m

    # The previously-followed physical state has this character.
    seed_transition = projector(0)
    seed_diffdm = projector(1)
    scanner.previous_descriptor = _TrackingDescriptor(
        mol=scanner.base_mol.copy(),
        transition_dm=seed_transition, state_diffdm=seed_diffdm,
    )
    scanner.previous_index = 0

    # Candidate roots: index 1 carries the seeded character (energy ordering
    # swapped so it is no longer at positional index 0).
    candidates = [
        _FakeExcitation(0, projector(2), projector(3)),
        _FakeExcitation(1, seed_transition.copy(), seed_diffdm.copy()),
        _FakeExcitation(2, projector(4), projector(5)),
    ]
    states = types.SimpleNamespace(excitations=candidates)

    selected = scanner._select_excitation(states, scanner.base_mol.copy())

    # Tracking must pick the reordered root (positional index 1) by overlap,
    # which differs from the requested state_index (0).
    assert selected.index == 1
    assert scanner.target.state_index == 0
    assert scanner.last_tracking is not None
    assert scanner.last_tracking.index == 1
    assert scanner.last_tracking.index != scanner.target.state_index
    assert np.argmax(scanner.last_tracking.scores) == 1
    # New diagnostic fields are populated and observable on every call.
    assert scanner.last_tracking.best_score == pytest.approx(1.0)
    assert scanner.last_tracking.gap > 0.0
    assert np.isfinite(scanner.last_tracking.gap)
    assert scanner.last_tracking.previous_index == 0
    assert scanner.last_tracking.switched is True


# ---------------------------------------------------------------------------
# Validation / construction-time branches (FINDING 10)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"target": "mp3"},
    {"method": "mp3"},
    {"target": GroundStateTarget(level=3)},
])
def test_ground_state_level_other_than_two_is_rejected(kwargs):
    with pytest.raises(NotImplementedError, match="MP2 ground-state"):
        adcc.NuclearGradientScanner(_h2o_scf(), **kwargs)


def test_invalid_follow_raises_eagerly_at_construction():
    with pytest.raises(ValueError, match="overlap.*index"):
        adcc.NuclearGradientScanner(
            _h2o_scf(), method="adc2", n_singlets=2, follow="bogus",
        )


def test_state_index_out_of_range_raises():
    scanner = adcc.NuclearGradientScanner(
        _h2o_scf(), method="adc2", n_singlets=2, state_index=5,
        follow="index", conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    with pytest.raises(ValueError, match="out of range"):
        scanner(scanner.initial_coords)


def test_empty_excitations_raises_runtime_error():
    scanner = adcc.NuclearGradientScanner(_h2o_scf(), method="adc2")
    states = types.SimpleNamespace(excitations=[])
    with pytest.raises(RuntimeError, match="did not return any excited states"):
        scanner._select_excitation(states, scanner.base_mol.copy())


@pytest.mark.parametrize("kwargs", [
    {"method": "mp2", "n_singlets": 3},
    {"target": "mp2", "n_singlets": 3},
])
def test_ground_state_run_adc_kwargs_emit_runtime_warning(kwargs):
    with pytest.warns(RuntimeWarning, match="Ignoring run_adc"):
        adcc.NuclearGradientScanner(_h2o_scf(), **kwargs)


# ---------------------------------------------------------------------------
# Density-overlap scoring branches (FINDING 10)
# ---------------------------------------------------------------------------

def test_safe_ao_ndarray_swallows_unavailable_channels():
    class FailingOperator:
        def to_ndarray(self):
            raise NotImplementedError("not available")

    class Excitation:
        transition_dm_ao = FailingOperator()

        @property
        def state_diffdm_ao(self):
            raise AttributeError("no diffdm")

    excitation = Excitation()
    assert _safe_ao_ndarray(excitation, "transition_dm_ao") is None
    assert _safe_ao_ndarray(excitation, "state_diffdm_ao") is None


def test_tracking_score_records_unavailable_and_raises_when_both_missing():
    scanner = adcc.NuclearGradientScanner(_h2o_scf(), method="adc2")
    mol = scanner.base_mol.copy()
    previous = _TrackingDescriptor(mol=mol.copy(),
                                   transition_dm=None, state_diffdm=None)
    current = _TrackingDescriptor(mol=mol.copy(),
                                  transition_dm=None, state_diffdm=None)
    unavailable = set()
    with pytest.raises(RuntimeError, match="neither transition_dm_ao"):
        scanner._tracking_score(previous, current, mol, unavailable)
    # Both channels are surfaced as unavailable before the weight==0 failure.
    assert unavailable == {"transition_dm_ao", "state_diffdm_ao"}


def test_density_overlap_score_zero_density_returns_nan():
    zero = np.zeros((2, 2))
    s = np.eye(2)
    assert np.isnan(density_overlap_score(zero, np.eye(2), s))


def test_density_overlap_score_nonidentity_metrics_matches_hand_value():
    old = np.array([[1.0, 0.0], [0.0, 0.0]])
    new = np.array([[1.0, 0.0], [0.0, 0.0]])
    s_cross = np.array([[0.9, 0.1], [0.1, 0.9]])
    s_old = np.array([[1.0, 0.2], [0.2, 1.0]])
    s_new = np.array([[1.0, 0.3], [0.3, 1.0]])

    # numerator = (D_old . s_cross . D_new . s_cross), norms use s_old/s_new.
    # With single populated diagonal element: numerator = s_cross[0,0]**2 = 0.81,
    # old_norm = s_old[0,0]**2 = 1.0, new_norm = s_new[0,0]**2 = 1.0.
    score = density_overlap_score(old, new, s_cross, s_old=s_old, s_new=s_new)
    assert score == pytest.approx(0.81)
