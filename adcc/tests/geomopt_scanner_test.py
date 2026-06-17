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
import numpy as np
import pytest
from numpy.testing import assert_allclose

import adcc
import adcc.backends
from adcc.gradients.scanner import density_overlap_score


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
        adcc.nuclear_gradient_scanner(_h2o_scf().mol, method="mp2")


def test_scanner_validates_coordinate_shape():
    scanner = adcc.nuclear_gradient_scanner(_h2o_scf(), method="mp2")
    with pytest.raises(ValueError, match="Expected coordinates"):
        scanner(np.zeros((2, 3)))


def test_ground_state_scanner_matches_explicit_gradient_loop():
    scanner = adcc.nuclear_gradient_scanner(
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
    scanner = adcc.nuclear_gradient_scanner(
        _h2o_scf(), method="mp2", gradient_kwargs={"eri_contraction": "full_ao"},
    )
    result = scanner.calc_new(scanner.initial_coords.ravel())
    assert set(result) == {"energy", "gradient"}
    assert isinstance(result["energy"], float)
    assert result["gradient"].shape == (9,)


def test_run_adc_kwargs_are_forwarded_with_native_names():
    scanner = adcc.nuclear_gradient_scanner(
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

    scanner = adcc.nuclear_gradient_scanner(_h2o_scf(), method="adc2")
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
    scanner = adcc.nuclear_gradient_scanner(mf, method="mp2")
    with pytest.raises(RuntimeError, match="did not converge"):
        scanner(scanner.initial_coords)


def test_excited_state_scanner_matches_explicit_gradient_loop():
    scanner = adcc.nuclear_gradient_scanner(
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
    scanner = adcc.nuclear_gradient_scanner(
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
    scanner = adcc.nuclear_gradient_scanner(
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
    scanner = adcc.nuclear_gradient_scanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=0,
        follow="overlap", tracking_min_score=2.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    with pytest.raises(RuntimeError, match="below threshold"):
        scanner(scanner.initial_coords)


def test_overlap_tracking_ambiguous_gap_raises():
    scanner = adcc.nuclear_gradient_scanner(
        _h2o_scf(), method="adc2", n_singlets=3, state_index=0,
        follow="overlap", tracking_min_gap=10.0, conv_tol=1e-9,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    scanner(scanner.initial_coords)  # seed
    with pytest.raises(RuntimeError, match="ambiguous"):
        scanner(scanner.initial_coords)


def test_scf_guess_continuity_updates_previous_state():
    scanner = adcc.nuclear_gradient_scanner(
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
