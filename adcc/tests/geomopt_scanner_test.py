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
