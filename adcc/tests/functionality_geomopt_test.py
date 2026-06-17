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
"""End-to-end geometry-optimisation smoke tests.

These drive the :func:`adcc.nuclear_gradient_scanner` through geomeTRIC (via
PySCF's geomopt bridge) and are skipped cleanly when either optional dependency
is unavailable.
"""
import importlib.util

import numpy as np
import pytest

import adcc
import adcc.backends


def _missing(*modules):
    return [m for m in modules if importlib.util.find_spec(m) is None]


_required = ["pyscf", "geometric"]
pytestmark = pytest.mark.skipif(
    "pyscf" not in adcc.backends.available() or _missing(*_required),
    reason="PySCF and geomeTRIC are required for end-to-end geomopt tests.",
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


def test_ground_state_optimization_reaches_stationary_point():
    from pyscf.geomopt import as_pyscf_method, geometric_solver

    scfres = _h2o_scf()
    scanner = adcc.nuclear_gradient_scanner(scfres, method="mp2")

    def energy_and_gradient(mol_at_step):
        return scanner(mol_at_step.atom_coords(unit="Bohr"))

    method = as_pyscf_method(scfres.mol, energy_and_gradient)
    mol_eq = geometric_solver.optimize(
        method, maxsteps=50,
        convergence_grms=1e-4, convergence_gmax=2e-4,
    )

    # The gradient at the optimised geometry must be (near) zero.
    _, gradient = scanner(mol_eq.atom_coords(unit="Bohr"))
    gmax = np.abs(gradient).max()
    assert gmax < 5e-4


def test_calc_new_drives_geometric_internal_engine():
    # Exercise the geomeTRIC custom-engine calc_new contract directly: flattened
    # Bohr coordinates in, energy plus flattened Hartree/Bohr gradient out.
    scanner = adcc.nuclear_gradient_scanner(_h2o_scf(), method="mp2")
    result = scanner.calc_new(scanner.initial_coords.ravel())
    assert set(result) == {"energy", "gradient"}
    assert np.isfinite(result["energy"])
    assert result["gradient"].shape == (3 * scanner.natoms,)
