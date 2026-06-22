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
"""End-to-end MECP/MECI optimisation smoke tests.

These drive the :class:`PairedStateGradientScanner` + :class:`MECPObjective`
penalty through geomeTRIC (via PySCF's geomopt bridge) and are skipped cleanly
when either optional dependency is unavailable.  They exercise the paired
root-tracking + distinctness guard across real optimisation steps; the heavy
full-optimisation convergence is left to the example script and these tests
keep ``maxsteps`` tiny.
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
    reason="PySCF and geomeTRIC are required for end-to-end MECP/MECI tests.",
)


def _ethylene_scf():
    """A small twisted ethylene SCF: the textbook minimal MECI system.

    A near-90 deg torsion about the C=C bond drives the two lowest singlet
    surfaces toward a conical intersection, keeping the system cheap at
    STO-3G.  This fixture is authored fresh (no ethylene fixture exists yet).
    """
    from pyscf import gto, scf
    # Twist one CH2 group ~80 deg to start close to the crossing seam.
    tw = np.deg2rad(80.0)
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([1.34, 0.0, 0.0])
    # Left CH2 (untwisted).
    h1l = c1 + np.array([0.0, 0.63, 0.0])
    h2l = c1 + np.array([0.0, -0.63, 0.0])
    # Right CH2 (twisted about the C=C axis = x).

    def twist(p, angle, pivot):
        r = p - pivot
        rot = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(angle), -np.sin(angle)],
                        [0.0, np.sin(angle), np.cos(angle)]])
        return pivot + rot @ r
    h1r = twist(c2 + np.array([0.0, 0.63, 0.0]), tw, c2)
    h2r = twist(c2 + np.array([0.0, -0.63, 0.0]), tw, c2)
    atoms = ["C", "C", "H", "H", "H", "H"]
    coords = np.stack([c1, c2, h1l, h2l, h1r, h2r])
    atom_str = "\n".join(
        f"{sym} {xyz[0]:.8f} {xyz[1]:.8f} {xyz[2]:.8f}"
        for sym, xyz in zip(atoms, coords)
    )
    mol = gto.M(
        atom=atom_str, basis="sto-3g", unit="Angstrom",
        symmetry=False, verbose=0, parse_arg=False,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-7
    return mf


def test_paired_scanner_objective_drives_meci_optimization():
    from pyscf.geomopt import as_pyscf_method, geometric_solver

    scfres = _ethylene_scf()
    scanner = adcc.PairedStateGradientScanner(
        scfres, method="adc2", states=(0, 1), n_singlets=3,
        follow="overlap", conv_tol=1e-8,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    objective = adcc.MECPObjective(scanner)  # default LCM penalty

    seen_gaps = []

    def energy_and_gradient(mol_at_step):
        energy, gradient = objective(mol_at_step.atom_coords(unit="Bohr"))
        e_lo, e_hi = objective.last_pair[0][0], objective.last_pair[1][0]
        seen_gaps.append(abs(e_hi - e_lo))
        return energy, gradient

    method = as_pyscf_method(scfres.mol, energy_and_gradient)
    geometric_solver.optimize(
        method, maxsteps=3,
        convergence_grms=1e-4, convergence_gmax=2e-4,
    )

    # Multiple steps ran and both surfaces were evaluated every step.
    assert len(seen_gaps) >= 2
    assert all(np.isfinite(gap) for gap in seen_gaps)
    assert scanner.last_trackings is not None
    # Tracking across steps kept the two followed roots distinct throughout.
    slot0, slot1 = scanner.last_trackings
    assert slot0 is not None and slot1 is not None
    assert slot0.index != slot1.index


def test_mecp_objective_calc_new_contract_end_to_end():
    # Reuse the ethylene system to confirm the MECPObjective's calc_new returns
    # the geomeTRIC custom-engine dict with finite energy and a flattened,
    # Bohr-scaled gradient.
    scfres = _ethylene_scf()
    scanner = adcc.PairedStateGradientScanner(
        scfres, method="adc2", states=(0, 1), n_singlets=3,
        follow="index", conv_tol=1e-8,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    objective = adcc.MECPObjective(scanner)
    result = objective.calc_new(scanner.initial_coords.ravel())
    assert set(result) == {"energy", "gradient"}
    assert isinstance(result["energy"], float)
    assert np.isfinite(result["energy"])
    assert result["gradient"].shape == (3 * scanner.natoms,)
    assert np.all(np.isfinite(result["gradient"]))


def test_mecp_ground_excited_pair_drives_optimization():
    from pyscf.geomopt import as_pyscf_method, geometric_solver

    scfres = _ethylene_scf()
    scanner = adcc.PairedStateGradientScanner(
        scfres, method="adc2", lower="mp2", upper=0, n_singlets=3,
        follow="index", conv_tol=1e-8,
        gradient_kwargs={"eri_contraction": "full_ao"},
    )
    objective = adcc.MECPObjective(scanner)

    def energy_and_gradient(mol_at_step):
        return objective(mol_at_step.atom_coords(unit="Bohr"))

    method = as_pyscf_method(scfres.mol, energy_and_gradient)
    # A couple of steps confirm the MECP objective drives geomeTRIC without
    # errors; full seam convergence is not required here.
    geometric_solver.optimize(
        method, maxsteps=2,
        convergence_grms=1e-4, convergence_gmax=2e-4,
    )
    # The ground/excited pair stayed finite and ordered.
    e_lo, e_hi = objective.last_pair[0][0], objective.last_pair[1][0]
    assert np.isfinite(e_lo) and np.isfinite(e_hi)
    assert e_lo <= e_hi
