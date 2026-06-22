#!/usr/bin/env python3
"""Minimum-energy conical intersection (MECI) optimisation with adcc gradients.

This example locates a conical-intersection seam point of twisted ethylene
(C2H4), the textbook minimal MECI system, using the paired-state scanner and the
Levine--Coe--Martinez penalty objective.  The
:class:`adcc.PairedStateGradientScanner` evaluates two excited-state surfaces at
one geometry from a single SCF + single
ADC, and :class:`adcc.MECPObjective` combines them into one
``(energy, gradient)`` for geomeTRIC.  **No derivative couplings** are required.

geomeTRIC is optional; install it separately, e.g. ``pip install geometric`` or
``pip install adcc[geomopt]`` once the optional extra is available.
"""

import numpy as np

import adcc
from pyscf import gto, scf
from pyscf.geomopt import as_pyscf_method, geometric_solver


def twisted_ethylene(twist_deg=80.0, basis="sto-3g"):
    """Build a twisted ethylene PySCF SCF near the S1/S0 crossing seam."""
    tw = np.deg2rad(twist_deg)
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([1.34, 0.0, 0.0])
    h1l = c1 + np.array([0.0, 0.63, 0.0])
    h2l = c1 + np.array([0.0, -0.63, 0.0])

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
        atom=atom_str, basis=basis, unit="Angstrom",
        symmetry=False, verbose=0, parse_arg=False,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-7
    return mf


def print_geometry(title, mol):
    print(f"\n{title} (Angstrom):")
    for i, xyz in enumerate(mol.atom_coords(unit="Angstrom")):
        print(f"{mol.atom_symbol(i):2s} "
              f"{xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")


if __name__ == "__main__":
    scfres = twisted_ethylene(twist_deg=80.0)
    mol = scfres.mol
    print_geometry("Initial twisted geometry", mol)

    paired = adcc.PairedStateGradientScanner(
        scfres,
        method="adc2",
        states=(0, 1),       # two tracked excited states (MECI)
        n_singlets=4,
        follow="overlap",
        conv_tol=1e-8,
        gradient_kwargs={"eri_contraction": "full_ao", "conv_tol": 1e-7},
    )
    # The default penalty uses the smoothed Levine--Coe--Martinez form, the same
    # formulation as geomeTRIC's built-in conical-intersection engine.
    objective = adcc.MECPObjective(paired)

    def energy_and_gradient(mol_at_step):
        energy, gradient = objective(mol_at_step.atom_coords(unit="Bohr"))
        e_lo, e_hi = objective.last_pair[0][0], objective.last_pair[1][0]
        gnorm = float((gradient ** 2).sum() ** 0.5)
        print(f"E_pen = {energy:.12f} Eh, |g| = {gnorm:.6e} Eh/Bohr, "
              f"gap = {abs(e_hi - e_lo):.6e} Eh")
        return energy, gradient

    method = as_pyscf_method(mol, energy_and_gradient)
    mol_ci = geometric_solver.optimize(
        method,
        maxsteps=40,
        convergence_grms=1e-4,
        convergence_gmax=2e-4,
    )
    print_geometry("MECI geometry", mol_ci)

    # Report the final two surfaces at the located geometry.
    (e_lo, _), (e_hi, _) = paired(mol_ci.atom_coords(unit="Bohr"))
    print(f"\nFinal surface energies: lower = {e_lo:.10f} Eh, "
          f"upper = {e_hi:.10f} Eh, gap = {abs(e_hi - e_lo):.2e} Eh")
