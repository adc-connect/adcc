#!/usr/bin/env python3
"""Spin-flip ADC conical-intersection optimisation with adcc gradients.

This example locates the S0/S1 conical intersection of twisted ethylene
(C2H4), the textbook minimal MECI system, using a **spin-flip ADC** setup:

* the Hartree--Fock reference is the **triplet** ground state (unrestricted
  SCF, ``spin = 2``), the natural diradical reference for the twisted geometry;
* a single spin-flip ADC(2) solve produces both the singlet ground state
  ``S0`` (the first spin-flip state) and the first excited singlet ``S1``;
* the :class:`adcc.PairedStateGradientScanner` evaluates these two surfaces at
  one geometry from a single SCF + single ADC, and
  :class:`adcc.MECPObjective` combines them into one ``(energy, gradient)``
  for geomeTRIC's penalty conical-intersection driver.

Because both surfaces come from the *same* spin-flip ADC solve, the
S0/S1 crossing is treated as a proper excited/excited MECI -- no separate
ground-state surface is optimised, which avoids the convergence problems of a
ground/excited MECP formulation near the diradical region.  **No derivative
couplings** are required.

geomeTRIC is optional; install it separately, e.g. ``pip install geometric`` or
``pip install adcc[geomopt]`` once the optional extra is available.
"""

import numpy as np

import adcc
from pyscf import gto, scf
from pyscf.geomopt import as_pyscf_method, geometric_solver


def twisted_ethylene(twist_deg=90.0, basis="6-31g"):
    """Build an unrestricted triplet-HF ethylene near the S0/S1 crossing seam.

    The triplet is the diradical reference for spin-flip ADC: the first
    spin-flip state recovers the closed-shell singlet ground state ``S0`` and
    the next one the first excited singlet ``S1``, so the S0/S1 crossing is
    accessible as a single-solve excited/excited MECI.
    """
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
        spin=2,            # 2S = 2 -> triplet reference for spin-flip ADC
        symmetry=False, verbose=0, parse_arg=False,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-7
    mf.max_cycles = 250
    return mf


def print_geometry(title, mol):
    print(f"\n{title} (Angstrom):")
    for i, xyz in enumerate(mol.atom_coords(unit="Angstrom")):
        print(f"{mol.atom_symbol(i):2s} "
              f"{xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")


if __name__ == "__main__":
    scfres = twisted_ethylene(twist_deg=90.0, basis="6-31g")
    mol = scfres.mol
    print_geometry("Initial twisted geometry (triplet reference)", mol)

    paired = adcc.PairedStateGradientScanner(
        scfres,
        method="adc2",
        states=(0, 1),       # first two spin-flip states: S0 (singlet GS)
        #                       and S1 (first excited singlet) -- a MECI
        n_spin_flip=4,        # spin-flip ADC, unrestricted reference only
        # For a MECI pair the two surfaces are the adiabatically lowest states
        # at each geometry (the seam is defined by an energy-ordered degeneracy,
        # not by fixed state character).  Density-overlap tracking -- which
        # serves single-surface optimization -- would *fight* the adiabatic
        # reordering near the seam and flip a slot onto a higher root.  Instead
        # follow="index" returns positional roots (0, 1) and lets the scanner
        # energy-sort them, exactly the contract geomeTRIC's conical-intersection
        # engine expects from its sub-engines.
        follow="index",
        conv_tol=1e-8,
        gradient_kwargs={"eri_contraction": "direct", "conv_tol": 1e-8},
    )
    # The default penalty uses the smoothed Levine--Coe--Martinez form, the same
    # formulation as geomeTRIC's built-in conical-intersection engine.  A larger
    # sigma enforces the degeneracy harder for a tighter final gap.
    objective = adcc.MECPObjective(paired, sigma=200.0, alpha=0.025)

    # Optional per-step "is it converging?" printer attached as a callback; read
    # the seam gap off objective.last_pair.  The objective accepts a PySCF Mole
    # directly (forwarded to the paired scanner), so it plugs straight into
    # as_pyscf_method without a wrapper.
    def _print_step(energy, gradient):
        e_lo, e_hi = objective.last_pair[0][0], objective.last_pair[1][0]
        gnorm = float((gradient ** 2).sum() ** 0.5)
        print(f"E_pen = {energy:.12f} Eh, |g| = {gnorm:.6e} Eh/Bohr, "
              f"gap = {abs(e_hi - e_lo):.6e} Eh")
    objective.step_callback = _print_step

    method = as_pyscf_method(mol, objective)
    mol_ci = geometric_solver.optimize(
        method,
        maxsteps=50,
        convergence_grms=3e-5,
        convergence_gmax=1e-4,
    )
    print_geometry("S0/S1 MECI geometry", mol_ci)

    # Report the final two spin-flip surfaces (S0 + S1) at the located geometry.
    (e_lo, _), (e_hi, _) = paired(mol_ci.atom_coords(unit="Bohr"))
    print(f"\nFinal surface energies: S0 = {e_lo:.10f} Eh, "
          f"S1 = {e_hi:.10f} Eh, gap = {abs(e_hi - e_lo):.2e} Eh")
