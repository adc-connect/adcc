#!/usr/bin/env python3
"""Excited-state geometry optimisation with adcc gradients.

This example uses :func:`adcc.nuclear_gradient_scanner` to expose an ADC excited
state as a PySCF/geomeTRIC-compatible energy-gradient callable.  The scanner
uses the configured PySCF SCF object as its template, then runs adcc and follows
the selected excited state with AO-basis transition/state-difference density
overlaps.

geomeTRIC is optional; install it separately, e.g. ``pip install geometric`` or
``pip install adcc[geomopt]`` once the optional extra is available.
"""

import adcc
from pyscf import gto, scf
from pyscf.geomopt import as_pyscf_method, geometric_solver


# Use Bohr coordinates and keep molecular symmetry disabled so the optimizer's
# Cartesian frame and the gradient frame stay identical during the scan.
mol = gto.M(
    atom="""
    O  0.000000000000  0.000000000000  0.000000000000
    H  0.000000000000  0.000000000000  1.795239827225
    H  1.693194615993  0.000000000000 -0.599043184453
    """,
    basis="sto-3g",
    unit="Bohr",
    symmetry=False,
    verbose=0,
)

# Configure SCF entirely on the PySCF object.  The scanner will use this object
# as the template and PySCF will preserve SCF guess continuity between geometry
# steps.
mf = scf.RHF(mol)
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-9

# Optimise the lowest singlet ADC(2) state.  Requesting a few singlet roots lets
# the scanner compare candidates and follow the same physical state if root
# ordering changes along the optimisation.  adcc keyword arguments such as
# conv_tol and n_singlets are forwarded unchanged to adcc.run_adc.
scanner = adcc.nuclear_gradient_scanner(
    mf,
    method="adc2",
    state_index=0,
    n_singlets=3,
    follow="overlap",
    conv_tol=1e-7,
)


def energy_and_gradient(mol_at_step):
    """PySCF geomopt callback: return energy and gradient for a Mole."""
    energy, gradient = scanner(mol_at_step.atom_coords(unit="Bohr"))
    print(
        f"E = {energy:.12f} Eh, |g| = "
        f"{float((gradient ** 2).sum() ** 0.5):.6e} Eh/Bohr"
    )
    return energy, gradient


if __name__ == "__main__":
    method = as_pyscf_method(mol, energy_and_gradient)
    mol_eq = geometric_solver.optimize(
        method,
        maxsteps=20,
        convergence_grms=1e-3,
        convergence_gmax=1.5e-3,
    )

    print("\nOptimised geometry (Bohr):")
    for symbol, xyz in zip(scanner.atom_symbols, mol_eq.atom_coords(unit="Bohr")):
        print(f"{symbol:2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")
