#!/usr/bin/env python3
"""Excited-state geometry optimisation with adcc gradients.

This example first relaxes water on the MP2 ground-state surface and then uses
that geometry as the starting point for an ADC(2) excited-state optimisation.
The :func:`adcc.nuclear_gradient_scanner` is built from a configured PySCF SCF
object, so PySCF owns all SCF settings while adcc keyword arguments are passed
unchanged to :func:`adcc.run_adc`.

geomeTRIC is optional; install it separately, e.g. ``pip install geometric`` or
``pip install adcc[geomopt]`` once the optional extra is available.
"""

import adcc
from pyscf import gto, scf
from pyscf.geomopt import as_pyscf_method, geometric_solver


def make_scf(mol):
    """Build the PySCF reference used as scanner template."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.conv_tol_grad = 1e-6
    return mf


def print_geometry(title, mol):
    print(f"\n{title} (Bohr):")
    for i, xyz in enumerate(mol.atom_coords(unit="Bohr")):
        print(f"{mol.atom_symbol(i):2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")


def make_energy_gradient(scanner):
    """PySCF geomopt callback: return energy and gradient for a Mole."""
    def energy_and_gradient(mol_at_step):
        energy, gradient = scanner(mol_at_step.atom_coords(unit="Bohr"))
        excitation = scanner.last_excitation
        state_info = ""
        if excitation is not None:
            state_info = (
                f", state = {excitation.index}, "
                f"omega = {excitation.excitation_energy:.8f} Eh"
            )
        print(
            f"E = {energy:.12f} Eh, |g| = "
            f"{float((gradient ** 2).sum() ** 0.5):.6e} Eh/Bohr"
            f"{state_info}"
        )
        return energy, gradient
    return energy_and_gradient


# Use Bohr coordinates and keep molecular symmetry disabled so the optimizer's
# Cartesian frame and the gradient frame stay identical during the scan.
mol = gto.M(
    atom="""
    O  0.000000000000  0.000000000000  0.000000000000
    H  0.000000000000  0.000000000000  1.795239827225
    H  1.693194615993  0.000000000000 -0.599043184453
    """,
    basis="cc-pvdz",
    unit="Bohr",
    symmetry=False,
    verbose=0,
)


if __name__ == "__main__":
    print_geometry("Initial geometry", mol)

    # Stage 1: relax the ground-state structure at MP2 level.  This provides a
    # much better starting point for the subsequent state-specific optimisation
    # than the arbitrary input geometry.
    mp2_scanner = adcc.nuclear_gradient_scanner(
        make_scf(mol),
        method="mp2",
        gradient_kwargs={"conv_tol": 1e-7},
    )
    mp2_method = as_pyscf_method(mol, make_energy_gradient(mp2_scanner))
    mol_mp2 = geometric_solver.optimize(
        mp2_method,
        maxsteps=20,
        convergence_grms=1e-3,
        convergence_gmax=1.5e-3,
    )
    print_geometry("MP2-optimised geometry", mol_mp2)

    # Stage 2: optimise the lowest singlet ADC(2) state.  Requesting a few
    # singlet roots lets the scanner compare candidates and follow the same
    # physical state if root ordering changes along the optimisation.
    adc_scanner = adcc.nuclear_gradient_scanner(
        make_scf(mol_mp2),
        method="adc2",
        state_index=0,
        n_singlets=3,
        follow="overlap",
        conv_tol=1e-6,
        gradient_kwargs={"conv_tol": 1e-7},
    )
    adc_method = as_pyscf_method(mol_mp2, make_energy_gradient(adc_scanner))
    mol_adc = geometric_solver.optimize(
        adc_method,
        maxsteps=20,
        convergence_grms=1e-3,
        convergence_gmax=1.5e-3,
    )
    print_geometry("ADC(2)-optimised demonstration geometry", mol_adc)
