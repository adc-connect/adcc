#!/usr/bin/env python3
"""Ground-state MP2 geometry optimisation with adcc gradients.

The scanner is built from a configured PySCF SCF object.  PySCF owns all SCF
settings and adcc runs MP2 plus the analytic gradient at each geometry.
"""

import adcc
from pyscf import gto, scf
from pyscf.geomopt import as_pyscf_method, geometric_solver


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

mf = scf.RHF(mol)
mf.conv_tol = 1e-11
mf.conv_tol_grad = 1e-9

scanner = adcc.NuclearGradientScanner(mf, method="mp2")


if __name__ == "__main__":
    # The scanner accepts a PySCF Mole directly (it reads atom_coords in Bohr),
    # so it can be passed straight to as_pyscf_method without a wrapper.
    method = as_pyscf_method(mol, scanner)
    mol_eq = geometric_solver.optimize(method, maxsteps=20)

    print("\nOptimised geometry (Bohr):")
    for symbol, xyz in zip(scanner.atom_symbols, mol_eq.atom_coords(unit="Bohr")):
        print(f"{symbol:2s} {xyz[0]:16.10f} {xyz[1]:16.10f} {xyz[2]:16.10f}")
