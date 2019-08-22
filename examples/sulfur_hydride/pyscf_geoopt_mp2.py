#!/usr/bin/env python3
from pyscf import gto, mp, scf
from pyscf.geomopt import berny_solver

# Starting geometry
mol = gto.M(
    atom='S 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvtz',
    verbose=3,
    unit="Bohr"
)

# HF optimisation
mf = scf.RHF(mol)
mol_hf_eq = berny_solver.optimize(mf)

# MP2 optimisation
mp2 = mp.MP2(scf.RHF(mol_hf_eq))
mol_mp2_eq = berny_solver.optimize(mp2)


print()
print("===========  Final MP2 geometry (bohr) ============")
print()
fmt = "{}  {:15.11g}  {:15.11g}  {:15.11g}"
coords = mol_mp2_eq.atom_coords()
n_atoms = len(coords)
for i in range(n_atoms):
    print(fmt.format(mol_mp2_eq.atom_symbol(i), *coords[i]))
