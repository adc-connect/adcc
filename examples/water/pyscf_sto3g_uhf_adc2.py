#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from pyscf import gto, scf

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='sto-3g',
    spin=2,
    unit="Bohr"
)
scfres = scf.UHF(mol)
scfres.conv_tol = 1e-8
scfres.conv_tol_grad = 1e-8
scfres.max_cycle = 100
scfres.verbose = 4
scfres.kernel()

# Run an adc2 calculation:
singlets = adcc.adc2(scfres, n_states=5)

print(singlets.describe())
