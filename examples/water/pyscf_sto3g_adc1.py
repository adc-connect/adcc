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
    unit="Bohr"
)

# Run RHF SCF
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-14
scfres.conv_tol_grad = 1e-10
scfres.kernel()

# Run an adc1 calculation:
state = adcc.adc1(scfres, n_singlets=5)

print(state.describe())
