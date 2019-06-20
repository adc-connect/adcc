#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from pyscf import gto, scf
from adcc.solver.adcman import jacobi_davidson

# Run SCF in pyscf
mol = gto.M(
    atom='H 0 0 0;'
         'F 0 0 2.5',
    basis='6-31G',
    unit="Bohr",
    spin=2  # =2S, ergo triplet
)
scfres = scf.UHF(mol)
scfres.conv_tol = 1e-14
scfres.grad_conv_tol = 1e-10
scfres.kernel()

refstate = adcc.ReferenceState(scfres)
matrix = adcc.AdcMatrix("adc2", refstate)

states = jacobi_davidson(matrix, print_level=100, n_spin_flip=5)
print(states[0].describe())
