#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from adcc.solver.adcman import jacobi_davidson

from pyscf import gto, scf

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
scfres.conv_tol_grad = 1e-10
scfres.kernel()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run solver and print results
states = adcc.adc2(scfres, n_spin_flip=5, conv_tol=1e-8)
print(states.describe())
