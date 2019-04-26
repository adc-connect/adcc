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

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

refstate = adcc.tmp_build_reference_state(scfres)
ground_state = adcc.LazyMp(refstate)
matrix = adcc.AdcMatrix("adc2", ground_state)

states = jacobi_davidson(matrix, print_level=100, n_spin_flip=5)
print(states[0].describe())
