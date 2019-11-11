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

# Explicitly initialise ADC virtual memory pool to (256 MiB)
# (if this call is missing than only RAM is used. This allows
#  to dump data to disk as well such that maximally up to the
#  specified amount of data (in bytes) will reside in RAM)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc2 calculation:
singlets = adcc.adc2(scfres, n_states=5)
# triplets = adcc.adc2(singlets.matrix, n_triplets=3)

print(singlets.describe())
print()
# print(triplets.describe())
