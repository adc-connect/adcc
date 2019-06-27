#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
from pyscf import gto, scf

import adcc

# Run SCF in pyscf
mol = gto.M(
    atom='S  -0.38539679062   0 -0.27282082253;'
         'H  -0.0074283962687 0  2.2149138578;'
         'H   2.0860198029    0 -0.74589639249',
    basis='cc-pvtz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()

print(adcc.banner())

# Run an adc2x calculation:
singlets = adcc.cvs_adc2x(scfres, core_orbitals=1, frozen_core=1, n_singlets=3)
triplets = adcc.cvs_adc2x(singlets.matrix, n_triplets=3)

print(singlets.describe())
print()
print(triplets.describe())
