#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from pyscf import gto, scf
from matplotlib import pyplot as plt

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
nmo = mol.nao

scfres.analyze()

print()
print("=========== CVS (1s to 2s) ================")
print()
singlets = adcc.cvs_adc2x(scfres, core_orbitals=2, n_singlets=10)
print(singlets.describe())
singlets.plot_spectrum(label="H2S CVS (1s and 2s)")


print()
print("=========== CVS (2s) ================")
print()
singlets = adcc.cvs_adc2x(scfres, core_orbitals=[1, 1 + nmo], n_singlets=10)
print(singlets.describe())
singlets.plot_spectrum(label="H2S CVS (2s)")

plt.legend()
plt.savefig("pyscf_ccpvtz_arbitrary_cvs_adc2x.pdf")
