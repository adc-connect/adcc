#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
from pyscf import gto, scf
from matplotlib import pyplot as plt

import adcc

# pyscf-H2O Hartree-Fock calculation
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvtz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()


plt.figure(figsize=(7, 5))
state = adcc.adc2(scfres, n_singlets=10)
state.plot_spectrum()

plt.savefig("plot_spectrum_water.png", bbox="tight")
plt.close()
