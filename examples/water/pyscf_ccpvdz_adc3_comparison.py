#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from pyscf import gto, scf
from matplotlib import pyplot as plt

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvdz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()

# Run an adc3 calculation:
singlets     = adcc.adc3(scfres,                   n_singlets=3)
singlets_fc  = adcc.adc3(scfres, frozen_core=1,    n_singlets=3)
singlets_fv2 = adcc.adc3(scfres, frozen_virtual=2, n_singlets=3)
singlets_fv4 = adcc.adc3(scfres, frozen_virtual=4, n_singlets=3)
singlets_fv6 = adcc.adc3(scfres, frozen_virtual=6, n_singlets=3)
singlets_fv8 = adcc.adc3(scfres, frozen_virtual=8, n_singlets=3)
singlets.plot_spectrum(label="ADC(3)")
singlets_fc.plot_spectrum(label="FC-ADC(3)")
singlets_fv2.plot_spectrum(label="FV-ADC(3) 2")
singlets_fv4.plot_spectrum(label="FV-ADC(3) 4")
singlets_fv6.plot_spectrum(label="FV-ADC(3) 6")
singlets_fv8.plot_spectrum(label="FV-ADC(3) 8")

plt.legend()
plt.show()
