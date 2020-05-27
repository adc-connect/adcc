#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import psi4

from matplotlib import pyplot as plt

mol = psi4.geometry("""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    symmetry c1
    units au
    """)

# set the number of cores equal to the auto-determined value from
# the adcc ThreadPool
psi4.set_num_threads(adcc.get_n_threads())
psi4.core.be_quiet()
psi4.set_options({'basis': "cc-pvdz",
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-8})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Run an adc2 calculation:
state = adcc.cvs_adc2(wfn, n_singlets=5, core_orbitals=1)

print(state.describe())

state.plot_spectrum()
plt.savefig("psi4_ccpvdz_cvs_adc2_spectrum.pdf")
