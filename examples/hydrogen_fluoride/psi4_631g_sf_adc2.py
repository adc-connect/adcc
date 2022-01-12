#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import psi4

# Run SCF in psi4
mol = psi4.geometry("""
    0 3
    H 0 0 0
    F 0 0 2.5
    symmetry c1
    units au
    no_reorient
    no_com
    """)

psi4.set_num_threads(adcc.get_n_threads())
psi4.core.be_quiet()
psi4.set_options({'basis': "6-31g",
                  'e_convergence': 1e-14,
                  'd_convergence': 1e-9,
                  'reference': 'uhf'})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Run solver and print results
states = adcc.adc2(wfn, n_spin_flip=5, conv_tol=1e-8)
print(states.describe())
print(states.describe_amplitudes())
