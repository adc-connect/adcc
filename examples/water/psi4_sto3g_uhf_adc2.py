#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import psi4

import adcc

mol = psi4.geometry("""
    0 3
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    symmetry c1
    units au
    """)

# set the number of cores equal to the auto-determined value from
# the adcc ThreadPool
psi4.set_num_threads(adcc.thread_pool.n_cores)
psi4.core.be_quiet()
psi4.set_options({'basis': "sto-3g",
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'reference': 'uhf',
                  'd_convergence': 1e-8})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Run an adc2 calculation:
state = adcc.adc2(wfn, n_states=5)

print(state.describe())
