#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import psi4

from scipy import constants
eV = constants.value("Hartree energy in eV")  # Hartree to eV

# Run SCF in psi4
mol = psi4.geometry("""
    C          8.64800        1.07500       -1.71100
    C          9.48200        0.43000       -0.80800
    C          9.39600        0.75000        0.53800
    C          8.48200        1.71200        0.99500
    C          7.65300        2.34500        0.05500
    C          7.73200        2.03100       -1.29200
    H         10.18300       -0.30900       -1.16400
    H         10.04400        0.25200        1.24700
    H          6.94200        3.08900        0.38900
    H          7.09700        2.51500       -2.01800
    N          8.40100        2.02500        2.32500
    N          8.73400        0.74100       -3.12900
    O          7.98000        1.33100       -3.90100
    O          9.55600       -0.11000       -3.46600
    H          7.74900        2.71100        2.65200
    H          8.99100        1.57500        2.99500
    symmetry c1
    no_reorient
    no_com
    """)

psi4.core.set_num_threads(4)
psi4.set_options({'basis': "sto-3g",
                  'scf_type': 'pk',
                  'pe': 'true',
                  'e_convergence': 1e-10,
                  'd_convergence': 1e-10})
psi4.set_module_options("pe", {"potfile": "pna_6w.pot"})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Run an adc2 calculation:
state = adcc.adc2(wfn, n_singlets=5, conv_tol=1e-8,
                  environment=True)
print(state.describe())
