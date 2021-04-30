#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab

import adcc
from pyscf import gto, scf
from pyscf.solvent import PE

from scipy import constants
eV = constants.value("Hartree energy in eV")  # Hartree to eV

mol = gto.M(
    atom="""
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
    """,
    basis='sto-3g',
)

scfres = PE(scf.RHF(mol), {"potfile": "pna_6w.pot"})
scfres.conv_tol = 1e-8
scfres.conv_tol_grad = 1e-6
scfres.max_cycle = 250
scfres.kernel()

# model the solvent through perturbative corrections
state_pt = adcc.adc2(scfres, n_singlets=5, conv_tol=1e-5,
                     environment=['ptss', 'ptlr'])

# now model the solvent through linear-response coupling
# in the ADC matrix, re-using the matrix from previous run.
# This will modify state_pt.matrix
state_lr = adcc.run_adc(state_pt.matrix, n_singlets=5, conv_tol=1e-5,
                        environment='linear_response')

print(state_pt.describe())
print(state_lr.describe())
