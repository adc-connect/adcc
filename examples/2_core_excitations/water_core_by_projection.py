#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import adcc.projection

from pyscf import gto, scf
from adcc.AdcMatrix import AdcMatrixProjected

# Aim of this script is to compute core excitations of water
# using a projection approach, i.e. where the non-CVS ADC matrix
# is projected into parts of the orbital subspaces to target core
# excitations specifically.

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvtz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()
print(adcc.banner())

# Construct the standard ADC(2)-x matrix and project it, such that
# only cv, ccvv and ocvv excitations are allowed. Choose one orbital (oxygen 1s)
# to make up the core space.
matrix = adcc.AdcMatrix("adc2x", adcc.ReferenceState(scfres))
pmatrix = AdcMatrixProjected(matrix, ["cv", "ccvv", "ocvv"], core_orbitals=1)

# Run the ADC calculation. Note that doubles guesses are not yet supported in this
# setup and like lead to numerical issues (zero excitations) if selected. So we
# explicitly disable them here.
state_proj = adcc.run_adc(pmatrix, n_singlets=7, conv_tol=1e-8, n_guesses_doubles=0)
print(state_proj.describe())

# For comparison we also run a standard CVS-ADC(2)-x calculation:
state_cvs = adcc.cvs_adc2x(scfres, n_singlets=7, conv_tol=1e-8, core_orbitals=1)
print(state_cvs.describe())

# Note that this calculation could also be used as an initial guess:
guesses = adcc.projection.transfer_cvs_to_full(state_cvs, pmatrix)
state_proj2 = adcc.run_adc(pmatrix, n_singlets=7, conv_tol=1e-8, guesses=guesses)
