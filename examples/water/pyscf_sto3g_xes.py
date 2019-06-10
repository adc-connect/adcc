#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import copy
import numpy as np

from pyscf import gto, scf

mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='sto-3g',
    unit="Bohr",
    verbose=0,
)

# Run normal SCF in pyscf
mf = scf.UHF(mol)
mf.conv_tol = 1e-15
mf.conv_tol_grad = 1e-10
mf.kernel()
print("Water SCF energy", mf.energy_tot())

# Make a core hole
mo0 = copy.deepcopy(mf.mo_coeff)
occ0 = copy.deepcopy(mf.mo_occ)
occ0[0][0] = 0.0  # alpha core hole
dm = mf.make_rdm1(mo0, occ0)

mf_core = scf.UHF(mol)
mf_core.conv_tol = 1e-15
mf_core.conv_tol_grad = 1e-10
scf.addons.mom_occ(mf_core, mo0, occ0)
mf_core.kernel(dm)
del dm
print("Water core hole energy", mf_core.energy_tot())

# Run an adc2 calculation:
state = adcc.adc2(mf_core, n_states=4, conv_tol=5e-8)

# Attach properties
state = adcc.attach_properties(state)

# Print results in a nice way
print()
print("  st  ex.ene. (au)         f     transition dipole moment (au)"
      "        state dip (au)")
for i, ampl in enumerate(state.eigenvectors):
    osc = state.oscillator_strengths[i]
    tdip = state.transition_dipole_moments[i]
    sdip = state.state_dipole_moments[i]
    # Print findings
    fmt = "{0:2d}  {1:12.8g} {2:9.3g}   [{3:9.3g}, {4:9.3g}, {5:9.3g}]"
    fmt += "   [{6:9.3g}, {7:9.3g}, {8:9.3g}]"
    # fmt += "   [{9:9.3g}, {10:9.3g}, {11:9.3g}]"
    print(state.kind[0], fmt.format(i, state.eigenvalues[i], osc, *tdip, *sdip))
