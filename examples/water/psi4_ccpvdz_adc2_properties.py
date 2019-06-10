#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import numpy as np

from scipy import constants
from matplotlib import pyplot as plt

import adcc

import psi4

# Hartree to eV
eV = constants.value("Hartree energy in eV")


def plot_spectrum(energies, strengths, width=0.045):
    energies = np.array(energies).flatten()
    strengths = np.array(strengths).flatten()

    def ngauss(en, osc, x, w):
        """A normalised Gaussian"""
        fac = osc / np.sqrt(2 * np.pi * w**2)
        return fac * np.exp(-(x - en)**2 / (2 * w**2))

    xval = np.arange(np.min(np.min(energies) - 1, 0),
                     np.max(energies) + 1, 0.01)
    yval = np.zeros(xval.size)
    for en, osc in zip(energies, strengths):
        yval += ngauss(en, osc, xval, width)
    plt.plot(xval, yval)


#
# Run SCF in psi4
#
mol = psi4.geometry("""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    symmetry c1
    units au
    no_reorient
    no_com
    """)

# set the number of cores equal to the auto-determined value from
# the adcc ThreadPool
psi4.set_num_threads(adcc.thread_pool.n_cores)
psi4.core.be_quiet()
psi4.set_options({'basis': "cc-pvdz",
                  'scf_type': 'pk',
                  'e_convergence': 1e-14,
                  'd_convergence': 1e-9})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

print(adcc.banner())

# Run an adc2 calculation:
state = adcc.adc2(wfn, n_singlets=7, conv_tol=1e-8)
state = adcc.attach_properties(state)

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

# Plot a spectrum
plot_spectrum(state.eigenvalues * eV, state.oscillator_strengths)
plt.xlabel("Excitation energy in eV")
plt.savefig("spectrum.pdf")

# Timings summary:
print(state.timer.describe())
print(state.reference_state.timer.describe())
