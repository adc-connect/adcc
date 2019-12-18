#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import psi4

from matplotlib import pyplot as plt

# Run SCF in psi4
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
psi4.set_num_threads(adcc.get_n_threads())
psi4.core.be_quiet()
psi4.set_options({'basis': "cc-pvdz",
                  'scf_type': 'pk',
                  'e_convergence': 1e-14,
                  'd_convergence': 1e-9})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

print(adcc.banner())

# Run an adc2 calculation:
state = adcc.adc2(wfn, n_singlets=7, conv_tol=1e-8)

# Print results
print()
print("  st  ex.ene. (au)         f     transition dipole moment (au)"
      "        state dip (au)")
for i, val in enumerate(state.excitation_energies):
    fmt = "{0:2d}  {1:12.8g} {2:9.3g}   [{3:9.3g}, {4:9.3g}, {5:9.3g}]"
    fmt += "   [{6:9.3g}, {7:9.3g}, {8:9.3g}]"
    print(state.kind[0], fmt.format(i, val, state.oscillator_strengths[i],
                                    *state.transition_dipole_moments[i],
                                    *state.state_dipole_moments[i]))

state.plot_spectrum()
plt.savefig("psi4_ccpvdz_adc2_spectrum.pdf")
plt.show()

# Print timings summary:
print()
print(state.timer.describe())
