#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import os
import adcc
import tempfile

from mpi4py import MPI
from matplotlib import pyplot as plt

import veloxchem as vlx
from veloxchem.mpitask import MpiTask

# Run SCF in VeloxChem
with tempfile.TemporaryDirectory() as tmpdir:
    infile = os.path.join(tmpdir, "vlx.in")
    outfile = os.path.join(tmpdir, "/dev/null")

    with open(infile, "w") as fp:
        fp.write("""
                 @jobs
                 task: hf
                 @end

                 @method settings
                 basis: cc-pvdz
                 @end

                 @molecule
                 charge: 0
                 multiplicity: 1
                 units: bohr
                 xyz:
                 O 0 0 0
                 H 0 0 1.795239827225189
                 H 1.693194615993441 0 -0.599043184453037
                 @end
                 """)
    task = MpiTask([infile, outfile], MPI.COMM_WORLD)
    scfdrv = vlx.ScfRestrictedDriver(task.mpi_comm, task.ostream)
    scfdrv.conv_thresh = 1e-9
    scfdrv.compute(task.molecule, task.ao_basis, task.min_basis)
    scfdrv.task = task

print(adcc.banner())

# Run an adc2 calculation:
state = adcc.adc2(scfdrv, n_singlets=7, conv_tol=1e-8)

print()
print("  st  ex.ene. (au)         f     transition dipole moment (au)"
      "        state dip (au)")
for i, val in enumerate(state.excitation_energy):
    fmt = "{0:2d}  {1:12.8g} {2:9.3g}   [{3:9.3g}, {4:9.3g}, {5:9.3g}]"
    fmt += "   [{6:9.3g}, {7:9.3g}, {8:9.3g}]"
    print(state.kind[0], fmt.format(i, val, state.oscillator_strength[i],
                                    *state.transition_dipole_moment[i],
                                    *state.state_dipole_moment[i]))

# Plot a spectrum
state.plot_spectrum()
plt.savefig("veloxchem_ccpvdz_adc2_spectrum.pdf")
plt.show()

print()
print(state.timer.describe())
