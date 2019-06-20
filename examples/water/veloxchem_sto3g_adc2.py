#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import os
import tempfile

from mpi4py import MPI

import adcc
import veloxchem as vlx

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
                 basis: sto-3g
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
    task = vlx.MpiTask([infile, outfile], MPI.COMM_WORLD)
    scfdrv = vlx.ScfRestrictedDriver(task.mpi_comm, task.ostream)
    scfdrv.conv_thresh = 1e-8
    scfdrv.compute(task.molecule, task.ao_basis, task.min_basis)
    scfdrv.task = task

# Run an adc2 calculation:
state = adcc.adc2(scfdrv, n_singlets=5)
print(state.describe())
