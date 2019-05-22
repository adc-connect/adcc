#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

from mpi4py import MPI
import veloxchem as vlx
import adcc
import IPython
import os
import tempfile

# Run SCF in pyscf
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

# Attach state densities
state = adcc.attach_state_densities(state)

IPython.embed()
