#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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

import molsturm
import adcc
import IPython

# Run SCF in molsturm
atoms = ["O", "H", "H"]
coords = [[0, 0, 0],
          [0, 0, 1.795239827225189],
          [1.693194615993441, 0, -0.599043184453037]]
system = molsturm.System(atoms, coords)

hfres = molsturm.hartree_fock(system, basis_type="gaussian",
                              basis_set_name="sto-3g",
                              conv_tol=1e-12, print_iterations=True)

# Initialise ADC memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc2 calculation:
state = adcc.adc2(hfres, n_singlets=5, n_triplets=3)

# Attach state densities
state = [adcc.attach_state_densities(kstate) for kstate in state]

IPython.embed()
