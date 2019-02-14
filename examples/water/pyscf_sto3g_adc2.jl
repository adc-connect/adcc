#!/usr/bin/env julia
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

# An example how to use adcc from julia

import PyCall

pyscf = PyCall.pyimport("pyscf")
adcc = PyCall.pyimport("adcc")

mol = pyscf[:gto][:M](
    atom="""
        O 0 0 0;
        H 0 0 1.795239827225189;
        H 1.693194615993441 0 -0.599043184453037
    """,
    basis="sto-3g",
    unit="Bohr"
)
scfres = pyscf[:scf][:RHF](mol)
scfres[:conv_tol] = 1e-14
scfres[:grad_conv_tol] = 1e-10
scfres[:kernel]()

# Initialise ADC memory (256 MiB)
adcc[:memory_pool][:initialise](max_memory=256 * 1024 * 1024)

# Run an adc2 calculation:
state = adcc[:adc2](scfres, n_singlets=5, n_triplets=3)

# Attach state densities
state = [adcc[:attach_state_densities](kstate) for kstate in state]

println(state[1][:describe]())
println(state[2][:describe]())
