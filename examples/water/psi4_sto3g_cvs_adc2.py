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

import psi4
import adcc

mol = psi4.geometry("""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    symmetry c1
    units au
    """)

# set the number of cores equal to the auto-determined value from
# the adcc ThreadPool
psi4.set_num_threads(adcc.thread_pool.n_cores)
psi4.core.be_quiet()
psi4.set_options({'basis': "sto-3g",
                  'scf_type': 'pk',
                  'e_convergence': 1e-12,
                  'd_convergence': 1e-8})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Run an adc2 calculation:
state = adcc.cvs_adc2(wfn, n_singlets=5, n_core_orbitals=1)
