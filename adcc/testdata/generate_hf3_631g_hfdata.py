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
import adcc.backends.pyscf

from adcc import hdf5io

from pyscf import gto, scf

# Run SCF in pyscf and converge super-tight using an EDIIS
mol = gto.M(
    atom='H 0 0 0;'
         'F 0 0 2.5',
    basis='6-31G',
    unit="Bohr",
    spin=2,  # =2S, ergo triplet
    verbose=4
)
mf = scf.UHF(mol)
mf.diis = scf.EDIIS()
mf.conv_tol = 1e-14
mf.conv_tol_grad = 1e-12
mf.diis_space = 3
mf.max_cycle = 500
mf.kernel()

hfdict = adcc.backends.pyscf.convert_scf_to_dict(mf)
hdf5io.save("hf3_631g_hfdata.hdf5", hfdict)
