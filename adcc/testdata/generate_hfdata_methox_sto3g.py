#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import sys

from pyscf import gto, scf
from os.path import dirname, join

from geometry import xyz

sys.path.insert(0, join(dirname(__file__), "adcc-testdata"))

import adcctestdata as atd  # noqa: E402

# Run SCF in pyscf and converge super-tight using an EDIIS
mol = gto.M(
    atom=xyz["r2methyloxirane"],
    basis='sto-3g',
    unit="Bohr",
    verbose=4
)
mf = scf.RHF(mol)
mf.diis = scf.EDIIS()
mf.conv_tol = 1e-13
mf.conv_tol_grad = 1e-12
mf.diis_space = 3
mf.max_cycle = 500
mf.kernel()
h5f = atd.dump_pyscf(mf, "methox_sto3g_hfdata.hdf5")

h5f["reference_cases"] = str({
    "gen":   {},
    "cvs":    {"core_orbitals":  1},
})
