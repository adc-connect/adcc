#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
from dump_pyscf import dump_pyscf
import test_cases

from pathlib import Path

from pyscf import gto, scf

data = test_cases.get(n_expected_cases=1, name="h2o", basis="sto-3g").pop()
hdf5_file = Path(__file__).resolve().parent.parent / "data"
hdf5_file /= f"{data.file_name}_hfdata.hdf5"

# Run SCF in pyscf and converge super-tight using an EDIIS
mol = gto.M(
    atom=data.xyz,
    basis=data.basis,
    unit=data.unit
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-13
mf.conv_tol_grad = 1e-12
mf.diis = scf.EDIIS()
mf.diis_space = 3
mf.max_cycle = 500
mf.kernel()
h5f = dump_pyscf(mf, str(hdf5_file))

# Store configuration parameters for interesting cases to generate
# reference data for. The data is stored as a stringified dict.
h5f["reference_cases"] = str({
    "gen":    {},
    "cvs":    {"core_orbitals":  1},
    "fc":     {"frozen_core":    1},
    "fv":     {"frozen_virtual": 1},
    "fv-cvs": {"core_orbitals":  1, "frozen_virtual": 1},
    "fc-fv":  {"frozen_core":    1, "frozen_virtual": 1},
})