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
import numpy as np
import adcc.backends.pyscf

from adcc import hdf5io
from pyscf import gto, scf
from geometry import xyz

# Run SCF in pyscf and converge super-tight using an EDIIS
mol = gto.M(
    atom=xyz["cn"],
    basis='cc-pvdz',
    unit="Bohr",
    spin=1,
    verbose=4
)
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-12
mf.diis = scf.EDIIS()
mf.diis_space = 3
mf.max_cycle = 600
mf = scf.addons.frac_occ(mf)
mf.kernel()
hfdict = adcc.backends.pyscf.convert_scf_to_dict(mf)

hfdict["reference_cases"] = {
    "gen":    {},
    "cvs":    {"core_orbitals":  1},
    "fc":     {"frozen_core":    1},
    "fv":     {"frozen_virtual": 3},
}

# Since CN has some symmetry some energy levels are degenerate,
# which can lead to all sort of inconsistencies. This code
# adds a fudge value of 1e-14 to make them numerically distinguishable
orben_f = hfdict["orben_f"]
for i in range(1, len(orben_f)):
    if np.abs(orben_f[i - 1] - orben_f[i]) < 1e-14:
        orben_f[i - 1] -= 1e-14
        orben_f[i] += 1e-14

hdf5io.save("cn_ccpvdz_hfdata.hdf5", hfdict)
