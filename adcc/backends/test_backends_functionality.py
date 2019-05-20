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

import pytest
import adcc
import unittest
import adcc.backends
import numpy as np

try:
    import psi4
    from pyscf import scf, gto
    import veloxchem as vlx
    from . import utils
    _psi4_pyscf_vlx = True
except ImportError:
    _psi4_pyscf_vlx = False


@pytest.mark.skipif(not _psi4_pyscf_vlx, reason="psi4 or pyscf or vlx "
                    "not found.")
class TestBackendsFunctionality(unittest.TestCase):

    def base_test(self, psi4_res=None, pyscf_res=None, vlx_res=None):
        state_psi4 = adcc.adc2(psi4_res, n_singlets=10)
        state_pyscf = adcc.adc2(pyscf_res, n_singlets=10)
        state_vlx = adcc.adc2(vlx_res, n_singlets=10)

        for a, b, c in zip(state_psi4.eigenvalues,
                           state_pyscf.eigenvalues, state_vlx.eigenvalues):
            np.testing.assert_allclose(a, b)
            np.testing.assert_allclose(a, c)

    def test_water_sto3g_rhf(self):
        water_xyz = """
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
        """

        # run psi4
        wfn = utils.run_psi4_hf(water_xyz, basis="aug-cc-pvdz")

        # run pyscf
        mf = utils.run_pyscf_hf(water_xyz, basis="aug-cc-pvdz")

        # cc-pvtz is wrong in VeloxChem
        scfdrv = utils.run_vlx_hf(water_xyz, basis="aug-cc-pvdz")
        self.base_test(psi4_res=wfn, pyscf_res=mf, vlx_res=scfdrv)
