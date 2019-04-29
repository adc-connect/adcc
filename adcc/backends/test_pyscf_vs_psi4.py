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
    _psi4_pyscf = True
except ImportError:
    _psi4_pyscf = False


@pytest.mark.skipif(not _psi4_pyscf, reason="psi4 or pyscf not found.")
class TestPyscfVsPsi4(unittest.TestCase):
    def run_psi_hf(self, mol, basis=None):
        psi4.core.be_quiet()
        psi4.set_options({'basis': basis,
                          'scf_type': 'pk',
                          'e_convergence': 1e-12,
                          'd_convergence': 1e-8})
        scf_e, wfn = psi4.energy('SCF', return_wfn=True)
        return wfn

    def run_pyscf_hf(self, mol):
        mf = scf.HF(mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-7
        mf.kernel()
        return mf

    def base_test(self, psi4_res=None, pyscf_res=None):
        psi4_res = adcc.backends.import_scf_results(psi4_res)
        state_psi4 = adcc.adc2(psi4_res, n_singlets=3)
        state_pyscf = adcc.adc2(pyscf_res, n_singlets=3)

        for a, b in zip(state_psi4.eigenvalues,
                        state_pyscf.eigenvalues):
            np.testing.assert_allclose(a, b)

    def test_water_sto3g_rhf(self):
        water_xyz = """
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
        """

        # run psi4
        mol = psi4.geometry("""
            {}
            symmetry c1
            units au
            """.format(water_xyz))
        wfn = self.run_psi_hf(mol, basis="sto-3g")

        # run pyscf
        mol = gto.M(
            atom=water_xyz,
            basis='sto-3g',
            unit="Bohr"
        )
        mf = self.run_pyscf_hf(mol)

        self.base_test(psi4_res=wfn, pyscf_res=mf)
