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

import os
import tempfile
import pytest
import adcc
import unittest
import adcc.backends
import numpy as np

try:
    import psi4
    from pyscf import scf, gto
    from mpi4py import MPI
    import veloxchem as vlx
    _psi4_pyscf_vlx = True
except ImportError:
    _psi4_pyscf_vlx = False


@pytest.mark.skipif(not _psi4_pyscf_vlx, reason="psi4 or pyscf or vlx "
                    "not found.")
class TestBackendsFunctionality(unittest.TestCase):
    def run_psi4_hf(self, mol, basis=None):
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
        mf.conv_tol_grad = 1e-8
        mf.kernel()
        return mf

    def run_vlx_hf(self, xyz, basis, charge=0, multiplicity=1):
        basis_dir = os.path.abspath(os.path.join(vlx.__path__[-1],
                                                 "..", "..", "..", "basis"))
        with tempfile.TemporaryDirectory() as tmpdir:
            infile = os.path.join(tmpdir, "vlx.in")
            outfile = os.path.join(tmpdir, "vlx.out")
            with open(infile, "w") as fp:
                lines = ["@jobs", "task: hf", "@end", ""]
                lines += ["@method settings",
                          "basis: {}".format(basis),
                          "basis path: {}".format(basis_dir), "@end", ""]
                lines += ["@molecule",
                          "charge: {}".format(charge),
                          "multiplicity: {}".format(multiplicity),
                          "units: bohr",
                          "xyz:\n{}".format("\n".join(xyz.split(";"))),
                          "@end"]
                fp.write("\n".join(lines))
            task = vlx.MpiTask([infile, outfile], MPI.COMM_WORLD)

            scfdrv = vlx.ScfRestrictedDriver(task.mpi_comm, task.ostream)
            # elec. gradient norm
            scfdrv.conv_thresh = 1e-8
            scfdrv.compute(task.molecule, task.ao_basis, task.min_basis)
            scfdrv.task = task
        return scfdrv

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
        mol = psi4.geometry("""
            {}
            symmetry c1
            units au
            """.format(water_xyz))
        wfn = self.run_psi4_hf(mol, basis="aug-cc-pvdz")

        # run pyscf
        mol = gto.M(
            atom=water_xyz,
            basis='aug-cc-pvdz',
            unit="Bohr"
        )
        mf = self.run_pyscf_hf(mol)

        # cc-pvtz is wrong in VeloxChem
        scfdrv = self.run_vlx_hf(water_xyz, "aug-cc-pvdz")

        self.base_test(psi4_res=wfn, pyscf_res=mf, vlx_res=scfdrv)
