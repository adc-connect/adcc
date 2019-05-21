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

import unittest
import pytest
from pytest import approx
import itertools

import numpy as np

import adcc
import adcc.backends
from adcc.backends import have_backend
from adcc.testdata.cache import cache
from adcc.solver.SolverStateBase import SolverStateBase
from ..misc import expand_test_templates

try:
    _psi4_pyscf_vlx = all(
        have_backend(b) for b in ["psi4", "veloxchem", "pyscf"]
    )
except ImportError:
    _psi4_pyscf_vlx = False


water_xyz = """
O 0 0 0
H 0 0 1.795239827225189
H 1.693194615993441 0 -0.599043184453037
"""


@pytest.mark.skipif(not _psi4_pyscf_vlx, reason="psi4 or pyscf or vlx "
                    "not found.")
class TestCrossReferenceBackends(unittest.TestCase):

    def base_test(self, psi4_res=None, pyscf_res=None, vlx_res=None):
        state_psi4 = adcc.adc2(psi4_res, n_singlets=5)
        state_pyscf = adcc.adc2(pyscf_res, n_singlets=5)
        state_vlx = adcc.adc2(vlx_res, n_singlets=5)

        for a, b, c in zip(state_psi4.eigenvalues,
                           state_pyscf.eigenvalues, state_vlx.eigenvalues):
            np.testing.assert_allclose(a, b)
            np.testing.assert_allclose(a, c)

    def base_test_cvs(self, psi4_res=None, pyscf_res=None, vlx_res=None):
        state_psi4 = adcc.cvs_adc2(psi4_res, n_singlets=5, n_core_orbitals=1)
        state_pyscf = adcc.cvs_adc2(pyscf_res, n_singlets=5, n_core_orbitals=1)
        state_vlx = adcc.cvs_adc2(vlx_res, n_singlets=5, n_core_orbitals=1)

        for a, b, c in zip(state_psi4.eigenvalues,
                           state_pyscf.eigenvalues, state_vlx.eigenvalues):
            np.testing.assert_allclose(a, b)
            np.testing.assert_allclose(a, c)

    def test_water_augccpvdz_rhf(self):
        # run psi4
        wfn = adcc.backends.run_hf("psi4", xyz=water_xyz, basis="aug-cc-pvdz")

        # run pyscf
        mf = adcc.backends.run_hf("pyscf", xyz=water_xyz, basis="aug-cc-pvdz")

        # cc-pvtz is wrong in VeloxChem
        scfdrv = adcc.backends.run_hf(
            "veloxchem", xyz=water_xyz, basis="aug-cc-pvdz"
        )
        self.base_test(psi4_res=wfn, pyscf_res=mf, vlx_res=scfdrv)
        self.base_test_cvs(psi4_res=wfn, pyscf_res=mf, vlx_res=scfdrv)


# The methods to test
methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]

# TODO: VeloxChem has convergence issues with water/def2-tzvp
# looks like the def2-tzvp basis file is wrong in VeloxChem
backends = ["pyscf", "psi4", "veloxchem"]

# for VeloxChem, because it only accepts with hypen
basis_dict = {
    "sto3g": "sto-3g",
    "def2tzvp": "def2-tzvp"
}


@expand_test_templates(list(itertools.product(methods, backends)))
class TestBackendsFunctionality(unittest.TestCase):
    def base_test(self, system, method, backend, kind, **args):
        basis = system.split("_")[-1]
        basis = basis_dict[basis]

        if basis == "def2-tzvp" and \
           backend == "veloxchem" and \
           system.startswith("h2o"):
            pytest.skip("VeloxChem does not support f-functions.")
        hfres = adcc.backends.run_hf(
            backend, xyz=water_xyz, basis=basis, conv_tol=1e-14,
            conv_tol_grad=1e-12
        )
        hf = adcc.backends.import_scf_results(hfres)
        refdata = cache.reference_data[system]

        args["conv_tol"] = 5e-8
        res = getattr(adcc, method.replace("-", "_"))(hf, **args)
        assert isinstance(res, SolverStateBase)

        ref = refdata[method][kind]["eigenvalues"]
        assert res.converged
        np.testing.assert_almost_equal(res.eigenvalues, ref)

        # TODO Compare transition dipole moment
        # TODO Compare excited state dipole moment

        # Test we do not use too many iterations
        if "sto3g" in system or "631g" in system:
            n_iter_bound = {
                "adc0": 1, "adc1": 4, "adc2": 6, "adc2x": 11, "adc3": 11,
                "cvs-adc0": 1, "cvs-adc1": 4, "cvs-adc2": 5, "cvs-adc2x": 11
            }
        else:
            n_iter_bound = {
                "adc0": 1, "adc1": 8, "adc2": 14, "adc2x": 14, "adc3": 14,
                "cvs-adc0": 1, "cvs-adc1": 6, "cvs-adc2": 14, "cvs-adc2x": 16
            }

        assert res.n_iter <= n_iter_bound[method]

    #
    # General
    #
    def template_h2o_sto3g_singlets(self, method, backend):
        self.base_test("h2o_sto3g", method, backend, "singlet", n_singlets=10)

    def template_h2o_def2tzvp_singlets(self, method, backend):
        self.base_test("h2o_def2tzvp", method, backend, "singlet", n_singlets=3)

    def template_h2o_sto3g_triplets(self, method, backend):
        self.base_test("h2o_sto3g", method, backend, "triplet", n_triplets=10)

    def template_h2o_def2tzvp_triplets(self, method, backend):
        self.base_test("h2o_def2tzvp", method, backend, "triplet", n_triplets=3)

    #
    # CVS
    #
    def template_cvs_h2o_sto3g_singlets(self, method, backend):
        n_singlets = 3
        if method in ["adc0", "adc1"]:
            n_singlets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, backend, "singlet",
                       n_singlets=n_singlets, n_core_orbitals=1)

    def template_cvs_h2o_def2tzvp_singlets(self, method, backend):
        self.base_test("h2o_def2tzvp", "cvs-" + method, backend, "singlet",
                       n_singlets=3, n_core_orbitals=1)

    def template_cvs_h2o_sto3g_triplets(self, method, backend):
        n_triplets = 3
        if method in ["adc0", "adc1"]:
            n_triplets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, backend, "triplet",
                       n_triplets=n_triplets, n_core_orbitals=1)

    def template_cvs_h2o_def2tzvp_triplets(self, method, backend):
        self.base_test("h2o_def2tzvp", "cvs-" + method, backend, "triplet",
                       n_triplets=3, n_core_orbitals=1)


# CVS-ADC(3) not supported
for backend in backends:
    delattr(
        TestBackendsFunctionality,
        "test_cvs_h2o_sto3g_singlets_adc3_{}".format(backend)
    )
    delattr(
        TestBackendsFunctionality,
        "test_cvs_h2o_sto3g_triplets_adc3_{}".format(backend)
    )
    delattr(
        TestBackendsFunctionality,
        "test_cvs_h2o_def2tzvp_singlets_adc3_{}".format(backend)
    )
    delattr(
        TestBackendsFunctionality,
        "test_cvs_h2o_def2tzvp_triplets_adc3_{}".format(backend)
    )
