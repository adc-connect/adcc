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
import adcc
import unittest
import numpy as np

from numpy.testing import assert_allclose

from adcc.testdata.cache import cache
from adcc.solver.SolverStateBase import SolverStateBase

import pytest

from .misc import assert_allclose_signfix, expand_test_templates
from pytest import approx

# The methods to test
methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]


@expand_test_templates(methods)
class TestFunctionality(unittest.TestCase):
    def base_test(self, system, method, kind, **args):
        hf = cache.hfdata[system]
        refdata = cache.reference_data[system]
        ref = refdata[method][kind]

        n_ref = len(ref["eigenvalues"])
        smallsystem = ("sto3g" in system or "631g" in system)

        # Run ADC and properties
        args["conv_tol"] = 3e-9
        res = getattr(adcc, method.replace("-", "_"))(hf, **args)
        if res.method.level > 2:
            res = adcc.attach_properties(res, method="adc2")
        else:
            res = adcc.attach_properties(res)

        # Checks
        assert isinstance(res, SolverStateBase)
        assert res.converged
        assert_allclose(res.eigenvalues[:n_ref], ref["eigenvalues"], atol=1e-7)

        if method == "adc0" and "cn" in system:
            # TODO Investigate this
            pytest.xfail("CN adc0 transition properties currently fail for "
                         "unknown reasons and are thus not explicitly tested.")

        for i in range(n_ref):
            # Computing the dipole moment implies a lot of cancelling in the
            # contraction, which has quite an impact on the accuracy.
            res_tdm = res.transition_dipole_moments[i]
            ref_tdm = ref["transition_dipole_moments"][i]

            # Test norm and actual values
            res_tdm_norm = np.sum(res_tdm * res_tdm)
            ref_tdm_norm = np.sum(ref_tdm * ref_tdm)
            assert res_tdm_norm == approx(ref_tdm_norm, abs=1e-5)

            # If the eigenpair is degenerate, then some rotation
            # in the eigenspace is possible, which reflects as a
            # rotation inside the dipole moments. This is the case
            # for example for the CN test system. For simplicity,
            # we only compare the norm of the transition dipole moment
            # in such cases and skip the test for the exact values.
            if "cn" not in system:
                assert_allclose_signfix(res_tdm, ref_tdm, atol=1e-5)

        # Computing the dipole moment implies a lot of cancelling in the
        # contraction, which has quite an impact on the accuracy.
        assert_allclose(res.state_dipole_moments[:n_ref],
                        ref["state_dipole_moments"], atol=1e-4)

        # Test we do not use too many iterations
        if smallsystem:
            n_iter_bound = {
                "adc0": 1, "adc1": 4, "adc2": 6, "adc2x": 13, "adc3": 13,
                "cvs-adc0": 1, "cvs-adc1": 4, "cvs-adc2": 5, "cvs-adc2x": 12
            }[method]
        else:
            n_iter_bound = {
                "adc0": 1, "adc1": 8, "adc2": 16, "adc2x": 17, "adc3": 17,
                "cvs-adc0": 1, "cvs-adc1": 7, "cvs-adc2": 16, "cvs-adc2x": 18
            }[method]
        assert res.n_iter <= n_iter_bound

    #
    # General
    #
    def template_h2o_sto3g_singlets(self, method):
        self.base_test("h2o_sto3g", method, "singlet", n_singlets=10)

    def template_h2o_def2tzvp_singlets(self, method):
        self.base_test("h2o_def2tzvp", method, "singlet", n_singlets=3)

    def template_h2o_sto3g_triplets(self, method):
        self.base_test("h2o_sto3g", method, "triplet", n_triplets=10)

    def template_h2o_def2tzvp_triplets(self, method):
        self.base_test("h2o_def2tzvp", method, "triplet", n_triplets=3)

    def template_cn_sto3g(self, method):
        kwargs = {}
        if method in ["adc0", "adc1"]:
            kwargs["max_subspace"] = 42
        self.base_test("cn_sto3g", method, "state", n_states=8, **kwargs)

    def template_cn_ccpvdz(self, method):
        kwargs = {}
        n_states = 5
        if method == "adc1":
            kwargs["max_subspace"] = 42
            n_states = 4
        self.base_test("cn_ccpvdz", method, "state",
                       n_states=n_states, **kwargs)

    #
    # CVS
    #
    def template_cvs_h2o_sto3g_singlets(self, method):
        n_singlets = 3
        if method in ["adc0", "adc1"]:
            n_singlets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, "singlet",
                       n_singlets=n_singlets, n_core_orbitals=1)

    def template_cvs_h2o_def2tzvp_singlets(self, method):
        self.base_test("h2o_def2tzvp", "cvs-" + method, "singlet", n_singlets=3,
                       n_core_orbitals=1)

    def template_cvs_h2o_sto3g_triplets(self, method):
        n_triplets = 3
        if method in ["adc0", "adc1"]:
            n_triplets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, "triplet",
                       n_triplets=n_triplets, n_core_orbitals=1)

    def template_cvs_h2o_def2tzvp_triplets(self, method):
        self.base_test("h2o_def2tzvp", "cvs-" + method, "triplet", n_triplets=4,
                       n_core_orbitals=1)

    def template_cvs_cn_sto3g(self, method):
        self.base_test("cn_sto3g", "cvs-" + method, "state",
                       n_states=6, n_core_orbitals=1)

    def template_cvs_cn_ccpvdz(self, method):
        kwargs = {}
        if method in ["adc0", "adc1"]:
            kwargs["max_subspace"] = 28
        self.base_test("cn_ccpvdz", "cvs-" + method, "state", n_states=5,
                       n_core_orbitals=1, **kwargs)

    #
    # Spin-flip
    #
    def template_hf3_spin_flip(self, method):
        self.base_test("hf3_631g", method, "spin_flip", n_spin_flip=9)


# CVS-ADC(3) not supported
delattr(TestFunctionality, "test_cvs_cn_sto3g_adc3")
delattr(TestFunctionality, "test_cvs_cn_ccpvdz_adc3")
delattr(TestFunctionality, "test_cvs_h2o_sto3g_singlets_adc3")
delattr(TestFunctionality, "test_cvs_h2o_sto3g_triplets_adc3")
delattr(TestFunctionality, "test_cvs_h2o_def2tzvp_singlets_adc3")
delattr(TestFunctionality, "test_cvs_h2o_def2tzvp_triplets_adc3")
