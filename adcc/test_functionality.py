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

from adcc.testdata.cache import cache
from adcc.solver.SolverStateBase import SolverStateBase

from .misc import expand_test_templates
from pytest import approx

# The methods to test
methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]


@expand_test_templates(methods)
class TestFunctionality(unittest.TestCase):
    def base_test(self, system, method, kind, **args):
        hf = cache.hfdata[system]
        refdata = cache.reference_data[system]

        args["conv_tol"] = 5e-8
        res = getattr(adcc, method.replace("-", "_"))(hf, **args)
        assert isinstance(res, SolverStateBase)

        ref = refdata[method][kind]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref, abs=1e-7)

        # TODO Compare transition dipole moment
        # TODO Compare excited state dipole moment

    #
    # General
    #
    def template_h2o_singlets(self, method):
        self.base_test("h2o_sto3g", method, "singlet", n_singlets=10)

    def template_h2o_triplets(self, method):
        self.base_test("h2o_sto3g", method, "triplet", n_triplets=10)

    def template_cn(self, method):
        kwargs = {}
        if method in ["adc0", "adc1"]:
            kwargs["max_subspace"] = 42
        self.base_test("cn_sto3g", method, "state", n_states=8, **kwargs)

    #
    # CVS
    #
    def template_cvs_h2o_singlets(self, method):
        n_singlets = 3
        if method in ["adc0", "adc1"]:
            n_singlets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, "singlet",
                       n_singlets=n_singlets, n_core_orbitals=1)

    def template_cvs_h2o_triplets(self, method):
        n_triplets = 3
        if method in ["adc0", "adc1"]:
            n_triplets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, "triplet",
                       n_triplets=n_triplets, n_core_orbitals=1)

    def template_cvs_cn(self, method):
        self.base_test("cn_sto3g", "cvs-" + method, "state",
                       n_states=6, n_core_orbitals=1)

    #
    # Spin-flip
    #
    def template_hf3_spin_flip(self, method):
        self.base_test("hf3_631g", method, "spin_flip", n_spin_flip=9)


# CVS-ADC(3) not supported
delattr(TestFunctionality, "test_cvs_cn_adc3")
delattr(TestFunctionality, "test_cvs_h2o_singlets_adc3")
delattr(TestFunctionality, "test_cvs_h2o_triplets_adc3")
