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

from adcc.testdata.cache import cache
from pytest import approx
import adcc
import unittest


class TestFunctionality(unittest.TestCase):
    def base_test(self, system, method, kind, **args):
        hf = cache.hfdata[system]
        refdata = cache.reference_data[system]

        res = getattr(adcc, method.replace("-", "_"))(hf, **args)
        assert type(res) == list
        assert len(res) == 1
        res = res[0]  # Extract the first (and only state)

        ref = refdata[method][kind]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref)

        # TODO Compare transition dipole moment
        # TODO Compare excited state dipole moment

    #
    # General
    #
    def test_h2o_adc2_singlets(self):
        self.base_test("h2o_sto3g", "adc2", "singlet", n_singlets=10)

    def test_h2o_adc2_triplets(self):
        self.base_test("h2o_sto3g", "adc2", "triplet", n_triplets=10)

    def test_cn_adc2(self):
        self.base_test("cn_sto3g", "adc2", "state", n_states=8)

    def test_h2o_adc3_singlets(self):
        self.base_test("h2o_sto3g", "adc3", "singlet", n_singlets=10)

    def test_h2o_adc3_triplets(self):
        self.base_test("h2o_sto3g", "adc3", "triplet", n_triplets=10)

    def test_cn_adc3(self):
        self.base_test("cn_sto3g", "adc3", "state", n_states=8)

    def test_h2o_adc2x_singlets(self):
        self.base_test("h2o_sto3g", "adc2x", "singlet", n_singlets=10)

    def test_h2o_adc2x_triplets(self):
        self.base_test("h2o_sto3g", "adc2x", "triplet", n_triplets=10)

    def test_cn_adc2x(self):
        self.base_test("cn_sto3g", "adc2x", "state", n_states=8)

    #
    # CVS
    #
    def test_h2o_cvs_adc2_singlets(self):
        self.base_test("h2o_sto3g", "cvs-adc2", "singlet", n_singlets=3,
                       n_core_orbitals=1)

    def test_h2o_cvs_adc2_triplets(self):
        self.base_test("h2o_sto3g", "cvs-adc2", "triplet", n_triplets=3,
                       n_core_orbitals=1)

    def test_cn_cvs_adc2(self):
        self.base_test("cn_sto3g", "cvs-adc2", "state", n_states=6,
                       n_core_orbitals=1)

    def test_h2o_cvs_adc2x_singlets(self):
        self.base_test("h2o_sto3g", "cvs-adc2x", "singlet", n_singlets=3,
                       n_core_orbitals=1)

    def test_h2o_cvs_adc2x_triplets(self):
        self.base_test("h2o_sto3g", "cvs-adc2x", "triplet", n_triplets=3,
                       n_core_orbitals=1)

    def test_cn_cvs_adc2x(self):
        self.base_test("cn_sto3g", "cvs-adc2x", "state", n_states=6,
                       n_core_orbitals=1)
