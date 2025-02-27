#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import adcc
import unittest
from pytest import approx

from adcc import ExcitedStates
from adcc.DataHfProvider import DataHfProvider

from .testdata_cache import testdata_cache


class TestHartreeFockProvider(unittest.TestCase):
    def base_test(self, system: str, **args):
        hf = DataHfProvider(testdata_cache._load_hfdata(system))
        refdata = testdata_cache.adcman_data(
            system, method="adc2", case="gen"
        )["singlet"]

        res = adcc.adc2(hf, n_singlets=9, **args)
        assert isinstance(res, ExcitedStates)
        assert res.converged

        ref = refdata["eigenvalues"]
        assert res.excitation_energy[:len(ref)] == approx(ref)

        refdata = testdata_cache.adcc_data(
            system, method="adc2", case="gen"
        )["singlet"]
        ref = refdata["eigenvalues"]
        assert res.excitation_energy[:len(ref)] == approx(ref)

    def test_h2o(self):
        self.base_test("h2o_sto3g")
