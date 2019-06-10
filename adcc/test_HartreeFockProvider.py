#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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

from adcc.DictHfProvider import DictHfProvider
from adcc.testdata.cache import cache
from adcc.solver.SolverStateBase import SolverStateBase

from pytest import approx


class TestHartreeFockProvider(unittest.TestCase):
    def base_test(self, system, **args):
        hf = DictHfProvider(cache.hfdata[system])
        refdata = cache.reference_data[system]

        res = adcc.adc2(hf, n_singlets=9, **args)
        assert isinstance(res, SolverStateBase)

        ref = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref)

    def test_h2o(self):
        self.base_test("h2o_sto3g")
