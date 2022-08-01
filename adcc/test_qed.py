#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import unittest

from adcc.misc import expand_test_templates
import adcc

from numpy.testing import assert_allclose

from adcc.testdata.cache import cache
from adcc.testdata.cache import qed_data

import itertools
import pytest

# In principle one could also test the approx method against the full
# method, by expanding them to the full matrix dimension. The smallest
# example would be HF sto-3g, since it contains only one virtual orbital,
# which keeps the matrix dimension low. This is important since we
# build most of the matrix in the approx method from properties, so this
# is very slow for a lot of states, compared to the standard method.
# However, even this test case would take quite some time...

testcases = ["methox_sto3g", "h2o_sto3g"]
methods = ["adc2"]

@expand_test_templates(list(itertools.product(testcases, methods)))
class qed_test(unittest.TestCase):
    def set_refstate(self, case):
        self.refstate = cache.refstate[case]
        self.refstate.coupling = [0.0, 0.0, 0.05]
        self.refstate.frequency = [0.0, 0.0, 0.5]
        self.refstate.qed_hf = True

    def template_approx(self, case, method):
        self.set_refstate(case)
        self.refstate.approx = True

        approx = adcc.adc2(self.refstate, n_singlets = 5, conv_tol = 1e-7)

        ref_name = f"{case}_{method}_approx"
        approx_ref = qed_data[ref_name]["excitation_energy"]

        assert_allclose(approx.qed_excitation_energy,
                        approx_ref, atol=1e-6)

    def template_full(self, case, method):
        self.set_refstate(case)

        full = adcc.adc2(self.refstate, n_singlets = 3, conv_tol = 1e-7)

        ref_name = f"{case}_{method}_full"
        full_ref = qed_data[ref_name]["excitation_energy"]

        assert_allclose(full.excitation_energy,
                        full_ref, atol=1e-6)