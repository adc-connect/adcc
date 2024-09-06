#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import numpy as np

from numpy.testing import assert_allclose
from adcc.testdata.cache import cache, qchem_data
from adcc.misc import expand_test_templates

import pytest


methods = ["ip_adc2", "ip_adc3", "ea_adc2", "ea_adc3"]
@expand_test_templates(methods)
class Runners_IP_EA_QChem():
    def base_test(self, *args, **kwargs):
        raise NotImplementedError

    def template_h2o_sto3g_doublet(self, method):
        self.base_test("h2o_sto3g", method, "doublet")


@expand_test_templates(methods)
class TestIpEaQChem(unittest.TestCase, Runners_IP_EA_QChem):
    def base_test(self, system, method, kind, prefix=""):
        basename = f"h2o_sto3g_{method}"
        qc_result = qchem_data[basename]
        
        method = prefix + method
        state = cache.adcc_states[system][method][kind]
        res_energies = state.excitation_energy
        res_poles = state.pole_strength

        min_states = min(len(res_energies),
                         len(qc_result["excitation_energy"]))
        assert_allclose(res_energies[:min_states],
                        qc_result["excitation_energy"][:min_states], atol=1e-6)
        assert_allclose(res_poles[:min_states],
                        qc_result["pole_strength"][:min_states], atol=1e-4)
