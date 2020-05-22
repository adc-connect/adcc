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
import itertools
import numpy as np
import adcc
import adcc.backends

from numpy.testing import assert_allclose

import pytest

from ..misc import expand_test_templates
from .testing import cached_backend_hf
from ..testdata.static_data import pe_potentials

backends = [b for b in adcc.backends.available() if b != "molsturm"]
basissets = ["sto3g", "cc-pvdz"]


@pytest.mark.skipif(len(backends) == 0,
                    reason="No backend found.")
@expand_test_templates(basissets)
class TestPolarizableEmbedding(unittest.TestCase):
    def template_pe_adc2_formaldehyde(self, basis):
        results = {}
        for b in backends:
            print(b)
            scfres = cached_backend_hf(b, "formaldehyde", basis,
                                       potfile=pe_potentials["fa_6w"])
            results[b] = adcc.adc2(scfres, n_singlets=5, conv_tol=1e-10)
            print(results[b].pe_ptss_correction)