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
from adcc.backends.testing import cached_backend_hf
from adcc.testdata.cache import gradient_data


backends = [b for b in adcc.backends.available()
            if b not in ["molsturm", "veloxchem"]]

molecules = gradient_data["molecules"]
basissets = gradient_data["basissets"]
methods = gradient_data["methods"]

combinations = list(itertools.product(molecules, basissets, methods, backends))


@pytest.mark.skipif(len(backends) == 0, reason="No backend found.")
@expand_test_templates(combinations)
class TestNuclearGradients(unittest.TestCase):
    def template_nuclear_gradient(self, molecule, basis, method, backend):
        grad_ref = gradient_data[molecule][basis][method]

        energy_ref = grad_ref["energy"]
        grad_fdiff = grad_ref["gradient"] 
        kwargs = grad_ref["config"]
        conv_tol = kwargs["conv_tol"]

        scfres = cached_backend_hf(backend, molecule, basis, conv_tol=1e-13)
        if "adc" in method:
            # TODO: convergence needs to be very very tight...
            # so we want to make sure all vectors are tightly converged
            n_limit = 2 # kwargs["n_singlets"]
            kwargs["n_singlets"] = kwargs["n_singlets"] + 2
            state = adcc.run_adc(scfres, method=method, **kwargs)
            for ee in state.excitations[:n_limit]:
                grad = adcc.nuclear_gradient(ee)
                assert_allclose(energy_ref[ee.index], ee.total_energy, atol=conv_tol)
                assert_allclose(
                    grad_fdiff[ee.index], grad["Total"], atol=1e-7,
                    err_msg=f'Gradient for state {ee.index} wrong.'
                )
        else:
            # MP2 gradients
            refstate = adcc.ReferenceState(scfres)
            mp = adcc.LazyMp(refstate)
            grad = adcc.nuclear_gradient(mp)
            assert_allclose(energy_ref, mp.energy(2), atol=1e-8)
            assert_allclose(
                grad_fdiff, grad["Total"], atol=1e-8
            )
