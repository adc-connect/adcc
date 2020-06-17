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
import numpy as np

from .misc import expand_test_templates
from numpy.testing import assert_allclose
from adcc.testdata.cache import cache

methods = ["adc1", "adc2", "adc2x", "adc3"]
methods += ["cvs-" + m for m in methods]


@expand_test_templates(methods)
class TestAdcMatrixDenseExport(unittest.TestCase):
    def base_test(self, case, method, conv_tol=1e-8, **kwargs):
        kwargs.setdefault("n_states", 10)
        n_states = kwargs["n_states"]
        if "cvs" in method:
            refstate = cache.refstate_cvs[case]
        else:
            refstate = cache.refstate[case]

        matrix = adcc.AdcMatrix(method, refstate)
        state = adcc.run_adc(matrix, method=method, conv_tol=conv_tol, **kwargs)

        dense = matrix.to_dense_matrix()
        assert_allclose(dense, dense.T, rtol=1e-10, atol=1e-12)

        n_decimals = 10
        spectrum = np.linalg.eigvalsh(dense)
        rounded = np.unique(np.round(spectrum, n_decimals))[:n_states]
        assert_allclose(state.excitation_energy, rounded, atol=10 * conv_tol)

        # TODO Test eigenvectors as well.

    def template_h2o(self, method):
        kwargs = {}
        if "cvs" in method:
            kwargs["n_states"] = 7
            kwargs["max_subspace"] = 30
        if method in ["cvs-adc2"]:
            kwargs["n_states"] = 5
        if method in ["cvs-adc1"]:
            kwargs["n_states"] = 2
        self.base_test("h2o_sto3g", method, **kwargs)

    # def template_cn(self, method):
    #    TODO Testing this for CN is a bit tricky, because
    #         the dense basis we employ is not yet spin-adapted
    #         and allows e.g. simultaneous α->α and α->β components to mix
    #         A closer investigation is needed here
    #    self.base_test("cn_sto3g", method, **kwargs)
