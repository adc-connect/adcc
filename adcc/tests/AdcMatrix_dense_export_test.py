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
import pytest
import numpy as np
from numpy.testing import assert_allclose

import adcc

from .testdata_cache import testdata_cache
from . import testcases


h2o = testcases.get_by_filename("h2o_sto3g").pop()
cases = ["gen", "cvs"]
assert all(c in h2o.cases for c in cases)
methods = ["adc1", "adc2", "adc2x", "adc3"]


@pytest.mark.parametrize("case", cases)
@pytest.mark.parametrize("method", methods)
class TestAdcMatrixDenseExport:
    # TODO Testing this for CN is a bit tricky, because
    #     the dense basis we employ is not yet spin-adapted
    #     and allows e.g. simultaneous α->α and α->β components to mix
    #     A closer investigation is needed here
    def test_h2o(self, case: str, method: str):
        if "cvs" in case and "cvs" not in method:
            method = f"cvs-{method}"
        method: adcc.AdcMethod = adcc.AdcMethod(method)
        n_states = 7
        if method.level == 1:  # only few states available
            n_states = 5
            if method.is_core_valence_separated:
                n_states = 1 if "fv" in case else 2
            elif "fc" in case and "fv" in case:
                n_states = 4
        elif method.level == 2 and method.is_core_valence_separated:
            # there seems to be a difference for higher cvs-adc2 states...
            n_states = 3

        conv_tol = 1e-8

        refstate = testdata_cache.refstate(h2o, case=case)
        matrix = adcc.AdcMatrix(method, refstate)
        state = adcc.run_adc(
            matrix, method=method, conv_tol=conv_tol, n_states=n_states,
            max_subspace=7 * n_states
        )

        dense = matrix.to_ndarray()
        assert_allclose(dense, dense.T, rtol=1e-10, atol=1e-12)

        spectrum = np.linalg.eigvalsh(dense)
        rounded = np.unique(np.round(spectrum, 10))[:n_states]
        assert_allclose(state.excitation_energy, rounded, atol=10 * conv_tol)
