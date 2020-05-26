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

import libadcc

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


class TestAdcMatrixInterface(unittest.TestCase):
    def test_properties_adc2(self):
        case = "h2o_sto3g"
        method = "adc2"

        reference_state = cache.refstate[case]
        ground_state = adcc.LazyMp(reference_state)
        matrix = adcc.AdcMatrix(method, ground_state)

        assert matrix.ndim == 2
        assert not matrix.is_core_valence_separated
        assert matrix.shape == (1640, 1640)
        assert len(matrix) == 1640

        assert matrix.blocks == ["s", "d"]
        assert matrix.has_block("s")
        assert matrix.has_block("d")
        assert not matrix.has_block("t")
        assert matrix.block_spaces("s") == ["o1", "v1"]
        assert matrix.block_spaces("d") == ["o1", "o1", "v1", "v1"]

        assert matrix.reference_state == reference_state
        assert matrix.mospaces == reference_state.mospaces
        assert isinstance(matrix.timer, adcc.timings.Timer)
        assert isinstance(matrix.to_cpp(), libadcc.AdcMatrix)

    def test_properties_cvs_adc1(self):
        case = "h2o_sto3g"
        method = "cvs-adc1"

        reference_state = cache.refstate_cvs[case]
        ground_state = adcc.LazyMp(reference_state)
        matrix = adcc.AdcMatrix(method, ground_state)

        assert matrix.ndim == 2
        assert matrix.is_core_valence_separated
        assert matrix.shape == (8, 8)
        assert len(matrix) == 8

        assert matrix.blocks == ["s"]
        assert matrix.has_block("s")
        assert not matrix.has_block("d")
        assert not matrix.has_block("t")
        assert matrix.block_spaces("s") == ["o2", "v1"]

        assert matrix.reference_state == reference_state
        assert matrix.mospaces == reference_state.mospaces
        assert isinstance(matrix.timer, adcc.timings.Timer)
        assert isinstance(matrix.to_cpp(), libadcc.AdcMatrix)

    def test_intermediates_adc2(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("adc2", ground_state)
        assert isinstance(matrix.intermediates, libadcc.AdcIntermediates)
        intermediates = libadcc.AdcIntermediates(ground_state)
        matrix.intermediates = intermediates
        assert matrix.intermediates == intermediates

    def test_diagonal_adc2(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("adc2", ground_state)
        cppmatrix = libadcc.AdcMatrix("adc2", ground_state)
        diff = cppmatrix.diagonal("s") - matrix.diagonal("s")
        assert diff.dot(diff) < 1e-12

    def test_matvec_adc2(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("adc2", ground_state)
        cppmatrix = libadcc.AdcMatrix("adc2", ground_state)

        vectors = [adcc.guess_zero(matrix) for i in range(3)]
        for vec in vectors:
            vec["s"].set_random()
            vec["d"].set_random()
        v, w, x = vectors

        # Compute references:
        refv = adcc.empty_like(v)
        refw = adcc.empty_like(w)
        refx = adcc.empty_like(x)
        cppmatrix.compute_matvec(v.to_cpp(), refv.to_cpp())
        cppmatrix.compute_matvec(w.to_cpp(), refw.to_cpp())
        cppmatrix.compute_matvec(x.to_cpp(), refx.to_cpp())

        # @ operator (1 vector)
        resv = matrix @ v
        diffv = refv - resv
        assert diffv["s"].dot(diffv["s"]) < 1e-12
        assert diffv["d"].dot(diffv["d"]) < 1e-12

        # @ operator (multiple vectors)
        resv, resw, resx = matrix @ [v, w, x]
        diffs = [refv - resv, refw - resw, refx - resx]
        for i in range(3):
            assert diffs[i]["s"].dot(diffs[i]["s"]) < 1e-12
            assert diffs[i]["d"].dot(diffs[i]["d"]) < 1e-12

        # compute matvec
        matrix.compute_matvec(v, resv)
        diffv = refv - resv
        assert diffv["s"].dot(diffv["s"]) < 1e-12
        assert diffv["d"].dot(diffv["d"]) < 1e-12

        matrix.compute_apply("ss", v["s"], resv["s"])
        cppmatrix.compute_apply("ss", v["s"], refv["s"])
        diffv = resv["s"] - refv["s"]
        assert diffv.dot(diffv) < 1e-12
