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
import adcc
import pytest
import libadcc
import unittest
import itertools
import numpy as np

from numpy.testing import assert_allclose

from adcc.AdcMatrix import AdcMatrixShifted
from adcc.testdata.cache import cache

from .misc import expand_test_templates

# Test diagonal, block-wise apply and matvec

# Reference data for cn_sto3g and h2o_sto3g contains
# a random vector and the result of block-wise application and matvec
# as well as reference results for the diagonal() call

testcases = ["h2o_sto3g", "cn_sto3g"]
basemethods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]
methods = [m for bm in basemethods for m in [bm, "cvs-" + bm]]

# TODO Also test these cases:
# methods += ["fc-adc2", "fv-adc2x", "fv-cvs-adc2x", "fc-fv-adc2"]


@expand_test_templates(list(itertools.product(testcases, methods)))
class TestAdcMatrix(unittest.TestCase):
    def construct_matrix(self, case, method):
        refdata = cache.reference_data[case]
        matdata = refdata[method]["matrix"]
        if "cvs" in method:
            refstate = cache.refstate_cvs[case]
        else:
            refstate = cache.refstate[case]

        matrix = adcc.AdcMatrix(method, refstate)
        return matrix, matdata

    def construct_input(self, case, method):
        refdata = cache.reference_data[case]
        firstkind = refdata["available_kinds"][0]
        state = cache.adc_states[case][method][firstkind]
        matdata = refdata[method]["matrix"]

        out = state.excitation_vector[0].copy()
        out["s"].set_from_ndarray(matdata["random_singles"])
        if "random_doubles" in matdata:
            out["d"].set_from_ndarray(matdata["random_doubles"])
        return out

    def template_diagonal(self, case, method):
        matrix, matdata = self.construct_matrix(case, method)

        diag_s = matrix.diagonal("s")
        assert_allclose(matdata["diagonal_singles"], diag_s.to_ndarray(),
                        rtol=1e-10, atol=1e-12)

        if "d" in matrix.blocks:
            diag_d = matrix.diagonal("d")
            assert_allclose(matdata["diagonal_doubles"], diag_d.to_ndarray(),
                            rtol=1e-10, atol=1e-12)

    def template_matvec(self, case, method):
        matrix, matdata = self.construct_matrix(case, method)
        invec = self.construct_input(case, method)

        outvec = matrix @ invec

        assert_allclose(matdata["matvec_singles"], outvec["s"].to_ndarray(),
                        rtol=1e-10, atol=1e-12)
        if "matvec_doubles" in matdata:
            assert_allclose(matdata["matvec_doubles"], outvec["d"].to_ndarray(),
                            rtol=1e-10, atol=1e-12)

    def template_compute_block(self, case, method):
        matrix, matdata = self.construct_matrix(case, method)
        invec = self.construct_input(case, method)
        outvec = invec.copy()

        for b1 in ["s", "d"]:
            for b2 in ["s", "d"]:
                if f"result_{b1}{b2}" not in matdata:
                    continue

                if b1 + b2 == "dd" and method in ["adc2x", "adc3"]:
                    pytest.xfail("ADC(2)-x and ADC(3) doubles-doubles apply"
                                 "is buggy in adccore.")

                matrix.compute_apply(b1 + b2, invec[b2], outvec[b1])
                assert_allclose(matdata[f"result_{b1}{b2}"],
                                outvec[b1].to_ndarray(),
                                rtol=1e-10, atol=1e-12)


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


@expand_test_templates(testcases)
class TestAdcMatrixShifted(unittest.TestCase):
    def construct_matrices(self, case, shift):
        reference_state = cache.refstate[case]
        ground_state = adcc.LazyMp(reference_state)
        matrix = adcc.AdcMatrix("adc3", ground_state)
        shifted = AdcMatrixShifted(matrix, shift)
        return matrix, shifted

    def template_diagonal(self, case):
        shift = -0.3
        matrix, shifted = self.construct_matrices(case, shift)

        for block in ("s", "d"):
            odiag = matrix.diagonal(block).to_ndarray()
            sdiag = shifted.diagonal(block).to_ndarray()
            assert np.max(np.abs(sdiag - shift - odiag)) < 1e-12

    def template_matmul(self, case):
        shift = -0.3
        matrix, shifted = self.construct_matrices(case, shift)

        vec = adcc.guess_zero(matrix)
        vec["s"].set_random()
        vec["d"].set_random()

        ores = matrix @ vec
        sres = shifted @ vec

        assert ores["s"].describe_symmetry() == sres["s"].describe_symmetry()
        assert ores["d"].describe_symmetry() == sres["d"].describe_symmetry()

        diff_s = sres["s"] - ores["s"] - shift * vec["s"]
        diff_d = sres["d"] - ores["d"] - shift * vec["d"]
        assert np.max(np.abs(diff_s.to_ndarray())) < 1e-12
        assert np.max(np.abs(diff_d.to_ndarray())) < 1e-12

    # TODO Test to_dense_matrix, compute_apply
