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
import unittest
import pytest
import itertools
import numpy as np

from numpy.testing import assert_allclose

from adcc.AdcMatrix import AdcMatrixShifted, AdcExtraTerm
from adcc.adc_pp.matrix import AdcBlock
from adcc.testdata.cache import cache

from .misc import expand_test_templates
from .Intermediates import Intermediates

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
        out.ph.set_from_ndarray(matdata["random_singles"])
        if "random_doubles" in matdata:
            out.pphh.set_from_ndarray(matdata["random_doubles"])
        return out

    def template_diagonal(self, case, method):
        matrix, matdata = self.construct_matrix(case, method)

        diag_s = matrix.diagonal().ph
        assert_allclose(matdata["diagonal_singles"], diag_s.to_ndarray(),
                        rtol=1e-10, atol=1e-12)

        if "pphh" in matrix.axis_blocks:
            diag_d = matrix.diagonal().pphh
            assert_allclose(matdata["diagonal_doubles"], diag_d.to_ndarray(),
                            rtol=1e-10, atol=1e-12)

    def template_matvec(self, case, method):
        matrix, matdata = self.construct_matrix(case, method)
        invec = self.construct_input(case, method)

        outvec = matrix @ invec

        assert_allclose(matdata["matvec_singles"], outvec.ph.to_ndarray(),
                        rtol=1e-10, atol=1e-12)
        if "matvec_doubles" in matdata:
            assert_allclose(matdata["matvec_doubles"], outvec.pphh.to_ndarray(),
                            rtol=1e-10, atol=1e-12)

    def template_compute_block(self, case, method):
        matrix, matdata = self.construct_matrix(case, method)
        invec = self.construct_input(case, method)
        translate = {"s": "ph", "d": "pphh"}

        for b1 in ["s", "d"]:
            for b2 in ["s", "d"]:
                if f"result_{b1}{b2}" not in matdata:
                    continue
                pb1, pb2 = translate[b1], translate[b2]
                ret = matrix.block_apply(pb1 + "_" + pb2, invec[pb2])
                assert_allclose(matdata[f"result_{b1}{b2}"],
                                ret.to_ndarray(),
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

        assert matrix.axis_blocks == ["ph", "pphh"]
        assert sorted(matrix.axis_spaces.keys()) == matrix.axis_blocks
        assert sorted(matrix.axis_lengths.keys()) == matrix.axis_blocks
        assert matrix.axis_spaces["ph"] == ["o1", "v1"]
        assert matrix.axis_spaces["pphh"] == ["o1", "o1", "v1", "v1"]
        assert matrix.axis_lengths["ph"] == 40
        assert matrix.axis_lengths["pphh"] == 1600

        assert matrix.reference_state == reference_state
        assert matrix.mospaces == reference_state.mospaces
        assert isinstance(matrix.timer, adcc.timings.Timer)

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

        assert matrix.axis_blocks == ["ph"]
        assert sorted(matrix.axis_spaces.keys()) == matrix.axis_blocks
        assert sorted(matrix.axis_lengths.keys()) == matrix.axis_blocks
        assert matrix.axis_spaces["ph"] == ["o2", "v1"]
        assert matrix.axis_lengths["ph"] == 8

        assert matrix.reference_state == reference_state
        assert matrix.mospaces == reference_state.mospaces
        assert isinstance(matrix.timer, adcc.timings.Timer)

    def test_intermediates_adc2(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("adc2", ground_state)
        assert isinstance(matrix.intermediates, Intermediates)
        intermediates = Intermediates(ground_state)
        matrix.intermediates = intermediates
        assert matrix.intermediates == intermediates

    def test_matvec_adc2(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("adc2", ground_state)

        vectors = [adcc.guess_zero(matrix) for i in range(3)]
        for vec in vectors:
            vec.set_random()
        v, w, x = vectors

        # Compute references:
        refv = matrix.matvec(v)
        refw = matrix.matvec(w)
        refx = matrix.matvec(x)

        # @ operator (1 vector)
        resv = matrix @ v
        diffv = refv - resv
        assert diffv.ph.dot(diffv.ph) < 1e-12
        assert diffv.pphh.dot(diffv.pphh) < 1e-12

        # @ operator (multiple vectors)
        resv, resw, resx = matrix @ [v, w, x]
        diffs = [refv - resv, refw - resw, refx - resx]
        for i in range(3):
            assert diffs[i].ph.dot(diffs[i].ph) < 1e-12
            assert diffs[i].pphh.dot(diffs[i].pphh) < 1e-12

        # compute matvec
        resv = matrix.matvec(v)
        diffv = refv - resv
        assert diffv.ph.dot(diffv.ph) < 1e-12
        assert diffv.pphh.dot(diffv.pphh) < 1e-12

        resv = matrix.rmatvec(v)
        diffv = refv - resv
        assert diffv.ph.dot(diffv.ph) < 1e-12
        assert diffv.pphh.dot(diffv.pphh) < 1e-12

        # Test apply
        resv.ph = matrix.block_apply("ph_ph", v.ph)
        resv.ph += matrix.block_apply("ph_pphh", v.pphh)
        refv = matrix.matvec(v)
        diffv = resv.ph - refv.ph
        assert diffv.dot(diffv) < 1e-12

    def test_extra_term(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix_adc1 = adcc.AdcMatrix("adc1", ground_state)
        with pytest.raises(TypeError):
            matrix_adc1 += 42
        matrix = adcc.AdcMatrix("adc2", ground_state)
        shift = -0.3
        shifted = AdcMatrixShifted(matrix, shift)
        # TODO: need to do this to differentiate between
        # diagonals for ph and pphh
        # if we just pass numbers, i.e., shift
        # we get 2*shift on the diagonal :(
        ones = matrix.diagonal().ones_like()

        with pytest.raises(TypeError):
            AdcExtraTerm(matrix, "fail")
        with pytest.raises(TypeError):
            AdcExtraTerm(matrix, {"fail": "not_callable"})

        def __shift_ph(hf, mp, intermediates):
            def apply(invec):
                return adcc.AmplitudeVector(ph=shift * invec.ph)
            diag = adcc.AmplitudeVector(ph=shift * ones.ph)
            return AdcBlock(apply, diag)

        def __shift_pphh(hf, mp, intermediates):
            def apply(invec):
                return adcc.AmplitudeVector(pphh=shift * invec.pphh)
            diag = adcc.AmplitudeVector(pphh=shift * ones.pphh)
            return AdcBlock(apply, diag)
        extra = AdcExtraTerm(
            matrix, {'ph_ph': __shift_ph, 'pphh_pphh': __shift_pphh}
        )
        # cannot add to 'pphh_pphh' in ADC(1) matrix
        with pytest.raises(ValueError):
            matrix_adc1 += extra
        shifted_2 = matrix + extra
        shifted_3 = extra + matrix
        for manual in [shifted_2, shifted_3]:
            assert_allclose(
                shifted.diagonal().ph.to_ndarray(),
                manual.diagonal().ph.to_ndarray(),
                atol=1e-12
            )
            assert_allclose(
                shifted.diagonal().pphh.to_ndarray(),
                manual.diagonal().pphh.to_ndarray(),
                atol=1e-12
            )
            vec = adcc.guess_zero(matrix)
            vec.set_random()
            ref = shifted @ vec
            ret = manual @ vec
            diff_s = ref.ph - ret.ph
            diff_d = ref.pphh - ret.pphh
            assert np.max(np.abs(diff_s.to_ndarray())) < 1e-12
            assert np.max(np.abs(diff_d.to_ndarray())) < 1e-12


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

        for block in ("ph", "pphh"):
            odiag = matrix.diagonal()[block].to_ndarray()
            sdiag = shifted.diagonal()[block].to_ndarray()
            assert np.max(np.abs(sdiag - shift - odiag)) < 1e-12

    def template_matmul(self, case):
        shift = -0.3
        matrix, shifted = self.construct_matrices(case, shift)

        vec = adcc.guess_zero(matrix)
        vec.set_random()

        ores = matrix @ vec
        sres = shifted @ vec

        assert ores.ph.describe_symmetry() == sres.ph.describe_symmetry()
        assert ores.pphh.describe_symmetry() == sres.pphh.describe_symmetry()

        diff_s = sres.ph - ores.ph - shift * vec.ph
        diff_d = sres.pphh - ores.pphh - shift * vec.pphh
        assert np.max(np.abs(diff_s.to_ndarray())) < 1e-12
        assert np.max(np.abs(diff_d.to_ndarray())) < 1e-12

    # TODO Test to_dense_matrix, compute_apply
