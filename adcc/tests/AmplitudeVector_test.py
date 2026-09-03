## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2021 by the adcc authors
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

import pytest
from numpy.testing import assert_allclose

from adcc import AdcMatrix, AmplitudeVector, LazyMp, guess_zero, zeros_like

from .testdata_cache import testdata_cache


class TestAmplitudeVector(unittest.TestCase):
    def test_functionality(self):
        ground_state = LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        matrix = AdcMatrix("adc2", ground_state)
        vectors = [guess_zero(matrix) for _ in range(2)]
        for vec in vectors:
            vec.set_random()
        v, w = vectors
        with pytest.raises(AttributeError):
            _ = v.pph
        with pytest.raises(AttributeError):
            v.pph = w.ph
        # setattr with expression
        z = zeros_like(v)
        z.ph = v.ph + w.ph
        z -= w
        assert_allclose(v.ph.to_ndarray(), z.ph.to_ndarray())

    def test_add(self):
        adc1 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc1", case="gen", kind="singlet"
        )
        adc2 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", case="gen", kind="singlet"
        )
        #####################
        # __add__ + __iadd__
        #####################
        # - 2 Amplitude vectors
        # both share the same blocks
        v1, v2 = adc2.excitation_vector[:2]
        res = v1 + v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph.to_ndarray() + v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        ref = v1.pphh.to_ndarray() + v2.pphh.to_ndarray()
        assert_allclose(res.pphh.to_ndarray(), ref, atol=1e-14)
        # check in-place addition
        res_inplace = v1.copy()
        res_inplace += v2
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # left misses 1 block
        v1, v2 = adc1.excitation_vector[0], adc2.excitation_vector[0]
        res = v1 + v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph.to_ndarray() + v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), v2.pphh.to_ndarray(), atol=1e-14)
        # check in-place addition
        res_inplace = v1.copy()
        res_inplace += v2
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # right misses 1 block
        v1, v2 = adc2.excitation_vector[0], adc1.excitation_vector[1]
        res = v1 + v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph.to_ndarray() + v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), v1.pphh.to_ndarray(), atol=1e-14)
        # check in-place addition
        res_inplace = v1.copy()
        res_inplace += v2
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # - Amplitude + Tensor
        v1, v2 = adc1.excitation_vector[0], adc1.excitation_vector[1].ph
        res = v1 + v2
        assert res.blocks == ["ph"]
        ref = v1.ph.to_ndarray() + v2.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        # check in-place addition
        res_inplace = v1.copy()
        res_inplace += v2
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        # - Amplitude + float
        v1, v2 = adc2.excitation_vector[0], 0.1
        res = v1 + v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph + v2
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        ref = v1.pphh + v2
        assert_allclose(res.pphh.to_ndarray(), ref.to_ndarray(), atol=1e-14)

    def test_mul(self):
        adc1 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc1", case="gen", kind="singlet"
        )
        adc2 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", case="gen", kind="singlet"
        )
        ####################
        # __mul__ + __imul__
        ####################
        # - 2 amplitudes
        # Both share the same blocks
        v1, v2 = adc2.excitation_vector[:2]
        res = v1 * v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph.to_ndarray() * v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        ref = v1.pphh.to_ndarray() * v2.pphh.to_ndarray()
        assert_allclose(res.pphh.to_ndarray(), ref, atol=1e-14)
        # check in-place mul
        res_inplace = v1.copy()
        res_inplace *= v2
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # Left amplitude misses a block
        v1, v2 = adc1.excitation_vector[0], adc2.excitation_vector[0]
        res = v1 * v2
        assert res.blocks == ["ph"]
        ref = v1.ph.to_ndarray() * v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        # check in-place mul
        res_inplace = v1.copy()
        res_inplace *= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        # Right amplitude misses a block
        v1, v2 = adc2.excitation_vector[0], adc1.excitation_vector[0]
        res = v1 * v2
        assert res.blocks == ["ph"]
        ref = v1.ph.to_ndarray() * v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        # check in-place mul
        res_inplace = v1.copy()
        res_inplace *= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        # Amplitude * tensor
        v1, v2 = adc1.excitation_vector[0], adc1.excitation_vector[1].ph
        res = v1 * v2
        assert res.blocks == ["ph"]
        ref = v1.ph.to_ndarray() * v2.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        # check in-place mul
        res_inplace = v1.copy()
        res_inplace *= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        # Amplitude * float
        v1, v2 = adc1.excitation_vector[0], 0.1
        res = v1 * v2
        assert res.blocks == ["ph"]
        ref = v1.ph * v2
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # check in-place mul
        res_inplace = v1.copy()
        res_inplace *= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)

    def test_div(self):
        adc1 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc1", case="gen", kind="singlet"
        )
        adc2 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", case="gen", kind="singlet"
        )
        ############################
        # __truediv__ + __itruediv__
        ############################
        # - 2 amplitudes
        # Both share the same blocks
        v1, v2 = adc2.excitation_vector[:2]
        res = v1 / v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph / v2.ph
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        ref = v1.pphh / v2.pphh
        assert_allclose(res.pphh.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # check in-place addition
        res_inplace = v1.copy()
        res_inplace /= v2
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # Left amplitude misses a block
        v1, v2 = adc1.excitation_vector[0], adc2.excitation_vector[0]
        res = v1 / v2
        assert res.blocks == ["ph"]
        ref = v1.ph / v2.ph
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # check in-place div
        res_inplace = v1.copy()
        res_inplace /= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        # Right amplitude misses a block
        v1, v2 = adc2.excitation_vector[0], adc1.excitation_vector[0]
        with pytest.raises(ZeroDivisionError):
            res = v1 / v2
        # check in-place div
        res_inplace = v1.copy()
        with pytest.raises(ZeroDivisionError):
            res_inplace /= v2
        # Amplitude / tensor
        v1, v2 = adc1.excitation_vector[0], adc1.excitation_vector[1].ph
        res = v1 / v2
        assert res.blocks == ["ph"]
        ref = v1.ph / v2
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # check in-place div
        res_inplace = v1.copy()
        res_inplace /= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        # Amplitude / float
        v1, v2 = adc1.excitation_vector[0], 0.1
        res = v1 / v2
        assert res.blocks == ["ph"]
        ref = v1.ph / v2
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # check in-place div
        res_inplace = v1.copy()
        res_inplace /= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)

    def test_sub(self):
        adc1 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc1", case="gen", kind="singlet"
        )
        adc2 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", case="gen", kind="singlet"
        )
        ####################
        # __sub__ + __isub__
        ####################
        # - 2 amplitudes
        # Both share the same blocks
        v1, v2 = adc2.excitation_vector[:2]
        res = v1 - v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph.to_ndarray() - v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        ref = v1.pphh.to_ndarray() - v2.pphh.to_ndarray()
        assert_allclose(res.pphh.to_ndarray(), ref, atol=1e-14)
        # check in-place subtraction
        res_inplace = v1.copy()
        res_inplace -= v2
        assert res_inplace.blocks == ["ph", "pphh"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # Left amplitude misses a block: 0 - x = -x
        v1, v2 = adc1.excitation_vector[0], adc2.excitation_vector[0]
        res = v1 - v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph.to_ndarray() - v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), -v2.pphh.to_ndarray(), atol=1e-14)
        # check in-place subtraction
        res_inplace = v1.copy()
        res_inplace -= v2
        assert res_inplace.blocks == ["ph", "pphh"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # Right amplitude misses a block: x - 0 = x
        v1, v2 = adc2.excitation_vector[0], adc1.excitation_vector[1]
        res = v1 - v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph.to_ndarray() - v2.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), v1.pphh.to_ndarray(), atol=1e-14)
        # check in-place subtraction
        res_inplace = v1.copy()
        res_inplace -= v2
        assert res_inplace.blocks == ["ph", "pphh"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), res_inplace.pphh.to_ndarray(), atol=1e-14)
        # - Amplitude - Tensor
        v1, v2 = adc1.excitation_vector[0], adc1.excitation_vector[1].ph
        res = v1 - v2
        assert res.blocks == ["ph"]
        ref = v1.ph.to_ndarray() - v2.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref, atol=1e-14)
        # check in-place subtraction
        res_inplace = v1.copy()
        res_inplace -= v2
        assert res_inplace.blocks == ["ph"]
        assert_allclose(res.ph.to_ndarray(), res_inplace.ph.to_ndarray(), atol=1e-14)
        # - Amplitude - float
        v1, v2 = adc2.excitation_vector[0], 0.1
        res = v1 - v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1.ph - v2
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        ref = v1.pphh - v2
        assert_allclose(res.pphh.to_ndarray(), ref.to_ndarray(), atol=1e-14)

    def test_dot(self):
        adc1 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc1", case="gen", kind="singlet"
        )
        adc2 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", case="gen", kind="singlet"
        )
        ####################
        # dot + __matmul__
        ####################
        # - 2 amplitudes
        # Both share the same blocks
        v1, v2 = adc2.excitation_vector[:2]
        ref = v1.ph.dot(v2.ph) + v1.pphh.dot(v2.pphh)
        assert v1.dot(v2) == pytest.approx(ref, abs=1e-14)
        assert v1 @ v2 == pytest.approx(ref, abs=1e-14)
        # Left amplitude misses a block
        v1, v2 = adc1.excitation_vector[0], adc2.excitation_vector[0]
        ref = v1.ph.dot(v2.ph)
        assert v1.dot(v2) == pytest.approx(ref, abs=1e-14)
        assert v1 @ v2 == pytest.approx(ref, abs=1e-14)
        # Right amplitude misses a block
        v1, v2 = adc2.excitation_vector[0], adc1.excitation_vector[0]
        ref = v1.ph.dot(v2.ph)
        assert v1.dot(v2) == pytest.approx(ref, abs=1e-14)
        assert v1 @ v2 == pytest.approx(ref, abs=1e-14)
        # The amplitudes have no block in common
        v = adc2.excitation_vector[0]
        v1, v2 = AmplitudeVector(ph=v.ph), AmplitudeVector(pphh=v.pphh)
        assert v1.dot(v2) == pytest.approx(0.0, abs=1e-14)
        assert v1 @ v2 == pytest.approx(0.0, abs=1e-14)
        # - Amplitude and a list of amplitudes
        # the result has to agree with the individual dot products
        v1 = adc2.excitation_vector[0]
        v2 = [adc2.excitation_vector[1], adc1.excitation_vector[0], AmplitudeVector(pphh=v1.pphh)]
        ref = [v1.dot(other) for other in v2]
        assert_allclose(v1.dot(v2), ref, atol=1e-14)
        assert_allclose(v1 @ v2, ref, atol=1e-14)
        # an empty list gives an empty array
        assert len(v1.dot([])) == 0

    def test_reflected_operations(self):
        adc2 = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", case="gen", kind="singlet"
        )
        ###################################
        # __radd__ + __rmul__ + __rsub__
        ###################################
        # - float + Amplitude
        v1, v2 = 0.1, adc2.excitation_vector[0]
        res = v1 + v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1 + v2.ph
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        ref = v1 + v2.pphh
        assert_allclose(res.pphh.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # - float * Amplitude
        res = v1 * v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1 * v2.ph
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        ref = v1 * v2.pphh
        assert_allclose(res.pphh.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # - float - Amplitude
        res = v1 - v2
        assert res.blocks == ["ph", "pphh"]
        ref = v1 - v2.ph
        assert_allclose(res.ph.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        ref = v1 - v2.pphh
        assert_allclose(res.pphh.to_ndarray(), ref.to_ndarray(), atol=1e-14)
        # - Tensor + Amplitude is not supported: the backend operators raise
        #   a TypeError instead of returning NotImplemented, so the reflected
        #   operations of the AmplitudeVector are never reached.
        v1, v2 = adc2.excitation_vector[0].ph, adc2.excitation_vector[0]
        with pytest.raises(TypeError):
            v1 + v2
        with pytest.raises(TypeError):
            v1 * v2
        with pytest.raises(TypeError):
            v1 - v2
