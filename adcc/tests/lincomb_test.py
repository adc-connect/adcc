## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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

from numpy.testing import assert_allclose

from adcc import AmplitudeVector, lincomb, zeros_like

from .testdata_cache import testdata_cache


class TestLincomb(unittest.TestCase):
    def setUp(self):
        refstate = testdata_cache.refstate("h2o_sto3g", case="gen")
        self.ph = [zeros_like(refstate.fov).set_random() for _ in range(2)]
        self.pphh = [zeros_like(refstate.oovv).set_random() for _ in range(2)]

    def test_tensor(self):
        a, b = self.ph
        ref = 2.0 * a.to_ndarray() - 0.5 * b.to_ndarray()
        for evaluate in (True, False):
            res = lincomb([2.0, -0.5], [a, b], evaluate=evaluate)
            assert_allclose(res.to_ndarray(), ref, atol=1e-14)

    def test_amplitude_vector_identical_blocks(self):
        a = AmplitudeVector(ph=self.ph[0], pphh=self.pphh[0])
        b = AmplitudeVector(ph=self.ph[1], pphh=self.pphh[1])
        res = lincomb([2.0, -0.5], [a, b], evaluate=True)
        assert sorted(res.keys()) == ["ph", "pphh"]
        for block in ("ph", "pphh"):
            ref = 2.0 * a[block].to_ndarray() - 0.5 * b[block].to_ndarray()
            assert_allclose(res[block].to_ndarray(), ref, atol=1e-14)

    def test_amplitude_vector_varying_blocks(self):
        # missing blocks are treated as zero
        a = AmplitudeVector(ph=self.ph[0])
        b = AmplitudeVector(ph=self.ph[1], pphh=self.pphh[1])
        res = lincomb([2.0, -0.5], [a, b], evaluate=True)
        assert sorted(res.keys()) == ["ph", "pphh"]
        ref_ph = 2.0 * a.ph.to_ndarray() - 0.5 * b.ph.to_ndarray()
        assert_allclose(res.ph.to_ndarray(), ref_ph, atol=1e-14)
        assert_allclose(res.pphh.to_ndarray(), -0.5 * b.pphh.to_ndarray(), atol=1e-14)
