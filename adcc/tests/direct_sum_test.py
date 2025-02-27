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
from numpy.testing import assert_allclose

from adcc import direct_sum, empty_like, nosym_like
from .testdata_cache import testdata_cache


class TestDirectSum(unittest.TestCase):
    def test_1_1(self):
        refstate = testdata_cache.refstate("cn_sto3g", case="gen")
        a = nosym_like(refstate.orbital_energies("o1")).set_random()
        b = nosym_like(refstate.orbital_energies("o1")).set_random()

        res = direct_sum("i+j", a, b)
        ref = a.to_ndarray()[:, None] + b.to_ndarray()[None, :]
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("i+j->ji", a, b)
        ref = a.to_ndarray()[None, :] + b.to_ndarray()[:, None]
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("i-j", a, b)
        ref = a.to_ndarray()[:, None] - b.to_ndarray()[None, :]
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("-i+j->ji", a, b)
        ref = -a.to_ndarray()[None, :] + b.to_ndarray()[:, None]
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("-i-j", a, b)
        ref = -a.to_ndarray()[:, None] - b.to_ndarray()[None, :]
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

    def test_2_2(self):
        refstate = testdata_cache.refstate("cn_sto3g", case="gen")
        a = nosym_like(refstate.fock("o1v1")).set_random()
        b = nosym_like(refstate.fock("v1o1")).set_random()
        no, nv = a.shape

        res = direct_sum("ia+bj", a, b)
        assert res.shape == (no, nv, nv, no)
        ref = a.to_ndarray()[:, :, None, None] + b.to_ndarray()[None, None, :, :]
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("-ia+bj->jabi", a, b)
        assert res.shape == (no, nv, nv, no)
        ref = -a.to_ndarray()[:, :, None, None] + b.to_ndarray()[None, None, :, :]
        ref = ref.transpose((3, 1, 2, 0))
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("-ia-bj->abij", a, b)
        assert res.shape == (nv, nv, no, no)
        ref = -a.to_ndarray()[:, :, None, None] - b.to_ndarray()[None, None, :, :]
        ref = ref.transpose((1, 2, 0, 3))
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

    def test_2_2_sym(self):
        refstate = testdata_cache.refstate("cn_sto3g", case="gen")
        a = empty_like(refstate.fock("o1o1")).set_random()
        b = empty_like(refstate.fock("v1v1")).set_random()
        no, _ = a.shape
        nv, _ = b.shape

        res = direct_sum("ij+ab", a, b)
        assert res.shape == (no, no, nv, nv)
        ref = a.to_ndarray()[:, :, None, None] + b.to_ndarray()[None, None, :, :]
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("-ij+ab->jabi", a, b)
        assert res.shape == (no, nv, nv, no)
        ref = -a.to_ndarray()[:, :, None, None] + b.to_ndarray()[None, None, :, :]
        ref = ref.transpose((1, 2, 3, 0))
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("-ij-ab->abij", a, b)
        assert res.shape == (nv, nv, no, no)
        ref = -a.to_ndarray()[:, :, None, None] - b.to_ndarray()[None, None, :, :]
        ref = ref.transpose((2, 3, 0, 1))
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

    def test_1_2_1(self):
        refstate = testdata_cache.refstate("cn_sto3g", case="gen")
        oeo = nosym_like(refstate.orbital_energies("o1")).set_random()
        fvv = nosym_like(refstate.fock("v1v1")).set_random()
        oev = nosym_like(refstate.orbital_energies("v1")).set_random()

        res = direct_sum("i+ab-c", oeo, fvv, oev)
        ref = (oeo.to_ndarray()[:, None, None, None]
               + fvv.to_ndarray()[None, :, :, None]
               - oev.to_ndarray()[None, None, None, :])
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

        res = direct_sum("ab-i-c->iacb", fvv, oeo, oev)
        ref = (fvv.to_ndarray()[:, :, None, None]
               - oeo.to_ndarray()[None, None, :, None]
               - oev.to_ndarray()[None, None, None, :])
        ref = ref.transpose((2, 0, 3, 1))
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)
