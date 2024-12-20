#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp, assert_equal

from adcc import OneParticleOperator, zeros_like
from adcc.OneParticleOperator import product_trace

from .testdata_cache import testdata_cache


class TestOneParticleOperator(unittest.TestCase):
    def test_to_ndarray(self):
        mp2diff = adcc.LazyMp(
            testdata_cache.refstate("h2o_sto3g", "gen")
        ).mp2_diffdm

        dm_oo = mp2diff.oo.to_ndarray()
        dm_ov = mp2diff.ov.to_ndarray()
        dm_vv = mp2diff.vv.to_ndarray()

        dm_o = np.hstack((dm_oo, dm_ov))
        dm_v = np.hstack((dm_ov.transpose(), dm_vv))
        dm_full = np.vstack((dm_o, dm_v))

        np.testing.assert_almost_equal(dm_full, mp2diff.to_ndarray(),
                                       decimal=12)

    def test_to_ndarray_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        dm = OneParticleOperator(ref.mospaces, is_symmetric=False)
        dm.oo.set_random()
        dm.ov.set_random()
        dm.vo.set_random()
        dm.vv.set_random()

        dm_oo = dm.oo.to_ndarray()
        dm_ov = dm.ov.to_ndarray()
        dm_vo = dm.vo.to_ndarray()
        dm_vv = dm.vv.to_ndarray()

        dm_o = np.hstack((dm_oo, dm_ov))
        dm_v = np.hstack((dm_vo, dm_vv))
        dm_full = np.vstack((dm_o, dm_v))

        assert_array_almost_equal_nulp(dm_full, dm.to_ndarray())

    def test_product_trace_symmetric(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        dipx_mo = ref.operators.electric_dipole[0]
        mp2diff_mo = adcc.LazyMp(ref).mp2_diffdm
        mp2diff_ao = mp2diff_mo.to_ao_basis(ref)

        mp2a = mp2diff_ao[0].to_ndarray()
        mp2b = mp2diff_ao[1].to_ndarray()
        dipx_ao = ref.operators.provider_ao.electric_dipole[0]
        dipx_ref = np.sum(mp2a * dipx_ao) + np.sum(mp2b * dipx_ao)

        oo = np.sum(
            mp2diff_mo.oo.to_ndarray() * dipx_mo.oo.to_ndarray()
        )
        ov = 2.0 * np.sum(
            mp2diff_mo.ov.to_ndarray() * dipx_mo.ov.to_ndarray()
        )
        vv = np.sum(
            mp2diff_mo.vv.to_ndarray() * dipx_mo.vv.to_ndarray()
        )
        dipx_np = oo + ov + vv

        assert dipx_np == pytest.approx(product_trace(mp2diff_mo, dipx_mo))
        assert product_trace(mp2diff_mo, dipx_mo) == pytest.approx(dipx_ref)
        assert product_trace(dipx_mo, mp2diff_mo) == pytest.approx(dipx_ref)

    def test_product_trace_nonsymmetric(self):
        ref = testdata_cache.refstate("cn_sto3g", "gen")
        dipx_mo = ref.operators.electric_dipole[0]
        mp2diff_mo = adcc.LazyMp(ref).mp2_diffdm
        mp2diff_nosym = OneParticleOperator(ref.mospaces, is_symmetric=False)
        mp2diff_nosym.oo = mp2diff_mo.oo
        mp2diff_nosym.ov = mp2diff_mo.ov
        mp2diff_nosym.vv = mp2diff_mo.vv
        mp2diff_nosym.vo = zeros_like(mp2diff_mo.ov.transpose())
        mp2diff_ao = mp2diff_nosym.to_ao_basis(ref)

        mp2a = mp2diff_ao[0].to_ndarray()
        mp2b = mp2diff_ao[1].to_ndarray()
        dipx_ao = ref.operators.provider_ao.electric_dipole[0]
        dipx_ref = np.sum(mp2a * dipx_ao) + np.sum(mp2b * dipx_ao)

        oo = np.sum(
            mp2diff_nosym.oo.to_ndarray() * dipx_mo.oo.to_ndarray()
        )
        ov = np.sum(
            mp2diff_nosym.ov.to_ndarray() * dipx_mo.ov.to_ndarray()
        )
        vo = np.sum(
            mp2diff_nosym.vo.to_ndarray() * dipx_mo.ov.to_ndarray().T
        )
        vv = np.sum(
            mp2diff_nosym.vv.to_ndarray() * dipx_mo.vv.to_ndarray()
        )
        dipx_np = oo + ov + vo + vv

        assert dipx_np == pytest.approx(product_trace(mp2diff_nosym, dipx_mo))
        assert product_trace(mp2diff_nosym, dipx_mo) == pytest.approx(dipx_ref)
        assert product_trace(dipx_mo, mp2diff_nosym) == pytest.approx(dipx_ref)

    def test_product_trace_both_nonsymmetric(self):
        ref = testdata_cache.refstate("cn_sto3g", "gen")
        dipx_mo = ref.operators.electric_dipole[0]
        mp2diff_mo = adcc.LazyMp(ref).mp2_diffdm
        mp2diff_nosym = OneParticleOperator(ref.mospaces, is_symmetric=False)
        dipx_nosym = OneParticleOperator(ref.mospaces, is_symmetric=False)

        mp2diff_nosym.oo = mp2diff_mo.oo
        mp2diff_nosym.ov = mp2diff_mo.ov
        mp2diff_nosym.vv = mp2diff_mo.vv
        mp2diff_nosym.vo = zeros_like(mp2diff_mo.ov.transpose())
        mp2diff_ao = mp2diff_nosym.to_ao_basis(ref)

        dipx_nosym.oo = dipx_mo.oo
        dipx_nosym.ov = dipx_mo.ov
        dipx_nosym.vv = dipx_mo.vv
        dipx_nosym.vo = zeros_like(dipx_mo.ov.transpose())
        dipx_ao = dipx_nosym.to_ao_basis(ref)

        mp2a = mp2diff_ao[0].to_ndarray()
        mp2b = mp2diff_ao[1].to_ndarray()
        dipxa = dipx_ao[0].to_ndarray()
        dipxb = dipx_ao[1].to_ndarray()
        dipx_ref = np.sum(mp2a * dipxa) + np.sum(mp2b * dipxb)

        oo = np.sum(
            mp2diff_nosym.oo.to_ndarray() * dipx_nosym.oo.to_ndarray()
        )
        ov = np.sum(
            mp2diff_nosym.ov.to_ndarray() * dipx_nosym.ov.to_ndarray()
        )
        vo = np.sum(
            mp2diff_nosym.vo.to_ndarray() * dipx_nosym.vo.to_ndarray()
        )
        vv = np.sum(
            mp2diff_nosym.vv.to_ndarray() * dipx_nosym.vv.to_ndarray()
        )
        dipx_np = oo + ov + vo + vv

        assert dipx_np == pytest.approx(product_trace(mp2diff_nosym, dipx_nosym))
        assert product_trace(mp2diff_nosym, dipx_nosym) == pytest.approx(dipx_ref)
        assert product_trace(dipx_nosym, mp2diff_nosym) == pytest.approx(dipx_ref)

    #
    # Test operators
    #
    def test_copy(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        mp2diff = adcc.LazyMp(ref).mp2_diffdm
        cpy = mp2diff.copy()

        assert cpy.blocks == mp2diff.blocks
        assert cpy.blocks_nonzero == mp2diff.blocks_nonzero
        assert cpy.reference_state == mp2diff.reference_state
        assert cpy.mospaces == mp2diff.mospaces

        for b in mp2diff.blocks:
            assert cpy.is_zero_block(b) == mp2diff.is_zero_block(b)
            if not mp2diff.is_zero_block(b):
                assert_equal(cpy.block(b).to_ndarray(),
                             mp2diff.block(b).to_ndarray())
                assert cpy.block(b) is not mp2diff.block(b)

    def test_add(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=False)
        a.oo.set_random()
        a.vo.set_random()
        a.vv.set_random()

        b = OneParticleOperator(ref.mospaces, is_symmetric=True)
        b.oo.set_random()
        b.vv.set_random()

        assert_array_almost_equal_nulp((a + b).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())
        assert_array_almost_equal_nulp((b + a).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())

    def test_add_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=True)
        a.oo.set_random()
        a.ov.set_random()
        a.vv.set_random()

        b = OneParticleOperator(ref.mospaces, is_symmetric=False)
        b.oo.set_random()
        b.ov.set_random()
        b.vo.set_random()
        b.vv.set_random()

        assert_array_almost_equal_nulp((a + b).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())
        assert_array_almost_equal_nulp((b + a).to_ndarray(),
                                       b.to_ndarray() + a.to_ndarray())

    def test_iadd(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=False)
        a.oo.set_random()
        a.vo.set_random()
        a.ov.set_random()
        a.vv.set_random()

        b = OneParticleOperator(ref.mospaces, is_symmetric=False)
        b.oo.set_random()
        b.ov.set_random()
        b.vv.set_random()

        ref = a.to_ndarray() + b.to_ndarray()
        a += b
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    def test_sub(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=True)
        a.oo.set_random()
        a.ov.set_random()
        a.vv.set_random()

        b = OneParticleOperator(ref.mospaces, is_symmetric=True)
        b.oo.set_random()
        b.ov.set_random()
        b.vv.set_random()

        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       a.to_ndarray() - b.to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       b.to_ndarray() - a.to_ndarray())
        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       (a + (-1 * b)).to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       (b + (-1 * a)).to_ndarray())

    def test_sub_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=True)
        a.oo.set_random()
        a.ov.set_random()
        a.vv.set_random()

        b = OneParticleOperator(ref.mospaces, is_symmetric=False)
        b.oo.set_random()
        b.ov.set_random()
        b.vo.set_random()
        b.vv.set_random()

        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       a.to_ndarray() - b.to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       b.to_ndarray() - a.to_ndarray())
        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       (a + (-1.0 * b)).to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       (b + (-1.0 * a)).to_ndarray())

    def test_isub(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=True)
        a.oo.set_random()
        a.ov.set_random()

        b = OneParticleOperator(ref.mospaces, is_symmetric=True)
        b.ov.set_random()
        b.vv.set_random()

        ref = a.to_ndarray() - b.to_ndarray()
        a -= b
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    def test_mul(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=True)
        a.oo.set_random()
        a.ov.set_random()
        assert_array_almost_equal_nulp((1.2 * a).to_ndarray(),
                                       1.2 * a.to_ndarray())

    def test_rmul(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=False)
        a.oo.set_random()
        a.vo.set_random()
        a.ov.set_random()
        assert_array_almost_equal_nulp((a * -1.8).to_ndarray(),
                                       -1.8 * a.to_ndarray())

    def test_imul(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=False)
        a.oo.set_random()
        a.vo.set_random()
        a.ov.set_random()
        a.vv.set_random()

        ref = 12 * a.to_ndarray()
        a *= 12
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    def test_block_functions(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, is_symmetric=True)
        # no AO transformation with only zero blocks possible
        with pytest.raises(ValueError):
            a.to_ao_basis(ref)
        a.oo.set_random()
        a.ov.set_random()
        a.vv.set_random()
        assert a.size == a.shape[0] * a.shape[1]
        assert not a.is_zero_block("v1o1")
        a.set_zero_block("o1o1")
        assert a.is_zero_block("o1o1")
        # access to zero blocks forbidden via block function
        with pytest.raises(KeyError):
            a.block("o1o1")
        # invalid block names
        with pytest.raises(KeyError):
            a["xyz"]
        with pytest.raises(KeyError):
            a["xyz"] = a.oo
        with pytest.raises(KeyError):
            a.set_zero_block("xyz")
        # invalid tensor shape
        with pytest.raises(ValueError):
            a.oo = a.ov
        # shortcuts
        np.testing.assert_allclose(a.oo.to_ndarray(),
                                   a["o1o1"].to_ndarray())
