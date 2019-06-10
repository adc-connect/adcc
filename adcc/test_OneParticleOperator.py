#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import adcc
import unittest
import numpy as np

from adcc import OneParticleOperator, zeros_like
from adcc.OneParticleOperator import product_trace
from adcc.testdata.cache import cache

from pytest import approx


class TestOneParticleOperator(unittest.TestCase):
    def test_to_ndarray(self):
        mp2diff = adcc.LazyMp(cache.refstate["h2o_sto3g"]).mp2_diffdm

        dm_oo = mp2diff["o1o1"].to_ndarray()
        dm_ov = mp2diff["o1v1"].to_ndarray()
        dm_vv = mp2diff["v1v1"].to_ndarray()

        dm_o = np.hstack((dm_oo, dm_ov))
        dm_v = np.hstack((dm_ov.transpose(), dm_vv))
        dm_full = np.vstack((dm_o, dm_v))

        np.testing.assert_almost_equal(dm_full, mp2diff.to_ndarray(),
                                       decimal=12)

    def test_product_trace_symmetric(self):
        ref = cache.refstate["h2o_sto3g"]
        dipx_mo = ref.operators.electric_dipole[0]
        mp2diff_mo = adcc.LazyMp(ref).mp2_diffdm
        mp2diff_ao = mp2diff_mo.transform_to_ao_basis(ref)

        mp2a = mp2diff_ao[0].to_ndarray()
        mp2b = mp2diff_ao[1].to_ndarray()
        dipx_ao = ref.operators.provider_ao.electric_dipole[0]
        dipx_ref = np.sum(mp2a * dipx_ao) + np.sum(mp2b * dipx_ao)

        oo = np.sum(
            mp2diff_mo["o1o1"].to_ndarray() * dipx_mo["o1o1"].to_ndarray()
        )
        ov = 2.0 * np.sum(
            mp2diff_mo["o1v1"].to_ndarray() * dipx_mo["o1v1"].to_ndarray()
        )
        vv = np.sum(
            mp2diff_mo["v1v1"].to_ndarray() * dipx_mo["v1v1"].to_ndarray()
        )
        dipx_np = oo + ov + vv

        assert dipx_np == approx(product_trace(mp2diff_mo, dipx_mo))
        assert product_trace(mp2diff_mo, dipx_mo) == approx(dipx_ref)
        assert product_trace(dipx_mo, mp2diff_mo) == approx(dipx_ref)

    def test_product_trace_nonsymmetric(self):
        ref = cache.refstate["cn_sto3g"]
        dipx_mo = ref.operators.electric_dipole[0]
        mp2diff_mo = adcc.LazyMp(ref).mp2_diffdm
        mp2diff_nosym = OneParticleOperator(ref.mospaces, is_symmetric=False)
        mp2diff_nosym.set_block("o1o1", mp2diff_mo["o1o1"])
        mp2diff_nosym.set_block("o1v1", mp2diff_mo["o1v1"])
        mp2diff_nosym.set_block("v1v1", mp2diff_mo["v1v1"])
        mp2diff_nosym.set_block("v1o1",
                                zeros_like(mp2diff_mo["o1v1"].transpose()))
        mp2diff_ao = mp2diff_nosym.transform_to_ao_basis(ref)

        mp2a = mp2diff_ao[0].to_ndarray()
        mp2b = mp2diff_ao[1].to_ndarray()
        dipx_ao = ref.operators.provider_ao.electric_dipole[0]
        dipx_ref = np.sum(mp2a * dipx_ao) + np.sum(mp2b * dipx_ao)

        oo = np.sum(
            mp2diff_nosym["o1o1"].to_ndarray() * dipx_mo["o1o1"].to_ndarray()
        )
        ov = np.sum(
            mp2diff_nosym["o1v1"].to_ndarray() * dipx_mo["o1v1"].to_ndarray()
        )
        vo = np.sum(
            mp2diff_nosym["v1o1"].to_ndarray() * dipx_mo["o1v1"].to_ndarray().T
        )
        vv = np.sum(
            mp2diff_nosym["v1v1"].to_ndarray() * dipx_mo["v1v1"].to_ndarray()
        )
        dipx_np = oo + ov + vo + vv

        assert dipx_np == approx(product_trace(mp2diff_nosym, dipx_mo))
        assert product_trace(mp2diff_nosym, dipx_mo) == approx(dipx_ref)
        assert product_trace(dipx_mo, mp2diff_nosym) == approx(dipx_ref)

    def test_product_trace_both_nonsymmetric(self):
        ref = cache.refstate["cn_sto3g"]
        dipx_mo = ref.operators.electric_dipole[0]
        mp2diff_mo = adcc.LazyMp(ref).mp2_diffdm
        mp2diff_nosym = OneParticleOperator(ref.mospaces, is_symmetric=False)
        dipx_nosym = OneParticleOperator(ref.mospaces, is_symmetric=False)

        mp2diff_nosym.set_block("o1o1", mp2diff_mo["o1o1"])
        mp2diff_nosym.set_block("o1v1", mp2diff_mo["o1v1"])
        mp2diff_nosym.set_block("v1v1", mp2diff_mo["v1v1"])
        mp2diff_nosym.set_block("v1o1",
                                zeros_like(mp2diff_mo["o1v1"].transpose()))
        mp2diff_ao = mp2diff_nosym.transform_to_ao_basis(ref)

        dipx_nosym.set_block("o1o1", dipx_mo["o1o1"])
        dipx_nosym.set_block("o1v1", dipx_mo["o1v1"])
        dipx_nosym.set_block("v1v1", dipx_mo["v1v1"])
        dipx_nosym.set_block("v1o1", zeros_like(dipx_mo["o1v1"].transpose()))
        dipx_ao = dipx_nosym.transform_to_ao_basis(ref)

        mp2a = mp2diff_ao[0].to_ndarray()
        mp2b = mp2diff_ao[1].to_ndarray()
        dipxa = dipx_ao[0].to_ndarray()
        dipxb = dipx_ao[1].to_ndarray()
        dipx_ref = np.sum(mp2a * dipxa) + np.sum(mp2b * dipxb)

        oo = np.sum(
            mp2diff_nosym["o1o1"].to_ndarray() * dipx_nosym["o1o1"].to_ndarray()
        )
        ov = np.sum(
            mp2diff_nosym["o1v1"].to_ndarray() * dipx_nosym["o1v1"].to_ndarray()
        )
        vo = np.sum(
            mp2diff_nosym["v1o1"].to_ndarray() * dipx_nosym["v1o1"].to_ndarray()
        )
        vv = np.sum(
            mp2diff_nosym["v1v1"].to_ndarray() * dipx_nosym["v1v1"].to_ndarray()
        )
        dipx_np = oo + ov + vo + vv

        assert dipx_np == approx(product_trace(mp2diff_nosym, dipx_nosym))
        assert product_trace(mp2diff_nosym, dipx_nosym) == approx(dipx_ref)
        assert product_trace(dipx_nosym, mp2diff_nosym) == approx(dipx_ref)
