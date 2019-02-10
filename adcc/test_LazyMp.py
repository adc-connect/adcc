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

from adcc.testdata.cache import cache
from pytest import approx
import unittest


class TestLazyMp2(unittest.TestCase):
    # TODO Later bootstrap MP2 fully from hfdata

    def test_df(self):
        mp = cache.prelim["h2o_sto3g"].ground_state
        refdata = cache.reference_data["h2o_sto3g"]

        assert mp.df("o1v1").to_ndarray() == approx(refdata["mp1"]["df_o1v1"])

    def test_t2(self):
        mp = cache.prelim["h2o_sto3g"].ground_state
        refdata = cache.reference_data["h2o_sto3g"]

        assert mp.t2("o1o1v1v1").to_ndarray() == \
            approx(refdata["mp1"]["t_o1o1v1v1"])

    def test_td(self):
        mp = cache.prelim["h2o_sto3g"].ground_state
        refdata = cache.reference_data["h2o_sto3g"]

        assert mp.td2("o1o1v1v1").to_ndarray() == \
            approx(refdata["mp2"]["td_o1o1v1v1"])

    def test_mp2_density_mo(self):
        mp2diff = cache.prelim["h2o_sto3g"].ground_state.mp2_diffdm
        refdata = cache.reference_data["h2o_sto3g"]

        assert mp2diff.is_symmetric
        assert mp2diff["o1o1"].to_ndarray() == approx(refdata["mp2"]["dm_o1o1"])
        assert mp2diff["o1v1"].to_ndarray() == approx(refdata["mp2"]["dm_o1v1"])
        assert mp2diff["v1v1"].to_ndarray() == approx(refdata["mp2"]["dm_v1v1"])

    def test_mp2_density_ao(self):
        mp = cache.prelim["h2o_sto3g"].ground_state
        refdata = cache.reference_data["h2o_sto3g"]

        mp2diff = mp.mp2_diffdm
        dm_ao = mp2diff.transform_to_ao_basis(mp.reference_state)
        dm_alpha, dm_beta = dm_ao

        assert dm_alpha.to_ndarray() == approx(refdata["mp2"]["dm_bb_a"])
        assert dm_beta.to_ndarray() == approx(refdata["mp2"]["dm_bb_b"])
