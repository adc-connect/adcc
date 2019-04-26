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
import unittest
import numpy as np

import adcc

from adcc.testdata.cache import cache


class TestOneParticleDensityMatrix(unittest.TestCase):
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
