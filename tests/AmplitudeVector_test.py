#!/usr/bin/env python3
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
import pytest
import unittest
import numpy as np

import adcc
from .testdata_cache import testdata_cache


class TestAmplitudeVector(unittest.TestCase):
    def test_functionality(self):
        ground_state = adcc.LazyMp(testdata_cache.refstate("h2o_sto3g", case="gen"))
        matrix = adcc.AdcMatrix("adc2", ground_state)
        vectors = [adcc.guess_zero(matrix) for _ in range(2)]
        for vec in vectors:
            vec.set_random()
        v, w = vectors
        with pytest.raises(AttributeError):
            v.pph
        with pytest.raises(AttributeError):
            v.pph = w.ph
        # setattr with expression
        z = adcc.zeros_like(v)
        z.ph = v.ph + w.ph
        z -= w
        np.testing.assert_allclose(
            v.ph.to_ndarray(), z.ph.to_ndarray()
        )
