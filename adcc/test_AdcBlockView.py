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

from .misc import expand_test_templates
from adcc.testdata.cache import cache

methods = ["adc1", "adc2", "adc2x", "adc3", "cvs-adc1", "cvs-adc2x"]


@expand_test_templates(methods)
class TestAdcMatrixBlockView(unittest.TestCase):
    def template_singles_view(self, method):
        if "cvs" in method:
            reference_state = cache.refstate_cvs["h2o_sto3g"]
            shape = (8, 8)
            spaces_s = ["o2", "v1"]
        else:
            reference_state = cache.refstate["h2o_sto3g"]
            shape = (40, 40)
            spaces_s = ["o1", "v1"]
        matrix = adcc.AdcMatrix(method, adcc.LazyMp(reference_state))
        view = adcc.AdcBlockView(matrix, "s")

        assert view.ndim == 2
        assert view.is_core_valence_separated == ("cvs" in method)
        assert view.shape == shape
        assert len(view) == shape[0]

        assert view.blocks == ["s"]
        assert view.has_block("s")
        assert not view.has_block("d")
        assert not view.has_block("t")
        assert view.block_spaces("s") == spaces_s

        assert view.reference_state == reference_state
        assert view.mospaces == reference_state.mospaces
        assert isinstance(view.timer, adcc.timings.Timer)
        assert view.to_cpp() == matrix.to_cpp()

        # Check diagonal
        diff = matrix.diagonal("s") - view.diagonal("s")
        assert diff.dot(diff) < 1e-12

        # Check @ (one vector and multiple vectors)
        invec = adcc.guess_zero(matrix)
        invec["s"].set_random()
        # "d" left as zero
        invec_singles = adcc.AmplitudeVector(invec["s"])

        ref = matrix @ invec
        res = view @ invec_singles
        diff = res["s"] - ref["s"]
        assert diff.dot(diff) < 1e-12

        res = view @ [invec_singles, invec_singles, invec_singles]
        diff = [res[i]["s"] - ref["s"] for i in range(3)]
        assert max(d.dot(d) for d in diff) < 1e-12

        # Missing: Check for compute_matvec and compute_apply
