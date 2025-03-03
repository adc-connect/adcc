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
from . import testcases
from .testdata_cache import testdata_cache

import pytest

methods = ["adc1", "adc2", "adc2x", "adc3"]
# only do the test for a single testcase
h2o = testcases.get_by_filename("h2o_sto3g").pop()


@pytest.mark.parametrize("case", h2o.cases)
@pytest.mark.parametrize("method", methods)
class TestAdcMatrixBlockView:
    def test_singles_view(self, method, case):
        n_occ = 10
        n_virt = 4
        spaces_ph = ["o1", "v1"]
        if "cvs" in case:
            n_occ = 2 * h2o.core_orbitals  # alpha and beta
            spaces_ph = ["o2", "v1"]
            if "cvs" not in method:
                method = f"cvs-{method}"
        elif "fc" in case:
            n_occ -= 2 * h2o.frozen_core  # alpha and beta
        if "fv" in case:
            n_virt -= 2 * h2o.frozen_virtual  # alpha and beta
        shape = (n_occ * n_virt, n_occ * n_virt)

        reference_state = testdata_cache.refstate(h2o, case)
        matrix = adcc.AdcMatrix(method, adcc.LazyMp(reference_state))
        view = matrix.block_view("ph_ph")

        assert view.ndim == 2
        assert view.is_core_valence_separated == ("cvs" in method)
        assert view.shape == shape
        assert len(view) == shape[0]

        assert view.axis_blocks == ["ph"]
        assert sorted(view.axis_spaces.keys()) == view.axis_blocks
        assert sorted(view.axis_lengths.keys()) == view.axis_blocks
        assert view.axis_spaces["ph"] == spaces_ph
        assert view.axis_lengths["ph"] == shape[0]

        assert view.reference_state == reference_state
        assert view.mospaces == reference_state.mospaces
        assert isinstance(view.timer, adcc.timings.Timer)

        # Check diagonal
        diff = matrix.diagonal().ph - view.diagonal().ph
        assert diff.dot(diff) < 1e-12

        # Check @ (one vector and multiple vectors)
        invec = adcc.guess_zero(matrix)
        invec.ph.set_random()
        # "d" left as zero
        invec_singles = adcc.AmplitudeVector(ph=invec.ph)

        ref = matrix @ invec
        res = view @ invec_singles
        diff = res.ph - ref.ph
        assert diff.dot(diff) < 1e-12

        res = view @ [invec_singles, invec_singles, invec_singles]
        diff = [res[i].ph - ref.ph for i in range(3)]
        assert max(d.dot(d) for d in diff) < 1e-12

        # Missing: Check for matvec and block_apply
