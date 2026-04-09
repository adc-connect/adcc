#!/usr/bin/env python3
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
import numpy as np
import pytest

from adcc import run_adc
from .. import testcases
from ..testdata_cache import testdata_cache


test_cases = testcases.get_by_filename("h2o_sto3g", "cn_sto3g")
methods = ["adc0", "adc1", "adc2", "adc3"]


@pytest.mark.parametrize("case", test_cases,
                         ids=[f"{c.name}-{c.basis}" for c in test_cases])
@pytest.mark.parametrize("method", methods)
class TestTransitionDm:
    def test_adcn(self, method: str, case: str):
        hf = testdata_cache.refstate(case, "gen")
        state = run_adc(hf, n_states=3, method=method)
        n_electrons = len(hf.foo.diagonal())
        prefac = 1.0 / (n_electrons - 1)
        state_tdm_2p = state.transition_dm_2p
        state_tdm = state.transition_dm
        for es in range(3):
            # partial traces are not implemented in adcc.einsum
            oo_1p = prefac * (
                np.einsum("ikjk->ij", state_tdm_2p[es].oooo.to_ndarray())
                + np.einsum("icjc->ij", state_tdm_2p[es].ovov.to_ndarray())
            )

            vv_1p = prefac * (
                np.einsum("akbk->ab", state_tdm_2p[es].vovo.to_ndarray())
                + np.einsum("acbc->ab", state_tdm_2p[es].vvvv.to_ndarray())
            )

            ov_1p = prefac * (
                np.einsum("ikbk->ib", state_tdm_2p[es].oovo.to_ndarray())
                + np.einsum("icbc->ib", state_tdm_2p[es].ovvv.to_ndarray())
            )

            vo_1p = prefac * (
                np.einsum("akjk->aj", state_tdm_2p[es].vooo.to_ndarray())
                + np.einsum("acjc->aj", state_tdm_2p[es].vvov.to_ndarray())
            )

            np.testing.assert_allclose(
                state_tdm[es].oo.to_ndarray(), oo_1p, atol=1e-12
            )
            np.testing.assert_allclose(
                state_tdm[es].vv.to_ndarray(), vv_1p, atol=1e-12
            )
            np.testing.assert_allclose(
                state_tdm[es].ov.to_ndarray(), ov_1p, atol=1e-12
            )
            np.testing.assert_allclose(
                state_tdm[es].vo.to_ndarray(), vo_1p, atol=1e-12
            )
