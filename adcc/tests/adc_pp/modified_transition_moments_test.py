#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import numpy as np

from adcc.adc_pp.modified_transition_moments import modified_transition_moments

from ..testdata_cache import testdata_cache
from .. import testcases


methods = ["adc0", "adc1", "adc2"]

test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz"
)
cases = [(case.file_name, c, kind)
         for case in test_cases for c in ["gen", "cvs"]
         for kind in ["singlet", "any"] if kind in case.kinds.pp]

operator_kinds = ["electric", "magnetic"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("system,case,kind", cases)
@pytest.mark.parametrize("op_kind", operator_kinds)
def test_modified_transition_moments(system: str, case: str, method: str, kind: str,
                                     op_kind: str):
    state = testdata_cache.adcc_states(
        system=system, method=method, kind=kind, case=case
    )
    ref = testdata_cache.adcc_data(
        system=system, method=method, case=case
    )[kind]

    n_ref = len(state.excitation_vector)

    if op_kind == "electric":
        dips = state.reference_state.operators.electric_dipole
        ref_tdm = ref["transition_dipole_moments"]
    elif op_kind == "magnetic":
        dips = state.reference_state.operators.magnetic_dipole("origin")
        ref_tdm = ref["transition_magnetic_dipole_moments_origin"]
    else:
        raise NotImplementedError(
            f"Test not implemented for operator kind {op_kind}"
        )

    if "cvs" in case and "cvs" not in method:
        method = f"cvs-{method}"

    mtms = modified_transition_moments(method, state.ground_state, dips)

    for i in range(n_ref):
        # Computing the scalar product of the eigenvector
        # and the modified transition moments yields
        # the transition dipole moment (doi.org/10.1063/1.1752875)
        excivec = state.excitation_vector[i]
        res_tdm = np.array([excivec @ mtms[i] for i in range(3)])

        # Test norm and actual values
        res_tdm_norm = np.sum(res_tdm * res_tdm)
        ref_tdm_norm = np.sum(ref_tdm[i] * ref_tdm[i])
        assert res_tdm_norm == pytest.approx(ref_tdm_norm, abs=1e-8)
        np.testing.assert_allclose(res_tdm, ref_tdm[i], atol=1e-8)
