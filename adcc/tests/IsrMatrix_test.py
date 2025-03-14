#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2022 by the adcc authors
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

from adcc.IsrMatrix import IsrMatrix
from adcc.State2States import State2States

from .testdata_cache import testdata_cache
from . import testcases


test_cases = testcases.get_by_filename("h2o_sto3g", "cn_sto3g")
cases = [(case.file_name, c, kind)
         for case in test_cases for c in ["gen", "cvs"] for kind in case.kinds.pp]

methods = ["adc0", "adc1", "adc2"]
operator_kinds = ["electric", "magnetic"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("system,case,kind", cases)
@pytest.mark.parametrize("operator_kind", operator_kinds)
class TestIsrMatrix:
    def test_matrix_vector_product(self, system: str, case: str, kind: str,
                                   method: str, operator_kind: str):
        state = testdata_cache.adcc_states(
            system=system, method=method, kind=kind, case=case
        )

        n_ref = len(state.excitation_vector)
        mp = state.ground_state
        if operator_kind == "electric":  # example of a symmetric operator
            dips = state.reference_state.operators.electric_dipole
        elif operator_kind == "magnetic":  # example of an asymmetric operator
            dips = state.reference_state.operators.magnetic_dipole("origin")
        else:
            raise NotImplementedError(f"Unexptected operator kind {operator_kind}")

        if "cvs" in case:  # cvs not implemented
            if "cvs" not in method:
                method = f"cvs-{method}"
            with pytest.raises(ValueError):
                matrix = IsrMatrix(method, mp, dips)
            return

        # computing Y_m @ B @ Y_n yields the state-to-state
        # transition dipole moments (n->m) (for n not equal to m)
        # they can either be obtained using the matvec method of the IsrMatrix
        # class or via the state-to-state transition density matrices
        # (the second method serves as a reference here)

        matrix = IsrMatrix(method, mp, dips)
        for ifrom in range(n_ref - 1):
            B_Yn = matrix @ state.excitations[ifrom].excitation_vector
            state2state = State2States(state, initial=ifrom)
            for j, ito in enumerate(range(ifrom + 1, n_ref)):
                s2s_tdm = [state.excitations[ito].excitation_vector @ vec
                           for vec in B_Yn]

                if operator_kind == "electric":
                    s2s_tdm_ref = state2state.transition_dipole_moment[j]
                else:
                    s2s_tdm_ref = \
                        state2state.transition_magnetic_dipole_moment("origin")[j]
                np.testing.assert_allclose(s2s_tdm, s2s_tdm_ref, atol=1e-12)


class TestIsrMatrixInterface(unittest.TestCase):
    def test_matvec(self):
        system = "h2o_sto3g"
        method = "adc2"
        kind = "singlet"

        state = testdata_cache.adcc_states(
            system=system, method=method, kind=kind, case="gen"
        )
        assert len(state.excitation_vector) > 1
        mp = state.ground_state
        dips = state.reference_state.operators.electric_dipole
        magdips = state.reference_state.operators.magnetic_dipole('origin')
        vecs = [exc.excitation_vector for exc in state.excitations[:2]]

        matrix_ref = IsrMatrix(method, mp, dips)
        refv1, refv2 = matrix_ref @ vecs

        # multiple vectors
        for i, dip in enumerate(dips):
            matrix = IsrMatrix(method, mp, dip)
            resv1, resv2 = matrix @ vecs
            diffs = [refv1[i] - resv1, refv2[i] - resv2]
            for j in range(2):
                assert diffs[j].ph.dot(diffs[j].ph) < 1e-12
                assert diffs[j].pphh.dot(diffs[j].pphh) < 1e-12

        # compute Y_n @ B @ Y_n with matvec and rmatvec
        # symmetric operators
        for vec1 in vecs:
            for vec2 in vecs:
                resl = [vec1 @ mvprod for mvprod in matrix_ref.matvec(vec2)]
                resr = [mvprod @ vec2 for mvprod in matrix_ref.rmatvec(vec1)]
                np.testing.assert_allclose(
                    np.array(resl), np.array(resr), atol=1e-12
                )

        # anti-symmetric operators
        magmatrix = IsrMatrix(method, mp, magdips)
        for vec1 in vecs:
            for vec2 in vecs:
                resl = [vec1 @ mvprod for mvprod in magmatrix.matvec(vec2)]
                resr = [mvprod @ vec2 for mvprod in magmatrix.rmatvec(vec1)]
                np.testing.assert_allclose(
                    np.array(resl), np.array(resr), atol=1e-12
                )
