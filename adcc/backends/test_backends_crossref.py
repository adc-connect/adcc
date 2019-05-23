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
import itertools
import numpy as np
import adcc
import adcc.backends

from numpy.testing import assert_allclose

from adcc.backends import have_backend
from adcc.testdata import geometry

import pytest

from ..misc import expand_test_templates

backends = ["pyscf", "psi4", "veloxchem"]


def compare_adc_results(adc_results, atol):
    for comb in list(itertools.combinations(adc_results, r=2)):
        state1 = adc_results[comb[0]]
        state2 = adc_results[comb[1]]
        assert_allclose(
            state1.eigenvalues, state2.eigenvalues
        )
        assert state1.n_iter == state2.n_iter

        blocks1 = state1.eigenvectors[0].blocks
        blocks2 = state2.eigenvectors[0].blocks
        assert blocks1 == blocks2
        for v1, v2 in zip(state1.eigenvectors, state2.eigenvectors):
            for block in blocks1:
                v1np = v1[block].to_ndarray()
                v2np = v2[block].to_ndarray()
                nonz_count1 = np.count_nonzero(np.abs(v1np) >= atol)
                if nonz_count1 == 0:
                    # Only zero elements in block.
                    continue
                # correcting the sign does not seem to
                # work for the doubles part of the vectors, probably
                # because of the anti-symmetry of the vector? (ms)

                # find non-zero elements in vector
                # nonz = np.nonzero((np.abs(v1np) >= 1e-14))
                # first_nonz = tuple(i[0] for i in nonz)
                # sgn1 = np.sign(v1np.item(first_nonz))
                # sgn2 = np.sign(v2np.item(first_nonz))
                # if sgn1 != sgn2:
                #     v2np *= -1.0
                assert_allclose(
                    np.abs(v1np), np.abs(v2np), atol=10 * atol,
                    err_msg="ADC vectors are not equal"
                            "in block {}".format(block)
                )


basissets = ["sto3g", "ccpvdz"]


@expand_test_templates(basissets)
class TestCrossReferenceBackends(unittest.TestCase):
    def run_adc(self, scfres, conv_tol):
        return adcc.adc2(scfres, n_singlets=5, conv_tol=conv_tol)

    def run_cvs_adc(self, scfres, conv_tol, n_core_orbitals=1):
        return adcc.cvs_adc2(
            scfres, n_singlets=5, n_core_orbitals=n_core_orbitals,
            conv_tol=conv_tol
        )

    def template_rhf_h2o(self, basis):
        backend_avail = [b for b in backends if have_backend(b)]
        if len(backend_avail) < 2:
            pytest.skip(
                "Not enough backends available for cross reference test."
                "Need at least 2."
            )
        h2o = geometry.xyz["h2o"]

        adc_results = {}
        cvs_results = {}
        for b in backend_avail:
            scfres = adcc.backends.run_hf(
                b, xyz=h2o, basis=basis, conv_tol_grad=1e-11
            )
            adc_res = self.run_adc(scfres, conv_tol=1e-10)
            adc_results[b] = adc_res
            cvs_res = self.run_cvs_adc(scfres, conv_tol=1e-10)
            cvs_results[b] = cvs_res

        compare_adc_results(adc_results, 5e-9)
        compare_adc_results(cvs_results, 5e-9)
