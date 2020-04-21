#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import unittest
import itertools
import numpy as np
import adcc
import adcc.backends

from numpy.testing import assert_allclose

import pytest

from ..misc import expand_test_templates
from .testing import cached_backend_hf

# molsturm is super slow
backends = [b for b in adcc.backends.available() if b != "molsturm"]
basissets = ["sto3g", "sto3g", "ccpvdz"]


@pytest.mark.skipif(len(backends) < 2,
                    reason="Need at least two available backends for cross "
                    "reference test.")
@expand_test_templates(basissets)
class TestCrossReferenceBackends(unittest.TestCase):
    def template_adc2_h2o(self, basis):
        results = {}
        for b in backends:
            scfres = cached_backend_hf(b, "h2o", basis)
            results[b] = adcc.adc2(scfres, n_singlets=5, conv_tol=1e-10)
        compare_adc_results(results, 5e-9)

    def template_adc2_uhf_ch2nh2(self, basis):
        results = {}
        # UHF not supported for VeloxChem
        if "veloxchem" in backends:
            backends.remove("veloxchem")
        if not len(backends):
            pytest.skip("Not enough backends that support UHF available.")
        for b in backends:
            scfres = cached_backend_hf(b, "ch2nh2", basis, multiplicity=2)
            results[b] = adcc.adc2(scfres, n_states=5, conv_tol=1e-10)
        compare_adc_results(results, 5e-9)

    def template_cvs_adc2_h2o(self, basis):
        results = {}
        for b in backends:
            scfres = cached_backend_hf(b, "h2o", basis)
            results[b] = adcc.cvs_adc2(scfres, n_singlets=5, core_orbitals=1,
                                       conv_tol=1e-10)
        compare_adc_results(results, 5e-9)

    def template_hf_properties_h2o(self, basis):
        results = {}
        for b in backends:
            results[b] = adcc.ReferenceState(cached_backend_hf(b, "h2o", basis))
        compare_hf_properties(results, 5e-9)


def compare_hf_properties(results, atol):
    for comb in itertools.combinations(results, r=2):
        refstate1 = results[comb[0]]
        refstate2 = results[comb[1]]

        assert_allclose(refstate1.nuclear_total_charge,
                        refstate2.nuclear_total_charge, atol=atol)
        assert_allclose(refstate1.nuclear_dipole,
                        refstate2.nuclear_dipole, atol=atol)

        if "electric_dipole" in refstate1.operators.available and \
           "electric_dipole" in refstate2.operators.available:
            assert_allclose(refstate1.dipole_moment,
                            refstate2.dipole_moment, atol=atol)


def compare_adc_results(adc_results, atol):
    for comb in itertools.combinations(adc_results, r=2):
        state1 = adc_results[comb[0]]
        state2 = adc_results[comb[1]]
        assert_allclose(
            state1.excitation_energies, state2.excitation_energies
        )
        assert state1.n_iter == state2.n_iter

        blocks1 = state1.excitation_vectors[0].blocks
        blocks2 = state2.excitation_vectors[0].blocks
        assert blocks1 == blocks2
        for v1, v2 in zip(state1.excitation_vectors, state2.excitation_vectors):
            for block in blocks1:
                v1np = v1[block].to_ndarray()
                v2np = v2[block].to_ndarray()
                nonz_count1 = np.count_nonzero(np.abs(v1np) >= atol)
                if nonz_count1 == 0:
                    # Only zero elements in block.
                    continue

                assert_allclose(
                    np.abs(v1np), np.abs(v2np), atol=10 * atol,
                    err_msg="ADC vectors are not equal"
                            "in block {}".format(block)
                )

        # test properties
        assert_allclose(state1.oscillator_strengths,
                        state2.oscillator_strengths, atol=atol)
        assert_allclose(state1.oscillator_strengths_velocity,
                        state2.oscillator_strengths_velocity, atol=atol)
        # TODO: currently always zero because we only test non-chiral molecules
        assert_allclose(state1.rotatory_strengths,
                        state2.rotatory_strengths, atol=atol)
        assert_allclose(state1.state_dipole_moments,
                        state2.state_dipole_moments, atol=atol)
        # TODO: use correct signfix (state-dependent) or test rotatory strength
        #  (chiral molecule) when implemented?
        assert_allclose(np.abs(state1.transition_magnetic_dipole_moments),
                        np.abs(state2.transition_magnetic_dipole_moments),
                        atol=atol)
