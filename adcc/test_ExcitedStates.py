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
import unittest
from numpy.testing import assert_allclose

from adcc.testdata.cache import cache
from adcc.OneParticleOperator import OneParticleOperator
from adcc.ExcitedStates import ExcitedStates
from .test_state_densities import Runners


class TestExcitationView(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")
        state = cache.adc_states[system][method][kind]
        n_ref = len(state.excitation_vectors)
        assert n_ref == state.size

        properties = {
            "excitation_energies": "excitation_energy",
            "oscillator_strengths": "oscillator_strength",
            "transition_dipole_moments": "transition_dipole_moment",
            "state_dipole_moments": "state_dipole_moment",
            "state_diffdms": "state_diffdm"
        }

        for i, exci in enumerate(state.excitations):
            for p in properties:
                ref = getattr(state, p)[i]
                res = getattr(exci, properties[p])
                if isinstance(ref, OneParticleOperator):
                    assert ref.blocks == res.blocks
                    for b in ref.blocks:
                        assert_allclose(ref[b].to_ndarray(), res[b].to_ndarray())
                else:
                    assert_allclose(ref, res)


class TestCustomExcitationEnergyCorrections(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")
        state = cache.adc_states[system][method][kind]

        def custom_correction1(exci):
            return exci.excitation_energy ** 2

        def custom_correction2(exci):
            return 2.0

        corrections = {
            "custom_correction1": custom_correction1,
            "custom_correction2": custom_correction2,
        }
        excitation_energies = state.excitation_energies.copy()

        state_corrected = ExcitedStates(state,
                                        excitation_energy_corrections=corrections)
        for i in range(state.size):
            assert hasattr(state_corrected, "custom_correction1")
            assert hasattr(state_corrected, "custom_correction2")
            assert_allclose(excitation_energies[i],
                            state_corrected.excitation_energies_uncorrected[i])
            corr = excitation_energies[i] ** 2 + 2.0
            assert_allclose(excitation_energies[i] + corr,
                            state_corrected.excitation_energies[i])
