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
import numpy as np

from numpy.testing import assert_allclose
from adcc.testdata.cache import cache

from .test_state_densities import Runners
from pytest import approx


class TestMagneticTransitionDipoleMoments(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")
        kind = "any" if kind == "state" else kind

        refdata = cache.adcc_reference_data[system]
        state = cache.adcc_states[system][method][kind]

        res_dms = state.transition_magnetic_dipole_moment
        ref = refdata[method][kind]
        n_ref = len(state.excitation_vector)
        assert_allclose(
            res_dms, ref["transition_magnetic_dipole_moments"][:n_ref], atol=1e-4
        )


class TestTransitionDipoleMomentsVelocity(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")
        kind = "any" if kind == "state" else kind

        refdata = cache.adcc_reference_data[system]
        state = cache.adcc_states[system][method][kind]

        res_dms = state.transition_dipole_moment_velocity
        ref = refdata[method][kind]
        n_ref = len(state.excitation_vector)
        assert_allclose(
            res_dms, ref["transition_dipole_moments_velocity"][:n_ref], atol=1e-4
        )


class TestRotatoryStrengths(unittest.TestCase, Runners):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")
        kind = "any" if kind == "state" else kind

        refdata = cache.adcc_reference_data[system]
        state = cache.adcc_states[system][method][kind]

        res_rots = state.rotatory_strength
        ref_tmdm = refdata[method][kind]["transition_magnetic_dipole_moments"]
        ref_tdmvel = refdata[method][kind]["transition_dipole_moments_velocity"]
        refevals = refdata[method][kind]["eigenvalues"]
        n_ref = len(state.excitation_vector)
        for i in range(n_ref):
            assert state.excitation_energy[i] == refevals[i]
            ref_dot = np.dot(ref_tmdm[i], ref_tdmvel[i])
            assert res_rots[i] == approx(ref_dot / refevals[i])
