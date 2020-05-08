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
import numpy as np

from .misc import expand_test_templates

from numpy.testing import assert_allclose
from adcc.testdata.cache import cache

from pytest import approx

basemethods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]
methods = [m for bm in basemethods for m in [bm, "cvs_" + bm]]


@expand_test_templates(methods)
class RunnersConsistency:
    def base_test(self, *args, **kwargs):
        raise NotImplementedError

    def template_methox_sto3g_singlet(self, method):
        self.base_test("methox_sto3g", method, "singlet")


class TestMagneticTransitionDipoleMoments(unittest.TestCase, RunnersConsistency):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        res_dms = state.transition_magnetic_dipole_moments
        ref = refdata[method][kind]
        n_ref = len(state.excitation_vectors)
        assert_allclose(
            res_dms, ref["transition_magnetic_dipole_moments"][:n_ref], atol=1e-4
        )


class TestTransitionDipoleMomentsVelocity(unittest.TestCase, RunnersConsistency):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        res_dms = state.transition_dipole_moments_velocity
        ref = refdata[method][kind]
        n_ref = len(state.excitation_vectors)
        assert_allclose(
            res_dms, ref["transition_dipole_moments_velocity"][:n_ref], atol=1e-4
        )


class TestRotatoryStrengths(unittest.TestCase, RunnersConsistency):
    def base_test(self, system, method, kind):
        method = method.replace("_", "-")

        refdata = cache.reference_data[system]
        state = cache.adc_states[system][method][kind]

        res_rots = state.rotatory_strengths
        ref_tmdm = refdata[method][kind]["transition_magnetic_dipole_moments"]
        ref_tdmvel = refdata[method][kind]["transition_dipole_moments_velocity"]
        refevals = refdata[method][kind]["eigenvalues"]
        n_ref = len(state.excitation_vectors)
        for i in range(n_ref):
            assert state.excitation_energies[i] == refevals[i]
            ref_dot = np.dot(ref_tmdm[i], ref_tdmvel[i])
            assert res_rots[i] == approx(ref_dot / refevals[i])
