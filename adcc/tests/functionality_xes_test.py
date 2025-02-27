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
import unittest
import pytest
import numpy as np
from numpy.testing import assert_allclose

import adcc
from adcc.backends import have_backend

from . import testcases


@pytest.mark.skipif(not have_backend("pyscf"), reason="pyscf not found.")
class TestFunctionalityXes(unittest.TestCase):
    # Test for XES calculations using pyscf / adcc

    def base_test(self, system: str, ref: dict):
        from adcc.backends.pyscf import run_core_hole

        system: testcases.TestCase = testcases.get_by_filename(system).pop()

        mf = run_core_hole(system.xyz, system.basis)
        state = adcc.adc2x(mf, conv_tol=1e-7, n_states=len(ref["eigenvalues"]))

        assert_allclose(state.excitation_energy, ref["eigenvalues"], atol=1e-6)

        # Computing the dipole moment implies a lot of cancelling in the
        # contraction, which has quite an impact on the accuracy.
        assert_allclose(state.oscillator_strength, ref["oscillator_strengths"],
                        atol=1e-4)
        assert_allclose(state.state_dipole_moment, ref["state_dipole_moments"],
                        atol=1e-4)

    def test_h2o_sto3g_adc2x_xes_singlets(self):
        ref = {}
        ref["eigenvalues"] = np.array([
            -19.61312030966488, -19.53503046052984, -19.30567103495504,
            -19.03686703652111, -18.97033973109105, -18.95165735507893,
            -18.94994083890595, -18.90692826540985, -18.88461188150268,
            -18.87778442694299])

        ref["oscillator_strengths"] = np.array([
            0.05035461011516018, 0.03962058660581094, 0.031211292115201725,
            1.136150679562653e-06, 0.00016786705751929378, 1.149042625218289e-09,
            0.0002824180316791623, 8.321992182153297e-05, 1.0616015344521101e-10,
            7.574230970577102e-10])

        ref["state_dipole_moments"] = np.array([
            [0.8806045671880932, 1.2064165118168874e-13, 0.6222480112064355],
            [0.6940504287188208, -1.4395573883255837e-13, 0.49028206407194075],
            [1.0467560044828614, -2.960053355673259e-14, 0.7394274748687037],
            [0.22520990007716524, -1.284945843433397e-11, 0.16778189446510727],
            [0.36531640078799776, 1.2660291475935138e-09, 0.26443331112341384],
            [0.37666647657062025, 1.6550371042775045e-07, 0.25756747786732237],
            [0.1752216284582866, 1.6992711802466892e-09, 0.13405532923190466],
            [0.2602860685582161, 7.339227199973813e-10, 0.1928141069176047],
            [0.43017102549320574, -1.7386733480222987e-08, 0.3115788202607428],
            [0.3521150147399802, -4.259803461929766e-07, 0.23755654819723393]
        ])
        self.base_test("h2o_sto3g", ref)

    def test_h2o_ccpvdz_adc2x_xes_singlets(self):
        ref = {}
        ref["eigenvalues"] = np.array([
            -19.519700798837476, -19.445092829708972, -19.269887129049895,
            -18.998700591075018, -18.948428933760578])
        ref["oscillator_strengths"] = np.array([
            0.052448756411532765, 0.04381087006608744, 0.037804719841347485,
            5.153812050043302e-06, 3.790267234406099e-05])
        ref["state_dipole_moments"] = np.array([
            [0.8208067509364902, -1.048772874567675e-13, 0.5798211061449533],
            [0.6800814813832732, -1.2487083270270364e-13, 0.4802418117812448],
            [0.993795308970137, -1.9141243421572952e-13, 0.7017671690535234],
            [-0.16173588644603332, -4.85408235339015e-09, -0.1088534193905284],
            [-0.19851759440650885, -1.825542427613543e-07, -0.1356563992145512]
        ])
        self.base_test("h2o_ccpvdz", ref)
