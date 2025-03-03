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
import unittest
import numpy as np

from .testdata_cache import testdata_cache


class TestHardCodedCisResults(unittest.TestCase):
    def test_methox_sto3g_singlet(self):
        vlx_result = {
            'excitation_energies': np.array(
                [0.37722854, 0.42890135, 0.50227467, 0.52006496,
                 0.55583901, 0.59846662]
            ),
            'rotatory_strengths': np.array(
                [-0.00170464, 0.00191897, 0.02054426, -0.00429405,
                 0.04966345, 0.03289564]
            ),
            'oscillator_strengths': np.array(
                [1.93724710e-04, 1.29809765e-03, 3.28195555e-01,
                 2.55565534e-02, 3.10309645e-01, 1.43808081e-02]
            )
        }
        hf = testdata_cache._load_hfdata("r2methyloxirane_sto3g")
        state = adcc.cis(hf, n_singlets=6, conv_tol=1e-8)
        np.testing.assert_allclose(vlx_result['excitation_energies'],
                                   state.excitation_energy, atol=1e-6)
        np.testing.assert_allclose(vlx_result['rotatory_strengths'],
                                   state.rotatory_strength('origin'), atol=1e-4)
        np.testing.assert_allclose(vlx_result['oscillator_strengths'],
                                   state.oscillator_strength, atol=1e-4)
