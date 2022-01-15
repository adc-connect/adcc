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
import unittest
import numpy as np
import pytest
from numpy.testing import assert_allclose

import adcc
from .misc import expand_test_templates

methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]
methods.extend([f"cvs-{m}" for m in methods])

_singlets = {
    "adc0": [0.99048446, 1.04450741, 1.15356339],
    "adc1": [0.48343609, 0.57420044, 0.60213699],
    "adc2": [0.47051314, 0.57255495, 0.59367339],
    "adc2x": [0.44511182, 0.55139508, 0.57146893],
    "adc3": [0.45197068, 0.55572706, 0.57861241],
    "cvs-adc0": [20.83623681, 20.99931574],
    "cvs-adc1": [20.10577515, 20.16512502],
    "cvs-adc2": [20.0045422,  20.08771799],
    "cvs-adc2x": [19.90528006, 19.9990468],
    "cvs-adc3": [19.93138526, 20.02055267],
}

_triplets = {
    "adc0": [0.99048446, 1.04450741, 1.15356339],
    "adc1": [0.40544164, 0.48351268, 0.52489129],
    "adc2": [0.40288477, 0.4913253,  0.52854722],
    "adc2x": [0.38557787, 0.48177562, 0.51336301],
    "adc3": [0.3924605,  0.4877098,  0.51755258],
    "cvs-adc0": [20.83623681, 20.99931574],
    "cvs-adc1": [20.04302084, 20.12099631],
    "cvs-adc2": [19.95732865, 20.05239094],
    "cvs-adc2x": [19.86572075, 19.97241019],
    "cvs-adc3": [19.89172578, 19.99386631],
}


def _residual_norms(state):
    residuals = [
        state.matrix @ v - e * v
        for e, v in zip(state.excitation_energy, state.excitation_vector)
    ]
    return np.array([r @ r for r in residuals])


@pytest.mark.skipif("pyscf" not in adcc.backends.available(),
                    reason="PySCF not found.")
@expand_test_templates(methods)
class TestSmoke(unittest.TestCase):
    conv_tol = 1e-7

    def _run_scf_h2o_sto3g(self):
        # NOTE: cached_backend_hf not used because adcc.testdata
        # is not included in installation
        scfres = adcc.backends.run_hf(
            "pyscf", xyz="""
            O 0 0 0
            H 0 0 1.795239827225189
            H 1.693194615993441 0 -0.599043184453037""",
            basis="sto-3g", conv_tol=self.conv_tol / 100,
            conv_tol_grad=self.conv_tol / 10
        )
        return scfres

    def template_test_h2o_sto3g(self, method):
        scfres = self._run_scf_h2o_sto3g()
        if "cvs" in method:
            refstate = adcc.ReferenceState(scfres, core_orbitals=1)
            n_states = 2
        else:
            refstate = adcc.ReferenceState(scfres)
            n_states = 3
        state_singlets = adcc.run_adc(refstate, method=method,
                                      n_singlets=n_states, conv_tol=self.conv_tol)
        assert np.all(_residual_norms(state_singlets) < self.conv_tol)
        state_triplets = adcc.run_adc(refstate, method=method,
                                      n_triplets=n_states, conv_tol=self.conv_tol)
        assert np.all(_residual_norms(state_triplets) < self.conv_tol)
        assert_allclose(state_singlets.excitation_energy,
                        _singlets[method], atol=self.conv_tol * 10)
        assert_allclose(state_triplets.excitation_energy,
                        _triplets[method], atol=self.conv_tol * 10)
