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
import adcc
import itertools
import adcc.backends

from numpy.testing import assert_allclose

import pytest

from ..misc import expand_test_templates
from .testing import cached_backend_hf
from ..testdata.static_data import pe_potentials
from ..testdata.qchem import qchem_data

from scipy import constants
eV = constants.value("Hartree energy in eV")


backends = [b for b in adcc.backends.available()
            if b not in ["molsturm", "veloxchem"]]
basissets = ["sto3g", "ccpvdz"]
methods = ["adc1", "adc2", "adc3"]


@pytest.mark.skipif(len(backends) == 0,
                    reason="No backend found.")
@expand_test_templates(list(itertools.product(basissets, methods, backends)))
class TestPolarizableEmbedding(unittest.TestCase):
    def template_pe_formaldehyde(self, basis, method, backend):
        basename = f"formaldehyde_{basis}_pe_{method}"
        qc_result = qchem_data[basename]
        scfres = cached_backend_hf(backend, "formaldehyde", basis,
                                   potfile=pe_potentials["fa_6w"])
        state = adcc.run_adc(scfres, method=method,
                             n_singlets=5, conv_tol=1e-10)
        assert_allclose(
            qc_result["excitation_energies_ev"],
            state.excitation_energies_uncorrected * eV,
            atol=1e-5
        )
        assert_allclose(
            qc_result["excitation_energies_ev"]
            + qc_result["pe_ptss_corrections_ev"]
            + qc_result["pe_ptlr_corrections_ev"],
            state.excitation_energies * eV,
            atol=1e-5
        )
        assert_allclose(
            qc_result["pe_ptss_corrections_ev"],
            state.pe_ptss_correction * eV,
            atol=1e-5
        )
        assert_allclose(
            qc_result["pe_ptlr_corrections_ev"],
            state.pe_ptlr_correction * eV,
            atol=1e-5
        )
