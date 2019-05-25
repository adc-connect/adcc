#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import pytest
import itertools

from .misc import expand_test_templates

import adcc

from adcc.testdata.cache import cache
from adcc.testdata import geometry

from .test_ReferenceState_refdata import compare_refstate_with_reference

# The methods to test
testcases = cache.hfimport.keys()

backends = ["pyscf", "psi4", "veloxchem"]


@expand_test_templates(list(itertools.product(testcases, backends)))
class TestBackendsImportReferenceData(unittest.TestCase):
    def base_test(self, system, backend, case):
        data = cache.hfdata[system]
        reference = cache.hfimport[system][case]

        if not data["restricted"]:
            pytest.skip("Unrestricted calculations not supported.")
        if not adcc.backends.have_backend(backend):
            pytest.skip("Backend {} not available.".format(backend))

        basis = system.split("_")[-1]
        molecule = system.split("_")[0]

        if basis == "def2tzvp" and \
           backend == "veloxchem" and molecule == "h2o":
            pytest.skip("VeloxChem does not support f-functions.")

        hfres = adcc.backends.run_hf(
            backend, xyz=geometry.xyz[molecule],
            basis=basis, conv_tol=1e-14, conv_tol_grad=1e-12,
        )
        scfres = adcc.backends.import_scf_results(hfres)
        compare_refstate_with_reference(
            data, reference, case, scfres, compare_orbcoeff=False,
            compare_eri_almost_abs=True
        )

    def template_generic(self, case, backend):
        self.base_test(case, backend, "gen")

    def template_cvs(self, case, backend):
        self.base_test(case, backend, "cvs")
