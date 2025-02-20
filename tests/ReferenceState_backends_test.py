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
import pytest

import adcc

from . import testcases
from .testdata_cache import testdata_cache
from .backends.testing import cached_backend_hf
from .ReferenceState_refdata_test import compare_refstate_with_reference


test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz", "ch2nh2_sto3g"
)
cases = [(case.file_name, c) for case in test_cases for c in case.cases]
backends = [b for b in adcc.backends.available() if b != "molsturm"]


@pytest.mark.parametrize("system,case", cases)
@pytest.mark.parametrize("backend", backends)
def test_backends_import_reference_data(system: str, case: str, backend: str):
    system: testcases.TestCase = testcases.get_by_filename(system).pop()

    if backend == "veloxchem":
        if system.basis == "def2-tzvp" and system.name == "h2o":
            pytest.skip("VeloxChem does not support f-functions.")

    compare_eri = "abs"
    if system.name == "cn":
        compare_eri = "off"

    conv_tol = 1e-11
    if backend == "veloxchem":
        conv_tol = 1e-7
    data = testdata_cache._load_hfdata(system)  # is also cached
    reference = testdata_cache.hfimport(system, case=case)
    # perform a new scf calculation with the backend
    scfres = cached_backend_hf(backend=backend, system=system, conv_tol=conv_tol)
    compare_refstate_with_reference(
        system=system, case=case, data=data, reference=reference, scfres=scfres,
        compare_orbcoeff=False, compare_eri=compare_eri
    )
