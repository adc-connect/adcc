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

from .misc import expand_test_templates
from numpy.testing import assert_allclose

import adcc

from adcc.testdata.cache import cache

# The methods to test
testcases = cache.hfimport.keys()


@expand_test_templates(testcases)
class TestReferenceStateReferenceData(unittest.TestCase):
    def base_test(self, system, case):
        data = cache.hfdata[system]
        reference = cache.hfimport[system][case]
        atol = data["threshold"]

        if case == "cvs":
            refstate = adcc.ReferenceState(
                data, core_orbitals=data["n_core_orbitals"]
            )
            subspaces = ["o1", "o2", "v1"]
        else:
            refstate = adcc.ReferenceState(data)
            subspaces = ["o1", "v1"]
        assert subspaces == [e.decode() for e in reference["subspaces"]]

        # General properties
        assert refstate.restricted == data["restricted"]
        assert refstate.spin_multiplicity == (1 if data["restricted"] else 0)
        assert refstate.has_core_occupied_space == ("o2" in subspaces)
        assert refstate.irreducible_representation == "A"
        assert refstate.n_orbs == data["n_orbs_alpha"] + data["n_orbs_beta"]
        assert refstate.n_orbs_alpha == data["n_orbs_alpha"]
        assert refstate.n_orbs_beta == data["n_orbs_beta"]
        assert refstate.n_alpha == data["n_alpha"]
        assert refstate.n_beta == data["n_beta"]
        assert refstate.conv_tol == data["threshold"]
        assert refstate.energy_scf == data["energy_scf"]
        assert refstate.mospaces.subspaces == subspaces

        for ss in subspaces:
            assert_allclose(refstate.orbital_energies(ss).to_ndarray(),
                            reference["orbital_energies"][ss], atol=atol)
        for ss in subspaces:
            orbcoeff = refstate.orbital_coefficients(ss + "b").to_ndarray()
            orbcoeff_ref = reference["orbital_coefficients"][ss + "b"]
            assert_allclose(orbcoeff, orbcoeff_ref, atol=atol)

        for ss in reference["fock"].keys():
            assert_allclose(refstate.fock(ss).to_ndarray(),
                            reference["fock"][ss], atol=atol)

        for ss in reference["eri"].keys():
            assert_allclose(refstate.eri(ss).to_ndarray(),
                            reference["eri"][ss], atol=atol)

    def template_generic(self, case):
        self.base_test(case, "gen")

    def template_cvs(self, case):
        self.base_test(case, "cvs")
