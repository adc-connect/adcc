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
import adcc
import unittest
import numpy as np

from numpy.testing import assert_allclose, assert_almost_equal
from adcc.testdata.cache import cache

from .misc import expand_test_templates

# The methods to test
testcases = cache.hfimport.keys()


def compare_refstate_with_reference(
    data, reference, case, scfres=None, compare_orbcoeff=True,
    compare_eri_almost_abs=False
):
    atol = data["threshold"]
    import_data = data
    if scfres:
        import_data = scfres

    if case == "cvs":
        refstate = adcc.ReferenceState(
            import_data, core_orbitals=data["n_core_orbitals"]
        )
        subspaces = ["o1", "o2", "v1"]
    else:
        refstate = adcc.ReferenceState(import_data)
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
    assert_allclose(refstate.energy_scf, data["energy_scf"], atol=atol)
    assert refstate.mospaces.subspaces == subspaces

    multipoles = data['multipoles']
    assert_allclose(refstate.nuclear_total_charge,
                    multipoles["nuclear_0"], atol=atol)
    assert_allclose(refstate.nuclear_dipole,
                    multipoles["nuclear_1"], atol=atol)

    if "electric_dipole" in refstate.operators.available and \
       "elec_1" in multipoles:
        refstate2 = adcc.ReferenceState(data)
        assert_allclose(refstate.dipole_moment,
                        refstate2.dipole_moment, atol=atol)

    for ss in subspaces:
        assert_allclose(refstate.orbital_energies(ss).to_ndarray(),
                        reference["orbital_energies"][ss], atol=atol)

    if compare_orbcoeff:
        for ss in subspaces:
            orbcoeff = refstate.orbital_coefficients(ss + "b").to_ndarray()
            orbcoeff_ref = reference["orbital_coefficients"][ss + "b"]
            assert_allclose(orbcoeff, orbcoeff_ref, atol=atol)

    for ss in reference["fock"].keys():
        assert_allclose(refstate.fock(ss).to_ndarray(),
                        reference["fock"][ss], atol=atol)

    if compare_eri_almost_abs:
        for ss in reference["eri"].keys():
            assert_almost_equal(np.abs(refstate.eri(ss).to_ndarray()),
                                np.abs(reference["eri"][ss]))
    else:
        for ss in reference["eri"].keys():
            assert_allclose(refstate.eri(ss).to_ndarray(),
                            reference["eri"][ss], atol=atol)


@expand_test_templates(testcases)
class TestReferenceStateReferenceData(unittest.TestCase):
    def base_test(self, system, case):
        data = cache.hfdata[system]
        reference = cache.hfimport[system][case]
        compare_refstate_with_reference(data, reference, case)

    def template_generic(self, case):
        self.base_test(case, "gen")

    def template_cvs(self, case):
        self.base_test(case, "cvs")
