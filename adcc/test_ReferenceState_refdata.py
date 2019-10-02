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
import ast
import adcc
import unittest
import numpy as np

from numpy.testing import assert_allclose, assert_almost_equal
from adcc.testdata.cache import cache

from .misc import expand_test_templates


def compare_refstate_with_reference(
    data, reference, spec, scfres=None, compare_orbcoeff=True,
    compare_eri_almost_abs=False
):
    atol = data["conv_tol"]
    import_data = data
    if scfres:
        import_data = scfres

    # TODO once hfdata is an HDF5 file
    # refcases = ast.literal_eval(data["reference_cases"][()])
    refcases = ast.literal_eval(data["reference_cases"])
    refstate = adcc.ReferenceState(import_data, **refcases[spec])
    subspaces = {
        "gen": ["o1", "v1"], "cvs": ["o1", "o2", "v1"],
        "fc": ["o1", "o3", "v1"], "fv": ["o1", "v1", "v2"],
        "fc-fv": ["o1", "o3", "v1", "v2"], "fv-cvs": ["o1", "o2", "v1", "v2"]
    }[spec]

    assert subspaces == [e.decode() for e in reference["subspaces"]]
    # General properties
    assert refstate.restricted == data["restricted"]
    assert refstate.spin_multiplicity == (1 if data["restricted"] else 0)
    assert refstate.has_core_occupied_space == ("o2" in subspaces)
    assert refstate.irreducible_representation == "A"
    assert refstate.n_orbs == 2 * data["n_orbs_alpha"]
    assert refstate.n_orbs_alpha == data["n_orbs_alpha"]
    assert refstate.n_orbs_beta == data["n_orbs_alpha"]
    assert refstate.n_alpha == sum(data["occupation_f"][:refstate.n_orbs_alpha])
    assert refstate.n_beta == sum(data["occupation_f"][refstate.n_orbs_alpha:])
    assert refstate.conv_tol == data["conv_tol"]
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


# The methods to test
all_cases = cache.hfimport.keys()
testcases = []
for spec in ["gen", "cvs", "fc", "fv", "fc_fv", "fv_cvs"]:
    testcases.extend([(spec, case) for case in all_cases
                      if spec.replace("_", "-") in cache.hfimport[case]])


@expand_test_templates(testcases)
class TestReferenceStateReferenceData(unittest.TestCase):
    def template_hfimport(self, spec, case):
        spec = spec.replace("_", "-")
        data = cache.hfdata[case]
        reference = cache.hfimport[case][spec]
        compare_refstate_with_reference(data, reference, spec)
