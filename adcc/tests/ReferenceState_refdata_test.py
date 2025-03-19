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
import adcc
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from .testdata_cache import testdata_cache
from . import testcases


def compare_refstate_with_reference(system: str, case: str,
                                    data: dict, reference: dict,
                                    scfres=None, compare_orbcoeff: bool = True,
                                    compare_eri: str = "value"):
    # - data is the data to build a ReferenceState on top. If scfres is given
    # it will be used instead of data. We check that the data of the constructed
    # ReferenceState is consistent with data!
    # - reference is the dumped hfimport data against which we check that the
    # orbital energies, orbital coefficients, fock matrix and ERI are correct.
    if isinstance(system, str):
        system = testcases.get_by_filename(system).pop()
    assert isinstance(system, testcases.TestCase)

    # Extract convergence tolerance setting for comparison threshold
    if scfres is None:
        import_data = data
        atol = data["conv_tol"]
        backend = None
    else:
        import_data = scfres
        if hasattr(scfres, "conv_tol"):
            atol = scfres.conv_tol
            backend = scfres.backend
        else:
            atol = data["conv_tol"]
            backend = None
    # construct the ReferenceState
    core_orbitals = system.core_orbitals if "cvs" in case else None
    frozen_core = system.frozen_core if "fc" in case else None
    frozen_virtual = system.frozen_virtual if "fv" in case else None
    refstate = adcc.ReferenceState(
        import_data, core_orbitals=core_orbitals, frozen_core=frozen_core,
        frozen_virtual=frozen_virtual
    )
    # collect the subspaces for the case
    subspaces = ["o1", "v1"]
    if "cvs" in case:
        subspaces.append("o2")
    if "fc" in case:
        subspaces.append("o3")
    if "fv" in case:
        subspaces.append("v2")
    subspaces = sorted(subspaces)

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
    assert refstate.conv_tol == atol  # because atol is set to be the SCF conv_tol
    assert_allclose(refstate.energy_scf, data["energy_scf"], atol=atol)
    assert refstate.mospaces.subspaces == subspaces

    multipoles = data['multipoles']
    assert_allclose(refstate.nuclear_total_charge,
                    multipoles["nuclear_0"], atol=atol)
    assert_allclose(refstate.nuclear_dipole,
                    multipoles["nuclear_1"], atol=atol)
    if backend is not None and backend in ["pyscf", "veloxchem"]:
        gauge_origins = ["origin", "mass_center", "charge_center"]
        for g_origin in gauge_origins:
            # adjust tolerance criteria for mass_center, because of different iso-
            # tropic mass averages.
            if g_origin == "mass_center":
                atol_nuc_quad = 2e-4
            else:
                atol_nuc_quad = atol
            assert_allclose(refstate.nuclear_quadrupole(g_origin),
                            multipoles[f"nuclear_2_{g_origin}"],
                            atol=atol_nuc_quad)

    if "electric_dipole" in refstate.operators.available \
            and "elec_1" in multipoles:
        refstate2 = adcc.ReferenceState(data)
        assert_allclose(
            refstate.dipole_moment, refstate2.dipole_moment, atol=atol
        )

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

    if compare_eri == "abs":
        decimal = 7
        if refstate.backend == "veloxchem":
            decimal = 6
        for ss in reference["eri"].keys():
            assert_almost_equal(np.abs(refstate.eri(ss).to_ndarray()),
                                np.abs(reference["eri"][ss]), decimal=decimal)
    elif compare_eri == "value":
        for ss in reference["eri"].keys():
            assert_allclose(refstate.eri(ss).to_ndarray(),
                            reference["eri"][ss], atol=atol)


test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz", "ch2nh2_sto3g"
)
cases = [(case.file_name, c) for case in test_cases for c in case.cases]


@pytest.mark.parametrize("system,case", cases)
class TestReferenceStateReferenceData:
    def test_hfimport(self, system: str, case: str):
        data = testdata_cache._load_hfdata(system)
        reference = testdata_cache.hfimport(system, case)
        compare_refstate_with_reference(
            system=system, case=case, data=data, reference=reference
        )
