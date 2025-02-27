#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import numpy as np
from numpy.testing import assert_almost_equal

import adcc
import adcc.backends
from adcc.backends import have_backend

from .testing import eri_asymm_construction_test

from .. import testcases


h2o = testcases.get_by_filename("h2o_sto3g").pop()


@pytest.mark.parametrize("system", [h2o.file_name])
@pytest.mark.skipif(not have_backend("molsturm"), reason="molsturm not found.")
class TestMolsturm:
    def base_test(self, scfres):
        hfdata = adcc.backends.import_scf_results(scfres)
        assert hfdata.backend == "molsturm"

        n_orbs_alpha = scfres["n_orbs_alpha"]
        n_alpha = scfres["n_alpha"]
        n_beta = scfres["n_beta"]
        assert hfdata.energy_scf == scfres["energy_ground_state"]
        assert scfres["n_orbs_alpha"] == scfres["n_orbs_beta"]

        params = scfres["input_parameters"]
        coords = np.asarray(params["system"]["coords"])
        charges = np.asarray(params["system"]["atom_numbers"])
        assert hfdata.get_nuclear_multipole(0)[0] == int(np.sum(charges))
        assert_almost_equal(hfdata.get_nuclear_multipole(1),
                            np.einsum('i,ix->x', charges, coords))

        if scfres["restricted"]:
            assert hfdata.restricted
            assert hfdata.spin_multiplicity == 2 * (n_alpha - n_beta) + 1
            assert np.all(hfdata.orben_f[:n_orbs_alpha]
                          == hfdata.orben_f[n_orbs_alpha:])
        else:
            assert hfdata.spin_multiplicity == 0
            assert not hfdata.restricted

        occu = np.zeros(2 * scfres["n_orbs_alpha"])
        occu[:n_alpha] = occu[n_orbs_alpha:n_orbs_alpha + n_beta] = 1.
        assert_almost_equal(hfdata.occupation_f, occu)

        assert_almost_equal(hfdata.orben_f, scfres["orben_f"])
        assert_almost_equal(hfdata.orbcoeff_fb,
                            np.transpose(scfres["orbcoeff_bf"]))
        assert_almost_equal(hfdata.fock_ff, scfres["fock_ff"])

        eri = np.empty((hfdata.n_orbs, hfdata.n_orbs,
                        hfdata.n_orbs, hfdata.n_orbs))
        sfull = slice(hfdata.n_orbs)
        hfdata.fill_eri_ffff((sfull, sfull, sfull, sfull), eri)
        assert_almost_equal(eri, scfres["eri_ffff"])

    def test_rhf(self, system):
        system = testcases.get_by_filename(system).pop()

        scfres = adcc.backends.run_hf("molsturm", system.xyz, system.basis)
        self.base_test(scfres)
        eri_asymm_construction_test(scfres)
        eri_asymm_construction_test(scfres, core_orbitals=1)

    def test_uhf(self, system):
        system = testcases.get_by_filename(system).pop()

        scfres = adcc.backends.run_hf("molsturm", system.xyz, system.basis,
                                      multiplicity=3, conv_tol_grad=1e-6)
        self.base_test(scfres)
        eri_asymm_construction_test(scfres)
        eri_asymm_construction_test(scfres, core_orbitals=1)
