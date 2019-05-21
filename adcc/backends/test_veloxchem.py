#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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


import pytest
import adcc
import unittest
import adcc.backends
from adcc.backends import have_backend
import numpy as np
from numpy.testing import assert_array_equal

from .eri_construction_test import eri_asymm_construction_test

if have_backend("veloxchem"):
    import veloxchem as vlx


@pytest.mark.skipif(
    not have_backend("veloxchem"), reason="Veloxchem not found."
)
class TestVeloxchem(unittest.TestCase):
    def base_test(self, scfdrv):
        hfdata = adcc.backends.import_scf_results(scfdrv)
        assert hfdata.backend == "veloxchem"

        n_orbs_alpha = hfdata.n_orbs_alpha
        if hfdata.restricted:
            assert hfdata.spin_multiplicity != 0
            assert hfdata.n_alpha >= hfdata.n_beta

            # Check SCF type fits
            assert isinstance(scfdrv, (vlx. ScfRestrictedDriver))

            assert hfdata.n_orbs_alpha == hfdata.n_orbs_beta
            assert_array_equal(hfdata.orben_f[:n_orbs_alpha],
                               hfdata.orben_f[n_orbs_alpha:])

            n_alpha = int(scfdrv.task.molecule.number_of_electrons() / 2)
            n_beta = n_alpha
            mo_energy = (scfdrv.mol_orbs.ea_to_numpy(),
                         scfdrv.mol_orbs.ea_to_numpy())
        else:
            raise NotImplementedError()

        # Check n_alpha and n_beta
        assert hfdata.n_alpha == n_alpha
        assert hfdata.n_beta == n_beta
        assert_array_equal(mo_energy[0], hfdata.orben_f[:n_orbs_alpha])
        assert_array_equal(mo_energy[1], hfdata.orben_f[n_orbs_alpha:])

        # Check the lowest n_alpha / n_beta orbitals are occupied
        assert_array_equal(np.sort(mo_energy[0]), mo_energy[0])
        assert_array_equal(np.sort(mo_energy[1]), mo_energy[1])

        # test symmetry of the ERI tensor
        ii, jj, kk, ll = 0, 1, 2, 3
        allowed_permutations = [
            (kk, ll, ii, jj),
            (jj, ii, ll, kk),
            (ll, kk, jj, ii),
            (jj, ii, kk, ll),
            (jj, ii, ll, kk),
        ]
        eri = np.empty((hfdata.n_orbs, hfdata.n_orbs,
                        hfdata.n_orbs, hfdata.n_orbs))
        sfull = slice(hfdata.n_orbs)
        hfdata.fill_eri_ffff((sfull, sfull, sfull, sfull), eri)
        for perm in allowed_permutations:
            eri_perm = np.transpose(eri, perm)
            np.testing.assert_almost_equal(eri_perm, eri)

        # TODO Many more tests
        #      Compare against reference data

        # TODO checks for values of
        # orben_f
        # fock_ff
        # orbcoeff_fb
        # Perhaps by transforming the fock matrix of the scfdrv object or so

    def test_water_sto3g_rhf(self):
        water_xyz = """
        O 0 0 0
        H 0 0 1.795239827225189
        H 1.693194615993441 0 -0.599043184453037
        """
        scfdrv = adcc.backends.run_hf(
            "veloxchem", xyz=water_xyz, basis="cc-pvdz"
        )
        eri_asymm_construction_test(scfdrv)
        eri_asymm_construction_test(scfdrv, core_orbitals=1)
        # self.base_test(scfdrv)
