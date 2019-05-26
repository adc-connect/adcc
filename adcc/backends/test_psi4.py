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
import unittest
import numpy as np

from ..misc import expand_test_templates
from .eri_construction_test import eri_asymm_construction_test

from numpy.testing import assert_almost_equal

import pytest
import adcc
import adcc.backends

from adcc.backends import have_backend
from adcc.testdata import geometry

basissets = ["sto3g", "ccpvdz"]


@expand_test_templates(basissets)
@pytest.mark.skipif(not have_backend("psi4"), reason="psi4 not found.")
class TestPsi4(unittest.TestCase):

    def base_test(self, wfn):
        hfdata = adcc.backends.import_scf_results(wfn)
        assert hfdata.backend == "psi4"

        # only RHF support until now
        assert hfdata.restricted

        n_orbs = 2 * wfn.nmo()

        assert hfdata.spin_multiplicity != 0
        assert hfdata.n_alpha == hfdata.n_beta

        assert hfdata.n_orbs_alpha == hfdata.n_orbs_beta
        assert np.all(hfdata.orben_f[:hfdata.n_orbs_alpha]
                      == hfdata.orben_f[hfdata.n_orbs_alpha:])

        # Check n_alpha and n_beta
        assert hfdata.n_alpha == wfn.nalpha()
        assert hfdata.n_beta == wfn.nbeta()

        assert hfdata.energy_scf == wfn.energy()
        assert hfdata.spin_multiplicity == wfn.molecule().multiplicity()

        # occupation_f
        assert_almost_equal(hfdata.occupation_f, np.hstack((
            np.asarray(wfn.occupation_a()), np.asarray(wfn.occupation_b())
        )))

        # orben_f
        assert_almost_equal(hfdata.orben_f,
                            np.hstack((wfn.epsilon_a(), wfn.epsilon_b())))
        # orbcoeff_fb
        assert_almost_equal(hfdata.orbcoeff_fb, np.transpose(np.hstack((
            np.asarray(wfn.Ca()),
            np.asarray(wfn.Cb()))
        )))

        # Fock matrix fock_ff
        fock_alpha_bb = np.asarray(wfn.Fa())
        fock_beta_bb = np.asarray(wfn.Fb())
        assert_almost_equal(fock_alpha_bb, fock_beta_bb)

        fock_ff = np.zeros((n_orbs, n_orbs))
        fock_alpha = np.einsum('ui,vj,uv', np.asarray(wfn.Ca()),
                               np.asarray(wfn.Ca()), fock_alpha_bb)
        fock_ff[:hfdata.n_orbs_alpha, :hfdata.n_orbs_alpha] = fock_alpha
        fock_beta = np.einsum('ui,vj,uv', np.asarray(wfn.Cb()),
                              np.asarray(wfn.Cb()), fock_beta_bb)
        fock_ff[hfdata.n_orbs_alpha:, hfdata.n_orbs_alpha:] = fock_beta
        assert_almost_equal(hfdata.fock_ff, fock_ff)

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
            assert_almost_equal(eri_perm, eri)

    def template_rhf_h2o(self, basis):
        wfn = adcc.backends.run_hf("psi4", geometry.xyz["h2o"], basis)
        self.base_test(wfn)
        eri_asymm_construction_test(wfn)
        eri_asymm_construction_test(wfn, core_orbitals=1)
