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
import unittest
import numpy as np
import adcc
import adcc.backends

from ..misc import expand_test_templates
from .testing import eri_asymm_construction_test, operator_import_test
from .eri_build_helper import eri_permutations

from numpy.testing import assert_almost_equal

from adcc.backends import have_backend
from adcc.testdata import geometry

import pytest

if have_backend("pyscf"):
    from pyscf import gto, scf

basissets = ["sto3g", "ccpvdz"]


@expand_test_templates(basissets)
@pytest.mark.skipif(not have_backend("pyscf"), reason="pyscf not found.")
class TestPyscf(unittest.TestCase):
    def run_core_hole(self, mol):
        # First normal run
        mf = scf.UHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()

        # make core hole
        mo0 = tuple(c.copy() for c in mf.mo_coeff)
        occ0 = tuple(o.copy() for o in mf.mo_occ)
        occ0[0][0] = 0.0
        dm0 = mf.make_rdm1(mo0, occ0)

        # Run second SCF with MOM
        chole = scf.UHF(mol)
        scf.addons.mom_occ_(chole, mo0, occ0)
        chole.conv_tol = 1e-12
        chole.kernel(dm0)
        return chole

    def base_test(self, scfres):
        hfdata = adcc.backends.import_scf_results(scfres)
        assert hfdata.backend == "pyscf"

        n_orbs_alpha = hfdata.n_orbs_alpha
        fock_bb = scfres.get_fock()
        if hfdata.restricted:
            assert hfdata.spin_multiplicity != 0
            assert hfdata.n_alpha >= hfdata.n_beta

            # Check SCF type fits
            assert isinstance(scfres, (scf.rhf.RHF, scf.rohf.ROHF))
            assert not isinstance(scfres, scf.uhf.UHF)

            assert hfdata.n_orbs_alpha == hfdata.n_orbs_beta
            assert np.all(hfdata.orben_f[:n_orbs_alpha]
                          == hfdata.orben_f[n_orbs_alpha:])

            mo_occ = (scfres.mo_occ / 2, scfres.mo_occ / 2)
            mo_energy = (scfres.mo_energy, scfres.mo_energy)
            mo_coeff = (scfres.mo_coeff, scfres.mo_coeff)
            fock_bb = (fock_bb, fock_bb)
        else:
            assert hfdata.spin_multiplicity == 0

            # Check SCF type fits
            assert isinstance(scfres, scf.uhf.UHF)

            mo_occ = scfres.mo_occ
            mo_energy = scfres.mo_energy
            mo_coeff = scfres.mo_coeff

        # Check n_alpha and n_beta
        assert hfdata.n_alpha == np.sum(mo_occ[0] > 0)
        assert hfdata.n_beta == np.sum(mo_occ[1] > 0)

        # Check the lowest n_alpha / n_beta orbitals are occupied
        occ_a = [mo_occ[0][mo_energy[0] == ene]
                 for ene in hfdata.orben_f[:hfdata.n_alpha]]
        occ_b = [mo_occ[1][mo_energy[1] == ene]
                 for ene in hfdata.orben_f[n_orbs_alpha:
                                           n_orbs_alpha + hfdata.n_beta]]
        assert np.all(np.asarray(occ_a) > 0)
        assert np.all(np.asarray(occ_b) > 0)

        # TODO: Implement full tests for UHF once the modern interface
        # is extended
        if not hfdata.restricted:
            return

        # occupation_f
        assert_almost_equal(hfdata.occupation_f,
                            np.hstack((mo_occ[0], mo_occ[1])))

        # orben_f
        assert_almost_equal(hfdata.orben_f,
                            np.hstack((mo_energy[0], mo_energy[1])))

        # orbcoeff_fb
        assert_almost_equal(hfdata.orbcoeff_fb, np.transpose(np.hstack((
            mo_coeff[0], mo_coeff[1]
        ))))

        # fock_ff
        fock = tuple(mo_coeff[i].transpose().conj() @ fock_bb[i] @ mo_coeff[i]
                     for i in range(2))
        fullfock_ff = np.zeros((hfdata.n_orbs, hfdata.n_orbs))
        fullfock_ff[:n_orbs_alpha, :n_orbs_alpha] = fock[0]
        fullfock_ff[n_orbs_alpha:, n_orbs_alpha:] = fock[1]
        assert_almost_equal(hfdata.fock_ff, fullfock_ff)

        # test symmetry of the ERI tensor
        allowed_permutations = [
            p.transposition for p in eri_permutations["chem"]
        ]
        eri = np.empty((hfdata.n_orbs, hfdata.n_orbs,
                        hfdata.n_orbs, hfdata.n_orbs))
        sfull = slice(hfdata.n_orbs)
        hfdata.fill_eri_ffff((sfull, sfull, sfull, sfull), eri)
        for perm in allowed_permutations:
            eri_perm = np.transpose(eri, perm)
            assert_almost_equal(eri_perm, eri)

    def template_rhf_h2o(self, basis):
        mf = adcc.backends.run_hf("pyscf", geometry.xyz["h2o"], basis)
        self.base_test(mf)

        # Test ERI
        eri_asymm_construction_test(mf)
        eri_asymm_construction_test(mf, core_orbitals=1)

        # Test dipole
        ao_dip = mf.mol.intor_symmetric('int1e_r', comp=3)
        operator_import_test(mf, list(ao_dip))

    def test_h2o_sto3g_core_hole(self):
        mol = gto.M(
            atom=geometry.xyz["h2o"],
            basis='sto-3g',
            unit="Bohr",
            # needed to disable commandline argument parsing in pyscf
            parse_arg=False,
        )
        mf = self.run_core_hole(mol)
        self.base_test(mf)
