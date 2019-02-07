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
import numpy as np

try:
    from pyscf import gto, scf
    _pyscf = True
except ImportError:
    _pyscf = False


@pytest.mark.skipif(not _pyscf, reason="pyscf not found.")
class TestPyscf(unittest.TestCase):
    def run_hf(self, mol):
        mf = scf.HF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        return mf

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
        if hfdata.restricted:
            assert hfdata.spin_multiplicity != 0
            assert hfdata.n_alpha >= hfdata.n_beta

            # Check SCF type fits
            assert isinstance(scfres, (scf.rhf.RHF, scf.rohf.ROHF))
            assert not isinstance(scfres, scf.uhf.UHF)

            assert hfdata.n_orbs_alpha == hfdata.n_orbs_beta
            assert np.all(hfdata.orben_f[:n_orbs_alpha] ==
                          hfdata.orben_f[n_orbs_alpha:])

            mo_occ = (scfres.mo_occ, scfres.mo_occ)
            mo_energy = (scfres.mo_energy, scfres.mo_energy)
        else:
            assert hfdata.spin_multiplicity == 0

            # Check SCF type fits
            assert isinstance(scfres, scf.uhf.UHF)

            mo_occ = scfres.mo_occ
            mo_energy = scfres.mo_energy

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

        # TODO checks for values of
        # orben_f
        # fock_ff
        # orbcoeff_fb
        # Perhaps by transforming the fock matrix of the scfres object or so

        # TODO Many more tests
        #      Compare against reference data

        # test symmetry of the ERI tensor
        ii, jj, kk, ll = 0, 1, 2, 3
        allowed_permutations = [
            (kk, ll, ii, jj),
            (jj, ii, ll, kk),
            (ll, kk, jj, ii),
            (jj, ii, kk, ll),
            (jj, ii, ll, kk),
        ]
        eri = hfdata._original_dict["eri_ffff"]
        for perm in allowed_permutations:
            eri_perm = np.transpose(eri, perm)
            np.testing.assert_almost_equal(eri_perm, eri)

    def test_water_sto3g_rhf(self):
        mol = gto.M(
            atom='O 0 0 0;'
                 'H 0 0 1.795239827225189;'
                 'H 1.693194615993441 0 -0.599043184453037',
            basis='sto-3g',
            unit="Bohr"
        )
        mf = self.run_hf(mol)
        self.base_test(mf)

    def test_water_sto3g_core_hole(self):
        mol = gto.M(
            atom='O 0 0 0;'
                 'H 0 0 1.795239827225189;'
                 'H 1.693194615993441 0 -0.599043184453037',
            basis='sto-3g',
            unit="Bohr"
        )
        mf = self.run_core_hole(mol)
        self.base_test(mf)
