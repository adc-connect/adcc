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

from numpy.testing import assert_almost_equal, assert_array_equal

from adcc.backends import have_backend
from adcc.testdata import geometry

import pytest

if have_backend("veloxchem"):
    import veloxchem as vlx

basissets = ["sto3g", "ccpvdz"]


@expand_test_templates(basissets)
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

        mo_coeff_a = scfdrv.mol_orbs.alpha_to_numpy()
        mo_coeff = (mo_coeff_a, mo_coeff_a)

        # occupation_f
        occu = np.zeros(2 * n_orbs_alpha)
        occu[:n_alpha] = occu[n_orbs_alpha:n_orbs_alpha + n_beta] = 1.
        assert_almost_equal(hfdata.occupation_f, occu)

        # orben_f
        assert_almost_equal(hfdata.orben_f,
                            np.hstack((mo_energy[0], mo_energy[1])))

        # orbcoeff_fb
        assert_almost_equal(hfdata.orbcoeff_fb, np.transpose(np.hstack((
            mo_coeff[0], mo_coeff[1])
        )))

        # fock_ff
        fock_bb = scfdrv.scf_tensors['F']

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
        scfdrv = adcc.backends.run_hf("veloxchem", geometry.xyz["h2o"], basis)
        self.base_test(scfdrv)

        # Test ERI
        eri_asymm_construction_test(scfdrv)
        eri_asymm_construction_test(scfdrv, core_orbitals=1)

        # Test dipole
        dipole_drv = vlx.ElectricDipoleIntegralsDriver(scfdrv.task.mpi_comm)
        dipole_mats = dipole_drv.compute(scfdrv.task.molecule,
                                         scfdrv.task.ao_basis)
        integrals = (dipole_mats.x_to_numpy(), dipole_mats.y_to_numpy(),
                     dipole_mats.z_to_numpy())
        operator_import_test(scfdrv, integrals)
