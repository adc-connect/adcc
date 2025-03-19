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
from numpy.testing import assert_almost_equal, assert_array_equal

import adcc
import adcc.backends
from adcc.backends import have_backend

from .testing import (eri_asymm_construction_test, eri_chem_permutations,
                      operator_import_from_ao_test)
from .. import testcases

if have_backend("veloxchem"):
    import veloxchem as vlx
    from veloxchem.veloxchemlib import (AngularMomentumIntegralsDriver,
                                        LinearMomentumIntegralsDriver)


h2o = testcases.get_by_filename("h2o_sto3g", "h2o_ccpvdz")
ch2nh2 = testcases.get(n_expected_cases=2, name="ch2nh2")


@pytest.mark.skipif(
    not have_backend("veloxchem"), reason="Veloxchem not found."
)
class TestVeloxchem:
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
            assert hfdata.n_alpha >= hfdata.n_beta

            # Check SCF type fits
            assert isinstance(scfdrv, (vlx. ScfUnrestrictedDriver))
            n_mo = scfdrv.mol_orbs.number_mos()
            n_alpha = np.sum(
                scfdrv.task.molecule.get_aufbau_alpha_occupation(n_mo) > 0)
            n_beta = np.sum(
                scfdrv.task.molecule.get_aufbau_beta_occupation(n_mo) > 0)
            mo_energy = (scfdrv.mol_orbs.ea_to_numpy(),
                         scfdrv.mol_orbs.eb_to_numpy())

        # Check n_alpha and n_beta
        assert hfdata.n_alpha == n_alpha
        assert hfdata.n_beta == n_beta
        assert_array_equal(mo_energy[0], hfdata.orben_f[:n_orbs_alpha])
        assert_array_equal(mo_energy[1], hfdata.orben_f[n_orbs_alpha:])

        # Check the lowest n_alpha / n_beta orbitals are occupied
        assert_array_equal(np.sort(mo_energy[0]), mo_energy[0])
        assert_array_equal(np.sort(mo_energy[1]), mo_energy[1])

        mo_coeff_a = scfdrv.mol_orbs.alpha_to_numpy()
        mo_coeff_b = scfdrv.mol_orbs.beta_to_numpy()
        mo_coeff = (mo_coeff_a, mo_coeff_b)

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
        eri = np.empty((hfdata.n_orbs, hfdata.n_orbs,
                        hfdata.n_orbs, hfdata.n_orbs))
        sfull = slice(hfdata.n_orbs)
        hfdata.fill_eri_ffff((sfull, sfull, sfull, sfull), eri)
        for perm in eri_chem_permutations:
            eri_perm = np.transpose(eri, perm)
            assert_almost_equal(eri_perm, eri)

    def operators_test(self, scfdrv):
        from adcc.backends.veloxchem import _transform_gauge_origin_to_xyz
        gauge_origins = ["origin", "mass_center", "charge_center"]

        # Test dipole
        dipole_drv = vlx.ElectricDipoleIntegralsDriver(scfdrv.task.mpi_comm)
        dipole_drv.origin = tuple(np.zeros(3))
        dipole_mats = dipole_drv.compute(scfdrv.task.molecule,
                                         scfdrv.task.ao_basis)
        integrals = (-1.0 * dipole_mats.x_to_numpy(),
                     -1.0 * dipole_mats.y_to_numpy(),
                     -1.0 * dipole_mats.z_to_numpy())
        operator_import_from_ao_test(scfdrv, integrals,
                                     operator="electric_dipole")

        # Test electric dipole velocity
        linmom_drv = LinearMomentumIntegralsDriver(scfdrv.task.mpi_comm)
        linmom_mats = linmom_drv.compute(scfdrv.task.molecule,
                                         scfdrv.task.ao_basis)
        integrals = (-1.0 * linmom_mats.x_to_numpy(),
                     -1.0 * linmom_mats.y_to_numpy(),
                     -1.0 * linmom_mats.z_to_numpy())
        operator_import_from_ao_test(scfdrv, integrals,
                                     operator="electric_dipole_velocity")

        for gauge_origin in gauge_origins:
            gauge_origin = _transform_gauge_origin_to_xyz(scfdrv, gauge_origin)

            # Test magnetic dipole
            angmom_drv = AngularMomentumIntegralsDriver(scfdrv.task.mpi_comm)
            angmom_drv.origin = tuple(gauge_origin)
            angmom_mats = angmom_drv.compute(scfdrv.task.molecule,
                                             scfdrv.task.ao_basis)
            integrals = (
                0.5 * angmom_mats.x_to_numpy(), 0.5 * angmom_mats.y_to_numpy(),
                0.5 * angmom_mats.z_to_numpy()
            )
            operator_import_from_ao_test(scfdrv, integrals,
                                         "magnetic_dipole", gauge_origin)

    @pytest.mark.parametrize("system", h2o, ids=[case.file_name for case in h2o])
    def test_rhf(self, system: testcases.TestCase):
        scfdrv = adcc.backends.run_hf("veloxchem", system.xyz, system.basis)
        self.base_test(scfdrv)
        self.operators_test(scfdrv)
        # Test ERI
        eri_asymm_construction_test(scfdrv)
        eri_asymm_construction_test(scfdrv, core_orbitals=1)

    @pytest.mark.parametrize("system", ch2nh2,
                             ids=[case.file_name for case in ch2nh2])
    def test_uhf(self, system: testcases.TestCase):
        scfdrv = adcc.backends.run_hf("veloxchem", system.xyz, system.basis,
                                      charge=system.charge,
                                      multiplicity=system.multiplicity)
        self.base_test(scfdrv)
        self.operators_test(scfdrv)
        # Test ERI
        eri_asymm_construction_test(scfdrv)
        eri_asymm_construction_test(scfdrv, core_orbitals=1)
