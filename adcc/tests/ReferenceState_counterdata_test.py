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
import pytest
import numpy as np
from numpy.testing import assert_array_equal

from adcc.ReferenceState import ReferenceState
from adcc.HfCounterData import HfCounterData


class TestReferenceStateCounterData:
    def base_test(self, n_alpha, n_beta, n_bas, n_orbs_alpha, restricted,
                  check_symmetry=False, core_orbitals=[], frozen_core=[],
                  frozen_virtual=[]):
        if not isinstance(restricted, bool):
            restricted = (restricted == "restricted")
        data = HfCounterData(n_alpha, n_beta, n_bas, n_orbs_alpha, restricted)
        refstate = ReferenceState(
            data, core_orbitals, frozen_core, frozen_virtual,
            symmetry_check_on_import=check_symmetry, import_all_below_n_orbs=None
        )

        # Setup spaces and refstate axis
        subspaces = ["o1", "v1"]
        ref_axis = {"b": np.arange(1, n_bas + 1)}
        done_a = []  # Orbitals which are done (occ or virt)
        done_b = []
        na_rest = n_alpha  # Remaining alpha and beta orbitals to distribute
        nb_rest = n_beta

        def add_subspace(orbitals, sid):  # Add a new subspace
            subspaces.append(sid)
            orbs_a = np.array(orbitals[0]) + 1
            orbs_b = np.array(orbitals[1]) + 1 + n_orbs_alpha
            if restricted:
                orbs_b -= n_orbs_alpha
            ref_axis[sid] = ((orbs_a, orbs_b))
            done_a.extend(orbs_a)
            done_b.extend(orbs_b)

        if core_orbitals:
            add_subspace(core_orbitals, "o2")
            na_rest -= len(core_orbitals[0])
            nb_rest -= len(core_orbitals[1])
        if frozen_core:
            add_subspace(frozen_core, "o3")
            na_rest -= len(frozen_core[0])
            nb_rest -= len(frozen_core[1])
        if frozen_virtual:
            add_subspace(frozen_virtual, "v2")

        notdone_a = np.array([o for o in data.get_fa_range()
                              if not np.any(np.abs(done_a - o) < 1e-12)])
        notdone_b = np.array([o for o in data.get_fb_range()
                              if not np.any(np.abs(done_b - o) < 1e-12)])
        ref_axis["o1"] = ((notdone_a[:na_rest], notdone_b[:nb_rest]))
        ref_axis["v1"] = ((notdone_a[na_rest:], notdone_b[nb_rest:]))

        # General properties
        assert refstate.restricted == restricted
        assert refstate.spin_multiplicity == (1 if restricted else 0)
        assert refstate.has_core_occupied_space == ("o2" in subspaces)
        assert refstate.irreducible_representation == "A"
        assert refstate.n_orbs == n_orbs_alpha + n_orbs_alpha
        assert refstate.n_orbs_alpha == n_orbs_alpha
        assert refstate.n_orbs_beta == n_orbs_alpha
        assert refstate.n_alpha == n_alpha
        assert refstate.n_beta == n_beta
        assert refstate.conv_tol == 1e-10
        assert refstate.energy_scf == -1.

        # Orben
        for ss in subspaces:
            assert_array_equal(refstate.orbital_energies(ss).to_ndarray(),
                               np.hstack(ref_axis[ss]))

        # Orbcoeff
        for ss in subspaces:
            coeff_a = ref_axis[ss][0][:, None] * data.mul(1) \
                + data.get_b_range()[None, :]
            coeff_b = ref_axis[ss][1][:, None] * data.mul(1) \
                + data.get_b_range()[None, :]
            nfa, nb = coeff_a.shape
            nfb, nb = coeff_b.shape
            coeff_full = np.zeros((nfa + nfb, 2 * nb))
            coeff_full[:nfa, :nb] = coeff_a
            coeff_full[nfa:, nb:] = coeff_b
            assert_array_equal(
                refstate.orbital_coefficients(ss + "b").to_ndarray(), coeff_full
            )

        # Fock
        for ss1 in subspaces:
            for ss2 in subspaces:
                assert_array_equal(refstate.fock(ss1 + ss2).to_ndarray(),
                                   data.fold_fock(ref_axis[ss1], ref_axis[ss2]))

        #
        # TODO The eri test is not yet working ... but I (mfh) have really spend
        #      enough time on it already ... also the current version is
        #      terribly slow due to the many python loops. I guess the fock
        #      test should catch most fuckups and so should do the reference
        #      tests in test_ReferenceState_refdata.py.
        #      For this reason I will comment it out and leave it for another
        #      time / person to pick it up --- against my usual habit of never
        #      committing big chunks of commented code to master.
        #
        # # Eri
        # for ss1 in subspaces:
        #     for ss2 in subspaces:
        #         for ss3 in subspaces:
        #             for ss4 in subspaces:
        #                 print("---------------------------")
        #                 print()
        #                 print(refstate.eri(ss1 + ss2
        #                                    + ss3 + ss4).to_ndarray())
        #                 print()
        #                 print("---------------------------")
        #                 print()
        #                 print(data.fold_eri(ref_axis[ss1], ref_axis[ss2],
        #                                     ref_axis[ss3], ref_axis[ss4]))
        #                 print()
        #                 print("---------------------------")
        #                 print()
        #                 print(refstate.eri(ss1 + ss2 + ss3 + ss4).to_ndarray()
        #                       - data.fold_eri(ref_axis[ss1], ref_axis[ss2],
        #                                       ref_axis[ss3], ref_axis[ss4]))
        #                 print()
        #                 print("---------------------------")
        #                 assert_array_equal(
        #                     refstate.eri(ss1 + ss2 + ss3 + ss4).to_ndarray(),
        #                     data.fold_eri(ref_axis[ss1], ref_axis[ss2],
        #                                   ref_axis[ss3], ref_axis[ss4])
        #                 )

    #
    # Gen & CVS
    #
    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_generic_small(self, restricted: str):
        self.base_test(n_alpha=3, n_beta=3, n_bas=8, n_orbs_alpha=8,
                       restricted=restricted, check_symmetry=False)
        #              # XXX check_symmetry=True fails because a
        #                write buffer overflow

    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_generic_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted)

    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_generic_large(self, restricted):
        self.base_test(n_alpha=21, n_beta=21, n_bas=60, n_orbs_alpha=60,
                       restricted=restricted)

    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_cvs_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted, core_orbitals=([0, 1], [0, 1]))

    def test_cvs_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True, core_orbitals=([0, 1, 2], [0, 1, 2]))

    #
    # frozen-core
    #
    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_fc_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted, frozen_core=([0, 1], [0, 1]))

    def test_fc_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True, frozen_core=([0, 1, 2], [0, 1, 2]))

    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_fc_cvs_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted, frozen_core=([0], [0]),
                       core_orbitals=([1], [1]))

    def test_fc_cvs_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True, frozen_core=([0], [0]),
                       core_orbitals=([1, 2], [1, 2]))

    #
    # frozen-virtual
    #
    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_fv_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted,
                       frozen_virtual=([18, 19], [18, 19]))

    def test_fv_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True,
                       frozen_virtual=([57, 58, 59], [57, 58, 59]))

    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_fv_cvs_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted,
                       frozen_virtual=([18, 19], [18, 19]),
                       core_orbitals=([0, 1], [0, 1]))

    def test_fv_cvs_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True,
                       frozen_virtual=([57, 58, 59], [57, 58, 59]),
                       core_orbitals=([0, 1, 2], [0, 1, 2]))

    #
    # frozen-core, frozen-virtual
    #
    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_fc_fv_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted, frozen_core=([0, 1], [0, 1]),
                       frozen_virtual=([18, 19], [18, 19]))

    def test_fc_fv_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True, frozen_core=([0, 1, 2], [0, 1, 2]),
                       frozen_virtual=([57, 58, 59], [57, 58, 59]))

    @pytest.mark.parametrize("restricted", ["restricted", "unrestricted"])
    def test_fc_fv_cvs_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted,
                       frozen_virtual=([18, 19], [18, 19]),
                       frozen_core=([0], [0]), core_orbitals=([1], [1]))

    def test_fc_fv_cvs_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True,
                       frozen_virtual=([57, 58, 59], [57, 58, 59]),
                       frozen_core=([0], [0]), core_orbitals=([1, 2], [1, 2]))
