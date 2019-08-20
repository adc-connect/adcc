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
import unittest
import numpy as np

from .misc import expand_test_templates
from .DictHfProvider import DictOperatorIntegralProvider

from numpy.testing import assert_array_equal

import adcc

from libadcc import HartreeFockProvider


class HfCounterData(HartreeFockProvider):
    """
    This class provides valid dummy data for any set of parameters
    given upon construction. This data can be imported and verified
    easily afterwards. All data is based on pain indices.
    """
    def __init__(self, n_alpha, n_beta, n_bas, n_orbs_alpha, restricted):
        if n_alpha != n_beta and restricted:
            raise ValueError("ROHF should not be tested")

        self.__n_alpha = n_alpha
        self.__n_beta = n_beta
        self.__n_bas = n_bas
        self.__n_orbs_alpha = n_orbs_alpha
        self.__restricted = restricted
        self.__mul = 10
        self.operator_integral_provider = DictOperatorIntegralProvider()

        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()

    def mul(self, n):
        return np.power(self.__mul, n)

    def get_backend(self):
        return "counterdata"

    def get_n_alpha(self):
        return self.__n_alpha

    def get_n_beta(self):
        return self.__n_beta

    def get_conv_tol(self):
        return 1e-10

    def get_restricted(self):
        return self.__restricted

    def get_energy_scf(self):
        return -1.0

    def get_spin_multiplicity(self):
        return 1

    def get_n_orbs_alpha(self):
        return self.__n_orbs_alpha

    def get_n_orbs_beta(self):
        return self.__n_orbs_alpha

    def get_n_bas(self):
        return self.__n_bas

    def get_fa_range(self):
        return np.arange(1, self.__n_orbs_alpha + 1)

    def get_fb_range(self):
        if self.__restricted:
            return self.get_fa_range()
        else:
            return self.__n_orbs_alpha + self.get_fa_range()

    def get_f_range(self):
        return (self.get_fa_range(), self.get_fb_range())

    def get_b_range(self):
        return np.arange(1, self.__n_bas + 1)

    def fold_fock(self, range1, range2):
        """
        Build a reference fock matrix from two reference axes, where
        in each axis the alpha and the beta values are given as a tuple.
        """
        na1 = len(range1[0])  # Number of alphas for first axis
        na2 = len(range2[0])  # Number of alphas for second axis
        n1 = len(range1[0]) + len(range1[1])  # Total number for first axis
        n2 = len(range2[0]) + len(range2[1])  # Total number for first axis
        res = np.zeros((n1, n2))
        for (i, vi) in enumerate(np.hstack(range1)):
            for (j, vj) in enumerate(np.hstack(range2)):
                if i < na1 and j >= na2:
                    continue  # ab block is zero
                if i >= na1 and j < na2:
                    continue  # ba block is zero
                if vi < vj:
                    res[i, j] = self.mul(1) * vi + vj
                else:
                    res[i, j] = vi + self.mul(1) * vj
        return res

    def fill_orbcoeff_fb(self, out):
        out[:] = (np.hstack(self.get_f_range())[:, None] * self.mul(1)
                  + self.get_b_range()[None, :])

    def fill_occupation_f(self, out):
        n_oa = self.__n_orbs_alpha
        out[:] = np.zeros(2 * n_oa)
        out[:self.__n_alpha] = 1.
        out[n_oa:n_oa + self.__n_beta] = 1.

    def fill_orben_f(self, out):
        out[:] = np.hstack(self.get_f_range())

    def fill_fock_ff(self, slices, out):
        out[:] = self.fold_fock(self.get_f_range(), self.get_f_range())[slices]

    def fill_eri_ffff(self, slices, out):
        raise NotImplementedError("eri_ffff")

    def has_eri_phys_asym_ffff(self):
        return True

    def fold_eri(self, range1, range2, range3, range4):
        # Notice: This function does not yet work fully
        #         and is super slow, too.
        # The idea is to generate a nice ERI tensor. The symmetry-checked
        # import does not yet complain, but the resulting data does not agree

        na1 = len(range1[0])  # Number of alphas for first axis
        na2 = len(range2[0])  # Number of alphas for second axis
        na3 = len(range3[0])
        na4 = len(range4[0])
        n_alphas = [na1, na2, na3, na4]

        # Full ranges per axis
        rfulls = (np.hstack(range1), np.hstack(range2),
                  np.hstack(range3), np.hstack(range4))

        # Total number of orbitals per axis
        n_orbs = tuple(len(rfulls[i]) for i in range(4))

        # Function to compute one eri value. Takes a block and 4 indices
        # and reads rfulls and n_alphas
        def compute_eri_value(block, i1, i2, i3, i4):
            idx = np.array([i1, i2, i3, i4])
            val = np.array([rfulls[i][idx[i]] for i in range(4)])

            # map equivalent spin blocks onto another
            fac = 1.0
            if self.__restricted:
                if block == "bbbb":  # bbbb -> aaaa
                    block = "aaaa"
                if block == "baba":  # baba -> abab
                    block = "abab"
                if block == "abba":  # abba -> -baba -> -abab
                    block = "abab"
                    fac *= -1
                if block == "baab":  # baab -> abba -> -baba -> -abab
                    block = "abab"
                    fac *= -1

            # Zero elements by index permutation
            if val[0] == val[1]:
                return 0
            if val[2] == val[3]:
                return 0

            # Canonicalise the values wrt. index permutations
            if val[0] > val[1]:  # deal with ijkl = -jikl
                val[1], val[0] = val[0], val[1]
                fac *= -1
            if val[2] > val[3]:  # deal with ijkl = -ijlk
                val[2], val[3] = val[3], val[2]
                fac *= -1
            if val[0] > val[2]:  # deal with ijkl = klij
                val[0], val[1], val[2], val[3] = \
                    val[2], val[3], val[0], val[1]

            return fac * (
                + self.mul(3) * val[0] + self.mul(2) * val[1]
                + self.mul(1) * val[2] + self.mul(0) * val[3]
            )

        # Run only over blocks, which are non-zero by spin
        res = np.zeros(n_orbs)
        for block in ["aaaa", "abab", "abba", "bbbb", "baba", "baab"]:
            # Build the ranges and iterate over them:
            ranges = []
            for i, s in enumerate(block):
                if s == "a":
                    ranges.append(range(n_alphas[i]))
                else:
                    ranges.append(range(n_alphas[i], n_orbs[i]))

            for i1 in ranges[0]:
                for i2 in ranges[1]:
                    for i3 in ranges[2]:
                        for i4 in ranges[3]:
                            res[i1, i2, i3, i4] = compute_eri_value(
                                block, i1, i2, i3, i4
                            )
        return res

    def fill_eri_phys_asym_ffff(self, slices, out):
        full = self.fold_eri(self.get_f_range(), self.get_f_range(),
                             self.get_f_range(), self.get_f_range())
        out[:] = full[slices]


@expand_test_templates(["restricted", "unrestricted"])
class TestReferenceStateCounterData(unittest.TestCase):
    def base_test(self, n_alpha, n_beta, n_bas, n_orbs_alpha, restricted,
                  check_symmetry=False, core_orbitals=[], frozen_core=[],
                  frozen_virtual=[]):
        if not isinstance(restricted, bool):
            restricted = (restricted == "restricted")
        data = HfCounterData(n_alpha, n_beta, n_bas, n_orbs_alpha, restricted)
        refstate = adcc.ReferenceState(data, core_orbitals, frozen_core,
                                       frozen_virtual,
                                       symmetry_check_on_import=check_symmetry,
                                       import_all_below_n_orbs=None)

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
    def template_0gen_small(self, restricted):
        self.base_test(n_alpha=3, n_beta=3, n_bas=8, n_orbs_alpha=8,
                       restricted=restricted, check_symmetry=False)
        #              # XXX check_symmetry=True fails because a
        #                write buffer overflow

    def template_0gen_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted)

    def template_0gen_large(self, restricted):
        self.base_test(n_alpha=21, n_beta=21, n_bas=60, n_orbs_alpha=60,
                       restricted=restricted)

    def template_cvs_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted, core_orbitals=([0, 1], [0, 1]))

    def test_cvs_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True, core_orbitals=([0, 1, 2], [0, 1, 2]))

    #
    # frozen-core
    #
    def template_fc_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted, frozen_core=([0, 1], [0, 1]))

    def test_fc_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True, frozen_core=([0, 1, 2], [0, 1, 2]))

    def template_fc_cvs_medium(self, restricted):
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
    def template_fv_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted,
                       frozen_virtual=([18, 19], [18, 19]))

    def test_fv_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True,
                       frozen_virtual=([57, 58, 59], [57, 58, 59]))

    def template_fv_cvs_medium(self, restricted):
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
    def template_fc_fv_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted, frozen_core=([0, 1], [0, 1]),
                       frozen_virtual=([18, 19], [18, 19]))

    def test_fc_fv_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True, frozen_core=([0, 1, 2], [0, 1, 2]),
                       frozen_virtual=([57, 58, 59], [57, 58, 59]))

    def template_fc_fv_cvs_medium(self, restricted):
        self.base_test(n_alpha=9, n_beta=9, n_bas=20, n_orbs_alpha=20,
                       restricted=restricted,
                       frozen_virtual=([18, 19], [18, 19]),
                       frozen_core=([0], [0]), core_orbitals=([1], [1]))

    def test_fc_fv_cvs_large_restricted(self):
        self.base_test(n_alpha=15, n_beta=15, n_bas=60, n_orbs_alpha=60,
                       restricted=True,
                       frozen_virtual=([57, 58, 59], [57, 58, 59]),
                       frozen_core=([0], [0]), core_orbitals=([1, 2], [1, 2]))
