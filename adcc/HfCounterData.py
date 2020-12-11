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
import numpy as np

from libadcc import HartreeFockProvider

from .DataHfProvider import DataOperatorIntegralProvider


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
        self.operator_integral_provider = DataOperatorIntegralProvider()

        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()

    def mul(self, n):
        return np.power(self.__mul, n)

    def get_backend(self):
        return "counterdata"

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
                if block == "bbbb":    # bbbb -> aaaa
                    block = "aaaa"
                elif block == "baba":  # baba -> abab
                    block = "abab"
                elif block == "abba":  # abba -> -baba -> -abab
                    block = "abab"
                    fac *= -1
                elif block == "baab":  # baab -> abba -> -baba -> -abab
                    block = "abab"
                    fac *= -1
                assert block in ("aaaa", "abab")

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
