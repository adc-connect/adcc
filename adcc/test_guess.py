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
import itertools
import numpy as np

from .misc import expand_test_templates
from numpy.testing import assert_array_equal

import adcc
import adcc.guess

from pytest import approx
from adcc.testdata.cache import cache

# The methods to test
methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]


@expand_test_templates(methods)
class TestGuess(unittest.TestCase):
    def assert_symmetry_no_spin_change(self, matrix, guess, block,
                                       spin_block_symmetrisation):
        """
        Assert a guess vector has the correct symmetry if no spin change
        occurs during the excitation (i.e. no spin-flip)
        """
        # Extract useful quantities
        refstate = matrix.reference_state
        nCa = noa = refstate.n_orbs_alpha("o1")
        nCb = nob = refstate.n_orbs_beta("o1")
        nva = refstate.n_orbs_alpha("v1")
        nvb = refstate.n_orbs_beta("v1")
        if refstate.has_core_valence_separation:
            nCa = refstate.n_orbs_alpha("o2")
            nCb = refstate.n_orbs_beta("o2")

        fac = 0
        if spin_block_symmetrisation == "symmetric":
            fac = 1
        if spin_block_symmetrisation == "antisymmetric":
            fac = -1

        # Singles
        gts = guess["s"].to_ndarray()
        assert gts.shape == (nCa + nCb, nva + nvb)
        assert np.max(np.abs(gts[nCa:, :nva])) == 0
        assert np.max(np.abs(gts[:nCa, nva:])) == 0

        if refstate.restricted:
            assert_array_equal(gts[:nCa, :nva],
                               fac * gts[nCa:, nva:])

        # Doubles
        if "d" not in matrix.blocks:
            return

        gtd = guess["d"].to_ndarray()
        assert gtd.shape == (noa + nob, nCa + nCb, nva + nvb, nva + nvb)

        assert np.max(np.abs(gtd[:noa, :nCa, nva:, nva:])) == 0  # aa->bb
        assert np.max(np.abs(gtd[noa:, nCa:, :nva, :nva])) == 0  # bb->aa
        assert np.max(np.abs(gtd[:noa, :nCa, :nva, nva:])) == 0  # aa->ab
        assert np.max(np.abs(gtd[:noa, :nCa, nva:, :nva])) == 0  # aa->ba
        assert np.max(np.abs(gtd[:noa, nCa:, :nva, :nva])) == 0  # ab->aa
        assert np.max(np.abs(gtd[noa:, :nCa, :nva, :nva])) == 0  # ba->aa
        assert np.max(np.abs(gtd[noa:, nCa:, nva:, :nva])) == 0  # bb->ba
        assert np.max(np.abs(gtd[noa:, nCa:, :nva, nva:])) == 0  # bb->ab
        assert np.max(np.abs(gtd[noa:, :nCa, nva:, nva:])) == 0  # ba->bb
        assert np.max(np.abs(gtd[:noa, nCa:, nva:, nva:])) == 0  # ab->bb

        if refstate.restricted:
            assert_array_equal(gtd[:noa, :nCa, :nva, :nva],        # aa->aa
                               fac * gtd[noa:, nCa:, nva:, nva:])  # bb->bb
            assert_array_equal(gtd[:noa, nCa:, :nva, nva:],        # ab->ab
                               fac * gtd[noa:, :nCa, nva:, :nva])  # ba->ba
            assert_array_equal(gtd[:noa, nCa:, nva:, :nva],        # ab->ba
                               fac * gtd[noa:, :nCa, :nva, nva:])  # ba->ab

        assert_array_equal(gtd.transpose((0, 1, 3, 2)), -gtd)
        if not refstate.has_core_valence_separation:
            assert_array_equal(gtd.transpose((1, 0, 2, 3)), -gtd)

        if block == "s":
            assert np.max(np.abs(gtd[:noa, :nCa, :nva, :nva])) == 0
            assert np.max(np.abs(gtd[noa:, nCa:, nva:, nva:])) == 0
            assert np.max(np.abs(gtd[:noa, nCa:, :nva, nva:])) == 0
            assert np.max(np.abs(gtd[noa:, :nCa, nva:, :nva])) == 0
            assert np.max(np.abs(gtd[:noa, nCa:, nva:, :nva])) == 0
            assert np.max(np.abs(gtd[noa:, :nCa, :nva, nva:])) == 0
            has_aa = np.max(np.abs(gts[:nCa, :nva])) > 0
            has_bb = np.max(np.abs(gts[nCa:, nva:])) > 0
            assert has_aa or has_bb
        elif block == "d":
            assert np.max(np.abs(gts[:nCa, :nva])) == 0
            assert np.max(np.abs(gts[nCa:, nva:])) == 0
            has_aaaa = np.max(np.abs(gtd[:noa, :nCa, :nva, :nva])) > 0
            has_bbbb = np.max(np.abs(gtd[noa:, nCa:, nva:, nva:])) > 0
            has_abab = np.max(np.abs(gtd[:noa, nCa:, :nva, nva:])) > 0
            has_baba = np.max(np.abs(gtd[noa:, :nCa, nva:, :nva])) > 0
            has_abba = np.max(np.abs(gtd[:noa, nCa:, nva:, :nva])) > 0
            has_baab = np.max(np.abs(gtd[noa:, :nCa, :nva, nva:])) > 0
            assert has_aaaa or has_abab or has_abba or \
                has_bbbb or has_baba or has_baab

    def assert_orthonormal(self, guesses):
        for (i, gi) in enumerate(guesses):
            for (j, gj) in enumerate(guesses):
                ref = 1 if i == j else 0
                assert adcc.dot(gi, gj) == approx(ref)

    def assert_guess_values(self, matrix, block, guesses):
        """
        Assert that the guesses correspond to the smallest
        diagonal values.
        """

        # Extract useful quantities
        refstate = matrix.reference_state
        nCa = noa = refstate.n_orbs_alpha("o1")
        nva = refstate.n_orbs_alpha("v1")
        if refstate.has_core_valence_separation:
            nCa = refstate.n_orbs_alpha("o2")

        # Make a list of diagonal indices, ordered by the corresponding
        # diagonal values
        sidcs = None
        if block == "s":
            diagonal = matrix.diagonal("s").to_ndarray()

            # Build list of indices, which would sort the diagonal
            sidcs = np.dstack(np.unravel_index(np.argsort(diagonal.ravel()),
                                               diagonal.shape))
            assert sidcs.shape[0] == 1
            sidcs = [
                idx for idx in sidcs[0]
                if any((idx[0] >= nCa and idx[1] >= nva,
                        idx[0]  < nCa and idx[1]  < nva))  # noqa: E221
            ]
        elif block == "d":
            diagonal = matrix.diagonal("d").to_ndarray()

            # Build list of indices, which would sort the diagonal
            sidcs = np.dstack(np.unravel_index(np.argsort(diagonal.ravel()),
                                               diagonal.shape))

            assert sidcs.shape[0] == 1
            sidcs = [
                idx for idx in sidcs[0]
                if any((idx[0]  < noa and idx[1]  < nCa and idx[2]  < nva and idx[3]  < nva,   # noqa: E221,E501
                        idx[0] >= noa and idx[1] >= nCa and idx[2] >= nva and idx[3] >= nva,   # noqa: E221,E501
                        idx[0]  < noa and idx[1] >= nCa and idx[2]  < nva and idx[3] >= nva,   # noqa: E221,E501
                        idx[0] >= noa and idx[1]  < nCa and idx[2] >= nva and idx[3]  < nva,   # noqa: E221,E501
                        idx[0]  < noa and idx[1] >= nCa and idx[2] >= nva and idx[3]  < nva,   # noqa: E221,E501
                        idx[0] >= noa and idx[1]  < nCa and idx[2]  < nva and idx[3] >= nva))  # noqa: E221,E501
            ]
            sidcs = [idx for idx in sidcs if idx[2] != idx[3]]
            if not refstate.has_core_valence_separation:
                sidcs = [idx for idx in sidcs if idx[0] != idx[1]]

        # Group the indices by corresponding diagonal value
        def grouping(x):
            return np.round(diagonal[tuple(x)], decimals=12)
        gidcs = [[tuple(gitem) for gitem in group]
                 for key, group in itertools.groupby(sidcs, grouping)]
        igroup = 0  # The current diagonal value group we are in
        for (i, guess) in enumerate(guesses):
            # Extract indices of non-zero elements
            nonzeros = np.dstack(np.where(guess[block].to_ndarray() != 0))
            assert nonzeros.shape[0] == 1
            nonzeros = [tuple(nzitem) for nzitem in nonzeros[0]]
            if i > 0 and igroup + 1 < len(gidcs):
                if nonzeros[0] in gidcs[igroup + 1]:
                    igroup += 1
            for nz in nonzeros:
                assert nz in gidcs[igroup]

    def base_test(self, case, method, block, max_guesses=10):
        if adcc.AdcMethod(method).is_core_valence_separated:
            prelim = cache.prelim_cvs[case]
        else:
            prelim = cache.prelim[case]
        matrix = adcc.AdcMatrix(method, prelim.ground_state)

        symmetrisations = ["none"]
        if matrix.reference_state.restricted:
            symmetrisations = ["symmetric", "antisymmetric"]

        for symm in symmetrisations:
            for n_guesses in range(1, max_guesses + 1):
                guesses = adcc.guess.guesses_from_diagonal(
                    matrix, n_guesses, block=block, spin_change=0,
                    spin_block_symmetrisation=symm
                )
                assert len(guesses) == n_guesses
                for gs in guesses:
                    self.assert_symmetry_no_spin_change(matrix, gs, block, symm)
                self.assert_orthonormal(guesses)
                self.assert_guess_values(matrix, block, guesses)

    def template_singles_h2o(self, method):
        self.base_test("h2o_sto3g", method, "s")

    def template_singles_h2o_cvs(self, method):
        self.base_test("h2o_sto3g", "cvs-" + method, "s", max_guesses=2)

    def template_singles_cn(self, method):
        self.base_test("cn_sto3g", method, "s")

    def template_singles_cn_cvs(self, method):
        self.base_test("cn_sto3g", "cvs-" + method, "s", max_guesses=7)

    # TODO These tests fails because of adcman, because some delta-Fock-based
    #      approximation is used for the diagonal instead of the actual
    #      doubles diagonal which would be employed for ADC(2) and ADC(3)
    # def test_doubles_h2o_adc2(self):
    #     self.base_test("h2o_sto3g", "adc2", "d", max_guesses=3)
    #
    # def test_doubles_h2o_adc3(self):
    #     self.base_test("h2o_sto3g", "adc3", "d", max_guesses=3)
    #
    # def test_doubles_cn_adc2(self):
    #     self.base_test("cn_sto3g", "adc2", "d")
    #
    # def test_doubles_cn_adc3(self):
    #    self.base_test("cn_sto3g", "adc3", "d")
    #
    # TODO Perhaps could be templatified as well one the issues are resolved

    def test_doubles_h2o_cvs_adc2(self):
        self.base_test("h2o_sto3g", "cvs-adc2", "d", max_guesses=5)

    def test_reference_h2o_adc2x(self):
        pass  # TODO Compare against hard-coded reference singles and doubles guess vectors

    def test_reference_h2o_adc3(self):
        pass  # TODO Compare against hard-coded reference singles and doubles guess vectors

    def test_reference_cn_adc2x(self):
        pass  # TODO Compare against hard-coded reference singles and doubles guess vectors

    def test_reference_cn_adc3(self):
        pass  # TODO Compare against hard-coded reference singles and doubles guess vectors

    # TODO
    # Test spin-flip guesses


delattr(TestGuess, "test_singles_cn_cvs_adc3")
delattr(TestGuess, "test_singles_h2o_cvs_adc3")
