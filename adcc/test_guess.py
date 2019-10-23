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
import unittest
import itertools
import numpy as np
import adcc
import adcc.guess

from .misc import expand_test_templates
from numpy.testing import assert_array_equal
from adcc.testdata.cache import cache

from pytest import approx

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
        mospaces = matrix.mospaces
        nCa = noa = mospaces.n_orbs_alpha("o1")
        nCb = nob = mospaces.n_orbs_beta("o1")
        nva = mospaces.n_orbs_alpha("v1")
        nvb = mospaces.n_orbs_beta("v1")
        if mospaces.has_core_occupied_space:
            nCa = mospaces.n_orbs_alpha("o2")
            nCb = mospaces.n_orbs_beta("o2")

        fac = 1
        if spin_block_symmetrisation == "symmetric":
            fac = 1
        if spin_block_symmetrisation == "antisymmetric":
            fac = -1

        # Singles
        gts = guess["s"].to_ndarray()
        assert gts.shape == (nCa + nCb, nva + nvb)
        assert np.max(np.abs(gts[nCa:, :nva])) == 0
        assert np.max(np.abs(gts[:nCa, nva:])) == 0

        if matrix.reference_state.restricted:
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

        if matrix.reference_state.restricted:
            assert_array_equal(gtd[:noa, :nCa, :nva, :nva],        # aa->aa
                               fac * gtd[noa:, nCa:, nva:, nva:])  # bb->bb
            assert_array_equal(gtd[:noa, nCa:, :nva, nva:],        # ab->ab
                               fac * gtd[noa:, :nCa, nva:, :nva])  # ba->ba
            assert_array_equal(gtd[:noa, nCa:, nva:, :nva],        # ab->ba
                               fac * gtd[noa:, :nCa, :nva, nva:])  # ba->ab

        assert_array_equal(gtd.transpose((0, 1, 3, 2)), -gtd)
        if not matrix.is_core_valence_separated:
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

    def assert_symmetry_spin_flip(self, matrix, guess, block):
        """
        Assert a guess vector has the correct symmetry if we have a
        spin-change of -1 (i.e. spin-flip)
        """
        assert not matrix.is_core_valence_separated
        mospaces = matrix.mospaces
        nCa = noa = mospaces.n_orbs_alpha("o1")
        nCb = nob = mospaces.n_orbs_beta("o1")
        nva = mospaces.n_orbs_alpha("v1")
        nvb = mospaces.n_orbs_beta("v1")

        # Singles
        gts = guess["s"].to_ndarray()
        assert gts.shape == (nCa + nCb, nva + nvb)
        assert np.max(np.abs(gts[:nCa, :nva])) == 0  # a->a
        assert np.max(np.abs(gts[nCa:, :nva])) == 0  # b->a
        assert np.max(np.abs(gts[nCa:, nva:])) == 0  # b->b

        # Doubles
        if "d" not in matrix.blocks:
            return

        gtd = guess["d"].to_ndarray()
        assert gtd.shape == (noa + nob, nCa + nCb, nva + nvb, nva + nvb)
        assert np.max(np.abs(gtd[:noa, :nCa, :nva, :nva])) == 0  # aa->aa
        assert np.max(np.abs(gtd[:noa, :nCa, nva:, nva:])) == 0  # aa->bb
        assert np.max(np.abs(gtd[:noa, nCa:, :nva, :nva])) == 0  # ab->aa
        assert np.max(np.abs(gtd[:noa, nCa:, :nva, nva:])) == 0  # ab->ab
        assert np.max(np.abs(gtd[:noa, nCa:, nva:, :nva])) == 0  # ab->ba
        assert np.max(np.abs(gtd[noa:, :nCa, :nva, :nva])) == 0  # ba->aa
        assert np.max(np.abs(gtd[noa:, :nCa, :nva, nva:])) == 0  # ba->ab
        assert np.max(np.abs(gtd[noa:, :nCa, nva:, :nva])) == 0  # ba->ba
        assert np.max(np.abs(gtd[noa:, nCa:, :nva, :nva])) == 0  # bb->aa
        assert np.max(np.abs(gtd[noa:, nCa:, :nva, nva:])) == 0  # bb->ab
        assert np.max(np.abs(gtd[noa:, nCa:, nva:, :nva])) == 0  # bb->ba
        assert np.max(np.abs(gtd[noa:, nCa:, nva:, nva:])) == 0  # bb->bb

        assert_array_equal(gtd.transpose((0, 1, 3, 2)), -gtd)
        if not matrix.is_core_valence_separated:
            assert_array_equal(gtd.transpose((1, 0, 2, 3)), -gtd)

        if block == "s":
            assert np.max(np.abs(gtd[:noa, :nCa, :nva, nva:])) == 0  # aa->ab
            assert np.max(np.abs(gtd[:noa, :nCa, nva:, :nva])) == 0  # aa->ba
            assert np.max(np.abs(gtd[:noa, nCa:, nva:, nva:])) == 0  # ab->bb
            assert np.max(np.abs(gtd[noa:, :nCa, nva:, nva:])) == 0  # ba->bb
            assert np.max(np.abs(gts[:nCa, nva:])) > 0
        elif block == "d":
            assert np.max(np.abs(gts[:nCa, nva:])) == 0
            has_aaab = np.max(np.abs(gtd[:noa, :nCa, :nva, nva:])) > 0
            has_aaba = np.max(np.abs(gtd[:noa, :nCa, nva:, :nva])) > 0
            has_abbb = np.max(np.abs(gtd[:noa, nCa:, nva:, nva:])) > 0
            has_babb = np.max(np.abs(gtd[noa:, :nCa, nva:, nva:])) > 0
            assert has_aaab or has_aaba or has_abbb or has_babb

    def assert_orthonormal(self, guesses):
        for (i, gi) in enumerate(guesses):
            for (j, gj) in enumerate(guesses):
                ref = 1 if i == j else 0
                assert adcc.dot(gi, gj) == approx(ref)

    def assert_guess_values(self, matrix, block, guesses, spin_flip=False):
        """
        Assert that the guesses correspond to the smallest
        diagonal values.
        """
        # Extract useful quantities
        mospaces = matrix.mospaces
        nCa = noa = mospaces.n_orbs_alpha("o1")
        nva = mospaces.n_orbs_alpha("v1")
        if mospaces.has_core_occupied_space:
            nCa = mospaces.n_orbs_alpha("o2")

        # Make a list of diagonal indices, ordered by the corresponding
        # diagonal values
        sidcs = None
        if block == "s":
            diagonal = matrix.diagonal("s").to_ndarray()

            # Build list of indices, which would sort the diagonal
            sidcs = np.dstack(np.unravel_index(np.argsort(diagonal.ravel()),
                                               diagonal.shape))
            assert sidcs.shape[0] == 1
            if spin_flip:
                sidcs = [idx for idx in sidcs[0]
                         if idx[0] < nCa and idx[1] >= nva]
            else:
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
            if spin_flip:
                sidcs = [
                    idx for idx in sidcs[0]
                    if any((idx[0]  < noa and idx[1]  < nCa and idx[2]  < nva and idx[3] >= nva,   # noqa: E221,E501
                            idx[0]  < noa and idx[1]  < nCa and idx[2] >= nva and idx[3]  < nva,   # noqa: E221,E501
                            idx[0]  < noa and idx[1] >= nCa and idx[2] >= nva and idx[3] >= nva,   # noqa: E221,E501
                            idx[0] >= noa and idx[1]  < nCa and idx[2] >= nva and idx[3] >= nva))  # noqa: E221,E501
                ]
            else:
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
            if not matrix.is_core_valence_separated:
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

    def base_test_no_spin_change(self, case, method, block, max_guesses=10):
        if adcc.AdcMethod(method).is_core_valence_separated:
            ground_state = adcc.LazyMp(cache.refstate_cvs[case])
        else:
            ground_state = adcc.LazyMp(cache.refstate[case])
        matrix = adcc.AdcMatrix(method, ground_state)

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

    def base_test_spin_flip(self, case, method, block, max_guesses=10):
        ground_state = adcc.LazyMp(cache.refstate[case])
        matrix = adcc.AdcMatrix(method, ground_state)
        for n_guesses in range(1, max_guesses + 1):
            guesses = adcc.guess.guesses_from_diagonal(
                matrix, n_guesses, block=block, spin_change=-1,
                spin_block_symmetrisation="none"
            )
            assert len(guesses) == n_guesses
            for gs in guesses:
                self.assert_symmetry_spin_flip(matrix, gs, block)
            self.assert_orthonormal(guesses)
            self.assert_guess_values(matrix, block, guesses, spin_flip=True)

    def template_singles_h2o(self, method):
        self.base_test_no_spin_change("h2o_sto3g", method, "s")

    def template_singles_h2o_cvs(self, method):
        self.base_test_no_spin_change("h2o_sto3g", "cvs-" + method, "s",
                                      max_guesses=2)

    def template_singles_cn(self, method):
        self.base_test_no_spin_change("cn_sto3g", method, "s")

    def template_singles_cn_cvs(self, method):
        self.base_test_no_spin_change("cn_sto3g", "cvs-" + method, "s",
                                      max_guesses=7)

    def template_singles_hf3(self, method):
        self.base_test_spin_flip("hf3_631g", method, "s")

    # TODO These tests fails because of adcman, because some delta-Fock-based
    #      approximation is used for the diagonal instead of the actual
    #      doubles diagonal which would be employed for ADC(2) and ADC(3)
    # def test_doubles_h2o_adc2(self):
    #     self.base_test_no_spin_change("h2o_sto3g", "adc2", "d", max_guesses=3)
    #
    # def test_doubles_h2o_adc3(self):
    #     self.base_test_no_spin_change("h2o_sto3g", "adc3", "d", max_guesses=3)
    #
    # def test_doubles_cn_adc2(self):
    #     self.base_test_no_spin_change("cn_sto3g", "adc2", "d")
    #
    # def test_doubles_cn_adc3(self):
    #    self.base_test_no_spin_change("cn_sto3g", "adc3", "d")
    #
    # TODO Perhaps could be templatified as well one the issues are resolved

    def test_doubles_h2o_cvs_adc2(self):
        self.base_test_no_spin_change("h2o_sto3g", "cvs-adc2", "d",
                                      max_guesses=5)

    def test_doubles_hf3_adc2(self):
        self.base_test_spin_flip("hf3_631g", "adc2", "d")

    # TODO See above
    # def test_doubles_hf3_adc3(self):
    #     self.base_test_spin_flip("hf3_631g", "adc3", "d")

    #
    # Tests against reference values
    #
    def base_reference(self, matrix, ref):
        symmetrisations = ["none"]
        if matrix.reference_state.restricted:
            symmetrisations = ["symmetric", "antisymmetric"]

        for block in ["s", "d"]:
            for symm in symmetrisations:
                ref_sb = ref[(block, symm)]
                guesses = adcc.guess.guesses_from_diagonal(
                    matrix, len(ref_sb), block, spin_change=0,
                    spin_block_symmetrisation=symm
                )
                assert len(guesses) == len(ref_sb)
                for gs in guesses:
                    self.assert_symmetry_no_spin_change(matrix, gs, block, symm)
                self.assert_orthonormal(guesses)

                for (i, guess) in enumerate(guesses):
                    guess_b = guess[block].to_ndarray()
                    nonzeros = np.dstack(np.where(guess_b != 0))
                    assert nonzeros.shape[0] == 1
                    nonzeros = [tuple(nzitem) for nzitem in nonzeros[0]]
                    values = guess_b[guess_b != 0]
                    assert nonzeros == ref_sb[i][0]
                    assert_array_equal(values, np.array(ref_sb[i][1]))

    def test_reference_h2o_adc2x(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("adc2x", ground_state)
        self.base_reference(matrix, self.get_ref_h2o())

    def test_reference_h2o_adc3(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("adc3", ground_state)
        self.base_reference(matrix, self.get_ref_h2o())

    def test_reference_cn_adc2x(self):
        ground_state = adcc.LazyMp(cache.refstate["cn_sto3g"])
        matrix = adcc.AdcMatrix("adc2x", ground_state)
        self.base_reference(matrix, self.get_ref_cn())

    def test_reference_cn_adc3(self):
        ground_state = adcc.LazyMp(cache.refstate["cn_sto3g"])
        matrix = adcc.AdcMatrix("adc3", ground_state)
        self.base_reference(matrix, self.get_ref_cn())

    def get_ref_h2o(self):
        sq8 = 1 / np.sqrt(8)
        sq12 = 1 / np.sqrt(12)
        sq48 = 1 / np.sqrt(48)
        symm = [1 / np.sqrt(2), 1 / np.sqrt(2)]
        asymm = [1 / np.sqrt(2), -1 / np.sqrt(2)]
        return {
            ("s", "symmetric"): [
                ([(4, 0), (9, 2)], symm), ([(4, 1), (9, 3)], symm),
                ([(3, 0), (8, 2)], symm), ([(3, 1), (8, 3)], symm),
                ([(2, 0), (7, 2)], symm), ([(2, 1), (7, 3)], symm),
                ([(1, 0), (6, 2)], symm), ([(1, 1), (6, 3)], symm),
            ],
            ("s", "antisymmetric"): [
                # nonzeros          values
                ([(4, 0), (9, 2)], asymm), ([(4, 1), (9, 3)], asymm),
                ([(3, 0), (8, 2)], asymm), ([(3, 1), (8, 3)], asymm),
                ([(2, 0), (7, 2)], asymm), ([(2, 1), (7, 3)], asymm),
                ([(1, 0), (6, 2)], asymm), ([(1, 1), (6, 3)], asymm),
            ],
            ("d", "symmetric"): [
                ([(4, 9, 0, 2), (4, 9, 2, 0), (9, 4, 0, 2), (9, 4, 2, 0)],
                 [0.5, -0.5, -0.5, 0.5]),
                ([(3, 9, 0, 2), (3, 9, 2, 0), (4, 8, 0, 2), (4, 8, 2, 0),
                  (8, 4, 0, 2), (8, 4, 2, 0), (9, 3, 0, 2), (9, 3, 2, 0)],
                 [sq8, -sq8, sq8, -sq8, -sq8, sq8, -sq8, sq8]),
                ([(3, 8, 0, 2), (3, 8, 2, 0), (8, 3, 0, 2), (8, 3, 2, 0)],
                 [0.5, -0.5, -0.5, 0.5]),
                ([(4, 9, 0, 3), (4, 9, 1, 2), (4, 9, 2, 1), (4, 9, 3, 0),
                  (9, 4, 0, 3), (9, 4, 1, 2), (9, 4, 2, 1), (9, 4, 3, 0)],
                 [sq8, sq8, -sq8, -sq8, -sq8, -sq8, sq8, sq8]),
                ([(3, 4, 0, 1), (3, 4, 1, 0), (3, 9, 0, 3), (3, 9, 1, 2),
                  (3, 9, 2, 1), (3, 9, 3, 0), (4, 3, 0, 1), (4, 3, 1, 0),
                  (4, 8, 0, 3), (4, 8, 1, 2), (4, 8, 2, 1), (4, 8, 3, 0),
                  (8, 4, 0, 3), (8, 4, 1, 2), (8, 4, 2, 1), (8, 4, 3, 0),
                  (8, 9, 2, 3), (8, 9, 3, 2), (9, 3, 0, 3), (9, 3, 1, 2),
                  (9, 3, 2, 1), (9, 3, 3, 0), (9, 8, 2, 3), (9, 8, 3, 2)],
                 [sq12, -sq12, sq48, -sq48, sq48, -sq48, -sq12, sq12,
                  -sq48, sq48, -sq48, sq48, sq48, -sq48,  sq48, -sq48,
                  sq12, -sq12, -sq48, sq48, -sq48, sq48, -sq12, sq12]),
                ([(3, 9, 0, 3), (3, 9, 1, 2), (3, 9, 2, 1), (3, 9, 3, 0),
                  (4, 8, 0, 3), (4, 8, 1, 2), (4, 8, 2, 1), (4, 8, 3, 0),
                  (8, 4, 0, 3), (8, 4, 1, 2), (8, 4, 2, 1), (8, 4, 3, 0),
                  (9, 3, 0, 3), (9, 3, 1, 2), (9, 3, 2, 1), (9, 3, 3, 0)],
                 [0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25,
                  -0.25, -0.25, 0.25, 0.25, -0.25, -0.25, 0.25, 0.25]),
                ([(3, 8, 0, 3), (3, 8, 1, 2), (3, 8, 2, 1), (3, 8, 3, 0),
                  (8, 3, 0, 3), (8, 3, 1, 2), (8, 3, 2, 1), (8, 3, 3, 0)],
                 [sq8, sq8, -sq8, -sq8, -sq8, -sq8, sq8, sq8]),
                ([(4, 9, 1, 3), (4, 9, 3, 1), (9, 4, 1, 3), (9, 4, 3, 1)],
                 [0.5, -0.5, -0.5, 0.5]),
            ],
            ("d", "antisymmetric"): [
                ([(3, 9, 0, 2), (3, 9, 2, 0), (4, 8, 0, 2), (4, 8, 2, 0),
                  (8, 4, 0, 2), (8, 4, 2, 0), (9, 3, 0, 2), (9, 3, 2, 0)],
                 [sq8, -sq8, -sq8, sq8, sq8, -sq8, -sq8, sq8]),
                ([(4, 9, 0, 3), (4, 9, 1, 2), (4, 9, 2, 1), (4, 9, 3, 0),
                  (9, 4, 0, 3), (9, 4, 1, 2), (9, 4, 2, 1), (9, 4, 3, 0)],
                 [sq8, -sq8, sq8, -sq8, -sq8, sq8, -sq8, sq8]),
                ([(3, 4, 0, 1), (3, 4, 1, 0), (4, 3, 0, 1), (4, 3, 1, 0),
                  (8, 9, 2, 3), (8, 9, 3, 2), (9, 8, 2, 3), (9, 8, 3, 2)],
                 [sq8, -sq8, -sq8, sq8, -sq8, sq8, sq8, -sq8]),
                ([(3, 9, 0, 3), (3, 9, 1, 2), (3, 9, 2, 1), (3, 9, 3, 0),
                  (4, 8, 0, 3), (4, 8, 1, 2), (4, 8, 2, 1), (4, 8, 3, 0),
                  (8, 4, 0, 3), (8, 4, 1, 2), (8, 4, 2, 1), (8, 4, 3, 0),
                  (9, 3, 0, 3), (9, 3, 1, 2), (9, 3, 2, 1), (9, 3, 3, 0)],
                 [0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25,
                  0.25, 0.25, -0.25, -0.25, -0.25, -0.25, 0.25, 0.25]),
                ([(3, 9, 0, 3), (3, 9, 1, 2), (3, 9, 2, 1), (3, 9, 3, 0),
                  (4, 8, 0, 3), (4, 8, 1, 2), (4, 8, 2, 1), (4, 8, 3, 0),
                  (8, 4, 0, 3), (8, 4, 1, 2), (8, 4, 2, 1), (8, 4, 3, 0),
                  (9, 3, 0, 3), (9, 3, 1, 2), (9, 3, 2, 1), (9, 3, 3, 0)],
                 [0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25,
                  -0.25, 0.25, -0.25, 0.25, -0.25, 0.25, -0.25, 0.25]),
                ([(2, 9, 0, 2), (2, 9, 2, 0), (4, 7, 0, 2), (4, 7, 2, 0),
                  (7, 4, 0, 2), (7, 4, 2, 0), (9, 2, 0, 2), (9, 2, 2, 0)],
                 [sq8, -sq8, -sq8, sq8, sq8, -sq8, -sq8, sq8]),
                ([(3, 8, 0, 3), (3, 8, 1, 2), (3, 8, 2, 1), (3, 8, 3, 0),
                  (8, 3, 0, 3), (8, 3, 1, 2), (8, 3, 2, 1), (8, 3, 3, 0)],
                 [sq8, -sq8, sq8, -sq8, -sq8,  sq8, -sq8, sq8]),
                ([(2, 8, 0, 2), (2, 8, 2, 0), (3, 7, 0, 2), (3, 7, 2, 0),
                  (7, 3, 0, 2), (7, 3, 2, 0), (8, 2, 0, 2), (8, 2, 2, 0)],
                 [sq8, -sq8, -sq8, sq8, sq8, -sq8, -sq8, sq8]),
            ],
        }

    def get_ref_cn(self):
        sq8 = 1 / np.sqrt(8)
        return {
            ("s", "none"): [
                ([(11, 3)], [1.]), ([(12, 3)], [1.]),
                ([( 6, 0)], [1.]), ([( 6, 1)], [1.]),  # noqa: E201
                ([(10, 3)], [1.]), ([( 5, 0)], [1.]),  # noqa: E201
                ([( 4, 1)], [1.]), ([(11, 5)], [1.]),  # noqa: E201
            ],
            ("d", "none"): [
                ([(6, 11, 0, 3), (6, 11, 3, 0), (11, 6, 0, 3), (11, 6, 3, 0)],
                 [-0.5, 0.5, 0.5, -0.5]),
                ([(6, 12, 0, 3), (6, 12, 3, 0), (12, 6, 0, 3), (12, 6, 3, 0)],
                 [-0.5,  0.5, 0.5, -0.5]),
                ([(6, 11, 1, 3), (6, 11, 3, 1), (11, 6, 1, 3), (11, 6, 3, 1)],
                 [0.5, -0.5, -0.5, 0.5]),
                ([(6, 12, 1, 3), (6, 12, 3, 1), (12, 6, 1, 3), (12, 6, 3, 1)],
                 [0.5, -0.5, -0.5, 0.5]),
                ([(4, 11, 0, 3), (4, 11, 3, 0), (11, 4, 0, 3), (11, 4, 3, 0)],
                 [0.5, -0.5, -0.5, 0.5]),
                ([(4, 12, 0, 3), (4, 12, 3, 0), (5, 11, 0, 3), (5, 11, 3, 0),
                  (11, 5, 0, 3), (11, 5, 3, 0), (12, 4, 0, 3), (12, 4, 3, 0)],
                 [sq8, -sq8, -sq8,  sq8,  sq8, -sq8, -sq8,  sq8]),
                ([(4, 12, 0, 3), (4, 12, 3, 0), (5, 11, 0, 3), (5, 11, 3, 0),
                  (11, 5, 0, 3), (11, 5, 3, 0), (12, 4, 0, 3), (12, 4, 3, 0)],
                 [sq8, -sq8,  sq8, -sq8, -sq8,  sq8, -sq8,  sq8]),
                ([(5, 12, 0, 3), (5, 12, 3, 0), (12, 5, 0, 3), (12, 5, 3, 0)],
                 [0.5, -0.5, -0.5, 0.5]),
            ],
        }
