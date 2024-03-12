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

from pytest import approx
from numpy.testing import assert_array_equal
from adcc.testdata.cache import cache

from .misc import expand_test_templates

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
        gts = guess.ph.to_ndarray()
        assert gts.shape == (nCa + nCb, nva + nvb)
        assert np.max(np.abs(gts[nCa:, :nva])) == 0
        assert np.max(np.abs(gts[:nCa, nva:])) == 0

        if matrix.reference_state.restricted:
            assert_array_equal(gts[:nCa, :nva],
                               fac * gts[nCa:, nva:])

        # Doubles
        if "pphh" not in matrix.axis_blocks:
            return

        gtd = guess.pphh.to_ndarray()
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

        if block == "ph":
            assert np.max(np.abs(gtd[:noa, :nCa, :nva, :nva])) == 0
            assert np.max(np.abs(gtd[noa:, nCa:, nva:, nva:])) == 0
            assert np.max(np.abs(gtd[:noa, nCa:, :nva, nva:])) == 0
            assert np.max(np.abs(gtd[noa:, :nCa, nva:, :nva])) == 0
            assert np.max(np.abs(gtd[:noa, nCa:, nva:, :nva])) == 0
            assert np.max(np.abs(gtd[noa:, :nCa, :nva, nva:])) == 0
            has_aa = np.max(np.abs(gts[:nCa, :nva])) > 0
            has_bb = np.max(np.abs(gts[nCa:, nva:])) > 0
            assert has_aa or has_bb
        elif block == "pphh":
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
        gts = guess.ph.to_ndarray()
        assert gts.shape == (nCa + nCb, nva + nvb)
        assert np.max(np.abs(gts[:nCa, :nva])) == 0  # a->a
        assert np.max(np.abs(gts[nCa:, :nva])) == 0  # b->a
        assert np.max(np.abs(gts[nCa:, nva:])) == 0  # b->b

        # Doubles
        if "pphh" not in matrix.axis_blocks:
            return

        gtd = guess.pphh.to_ndarray()
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

        if block == "ph":
            assert np.max(np.abs(gtd[:noa, :nCa, :nva, nva:])) == 0  # aa->ab
            assert np.max(np.abs(gtd[:noa, :nCa, nva:, :nva])) == 0  # aa->ba
            assert np.max(np.abs(gtd[:noa, nCa:, nva:, nva:])) == 0  # ab->bb
            assert np.max(np.abs(gtd[noa:, :nCa, nva:, nva:])) == 0  # ba->bb
            assert np.max(np.abs(gts[:nCa, nva:])) > 0
        elif block == "pphh":
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
        if block == "ph":
            diagonal = matrix.diagonal().ph.to_ndarray()

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
        elif block == "pphh":
            diagonal = matrix.diagonal().pphh.to_ndarray()

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
        self.base_test_no_spin_change("h2o_sto3g", method, "ph")

    def template_singles_h2o_cvs(self, method):
        self.base_test_no_spin_change("h2o_sto3g", "cvs-" + method, "ph",
                                      max_guesses=2)

    def template_singles_cn(self, method):
        self.base_test_no_spin_change("cn_sto3g", method, "ph")

    def template_singles_cn_cvs(self, method):
        self.base_test_no_spin_change("cn_sto3g", "cvs-" + method, "ph",
                                      max_guesses=7)

    def template_singles_hf3(self, method):
        self.base_test_spin_flip("hf3_631g", method, "ph")

    # TODO These tests fails because of adcman, because some delta-Fock-based
    #      approximation is used for the diagonal instead of the actual
    #      doubles diagonal which would be employed for ADC(2) and ADC(3)
    # def test_doubles_h2o_adc2(self):
    #     self.base_test_no_spin_change("h2o_sto3g", "adc2", "pphh", max_guesses=3)
    #
    # def test_doubles_h2o_adc3(self):
    #     self.base_test_no_spin_change("h2o_sto3g", "adc3", "pphh", max_guesses=3)
    #
    # def test_doubles_cn_adc2(self):
    #     self.base_test_no_spin_change("cn_sto3g", "adc2", "pphh")
    #
    # def test_doubles_cn_adc3(self):
    #    self.base_test_no_spin_change("cn_sto3g", "adc3", "pphh")
    #
    # TODO Perhaps could be templatified as well one the issues are resolved

    def test_doubles_h2o_cvs_adc2(self):
        self.base_test_no_spin_change("h2o_sto3g", "cvs-adc2", "pphh",
                                      max_guesses=5)

    def test_doubles_hf3_adc2(self):
        self.base_test_spin_flip("hf3_631g", "adc2", "pphh")

    # TODO See above
    # def test_doubles_hf3_adc3(self):
    #     self.base_test_spin_flip("hf3_631g", "adc3", "pphh")

    #
    # Tests against reference values
    #
    def base_reference(self, matrix, ref):
        symmetrisations = ["none"]
        if matrix.reference_state.restricted:
            symmetrisations = ["symmetric", "antisymmetric"]

        for block in ["ph", "pphh"]:
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
            ("ph", "symmetric"): [
                ([(4, 0), (9, 2)], symm), ([(4, 1), (9, 3)], symm),
                ([(3, 0), (8, 2)], symm), ([(3, 1), (8, 3)], symm),
                ([(2, 0), (7, 2)], symm), ([(2, 1), (7, 3)], symm),
                ([(1, 0), (6, 2)], symm), ([(1, 1), (6, 3)], symm),
            ],
            ("ph", "antisymmetric"): [
                # nonzeros          values
                ([(4, 0), (9, 2)], asymm), ([(4, 1), (9, 3)], asymm),
                ([(3, 0), (8, 2)], asymm), ([(3, 1), (8, 3)], asymm),
                ([(2, 0), (7, 2)], asymm), ([(2, 1), (7, 3)], asymm),
                ([(1, 0), (6, 2)], asymm), ([(1, 1), (6, 3)], asymm),
            ],
            ("pphh", "symmetric"): [
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
            ("pphh", "antisymmetric"): [
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
            ("ph", "none"): [
                ([(11, 3)], [1.]), ([(12, 3)], [1.]),
                ([( 6, 0)], [1.]), ([( 6, 1)], [1.]),  # noqa: E201
                ([(10, 3)], [1.]), ([( 5, 0)], [1.]),  # noqa: E201
                ([( 4, 1)], [1.]), ([(11, 5)], [1.]),  # noqa: E201
            ],
            ("pphh", "none"): [
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


@expand_test_templates(methods)
class TestGuessIpEa(unittest.TestCase):
    def assert_symmetry_ip_ea(self, matrix, guess, block, is_alpha=True):
        """
        Assert a guess vector has the correct symmetry.
        Alpha ionization/electron attachment is assumed if the reference is
        restricted.
        For an alpha attachment/detachment, all corresponding beta blocks
        should be unpopulated and vice versa.
        """
        # Extract useful quantities
        mospaces = matrix.mospaces
        noa = mospaces.n_orbs_alpha("o1")
        nob = mospaces.n_orbs_beta("o1")
        nva = mospaces.n_orbs_alpha("v1")
        nvb = mospaces.n_orbs_beta("v1")
        blocks = sorted(guess.keys())
        assert blocks[0] in ["p", "h"]
        if len(blocks) == 2:
            assert blocks[1] in ["pph", "phh"]
        is_ip = matrix.type == "ip"

        # Singles
        gts = guess.get(blocks[0]).to_ndarray()
        gts_shape = (noa + nob,) if is_ip else (nva + nvb,)
        assert gts.shape == gts_shape
        if is_ip:
            if is_alpha:
                assert np.max(np.abs(gts[noa:])) == 0
            else:
                assert np.max(np.abs(gts[:noa])) == 0
        # EA-ADC
        else:
            if is_alpha:
                assert np.max(np.abs(gts[nva:])) == 0
            else:
                assert np.max(np.abs(gts[:nva])) == 0

        # Doubles
        if len(blocks) != 2:
            return

        gtd = guess.get(blocks[1]).to_ndarray()
        if is_ip:
            gtd_shape = (noa + nob, noa + nob, nva + nvb)
        # EA-ADC
        else:
            gtd_shape = (noa + nob, nva + nvb, nva + nvb)
        assert gtd.shape == gtd_shape

        # Make sure the wrong attached/detached spins are unpopulated
        if is_ip:
            if is_alpha:
                assert np.max(np.abs(gtd[:noa, noa:, :nva])) == 0  # ab->a
                assert np.max(np.abs(gtd[noa:, :noa, :nva])) == 0  # ba->a
                assert np.max(np.abs(gtd[noa:, noa:, nva:])) == 0  # bb->b
                assert np.max(np.abs(gtd[noa:, noa:, :nva])) == 0  # bb->a
            else:
                assert np.max(np.abs(gtd[:noa, noa:, nva:])) == 0  # ab->b
                assert np.max(np.abs(gtd[noa:, :noa, nva:])) == 0  # ba->b
                assert np.max(np.abs(gtd[:noa, :noa, :nva])) == 0  # aa->a
                assert np.max(np.abs(gtd[:noa, :noa, nva:])) == 0  # aa->b

        # EA-ADC
        else:
            if is_alpha:
                assert np.max(np.abs(gtd[:noa, :nva, nva:])) == 0  # a->ab
                assert np.max(np.abs(gtd[:noa, nva:, :nva])) == 0  # a->ba
                assert np.max(np.abs(gtd[noa:, nva:, nva:])) == 0  # b->bb
                assert np.max(np.abs(gtd[:noa, nva:, nva:])) == 0  # a->bb
            else:
                assert np.max(np.abs(gtd[noa:, :nva, nva:])) == 0  # b->ab
                assert np.max(np.abs(gtd[noa:, nva:, :nva])) == 0  # b->ba
                assert np.max(np.abs(gtd[:noa, :nva, :nva])) == 0  # a->aa
                assert np.max(np.abs(gtd[noa:, :nva, :nva])) == 0  # b->aa

        if matrix.reference_state.restricted:
            # Restricted automatically means alpha ionization/electron
            # attachment. Additionally forbid spin-flip blocks with right spin.
            if is_ip:
                assert np.max(np.abs(gtd[:noa, :noa, nva:])) == 0  # aa->b
            # EA-ADC
            else:
                assert np.max(np.abs(gtd[:noa, nva:, nva:])) == 0  # a->bb

        if is_ip:
            assert_array_equal(gtd.transpose((1, 0, 2)), -gtd)
        else:
            assert_array_equal(gtd.transpose((0, 2, 1)), -gtd)

        if block in ["h", "p"]:
            if is_ip:
                if is_alpha:
                    assert np.max(np.abs(gtd[:noa, noa:, nva:])) == 0  # ab->b
                    assert np.max(np.abs(gtd[noa:, :noa, nva:])) == 0  # ba->b
                    assert np.max(np.abs(gtd[:noa, :noa, :nva])) == 0  # aa->a

                    assert np.max(np.abs(gts[:noa])) > 0  # has_alpha
                else:
                    assert np.max(np.abs(gtd[:noa, noa:, :nva])) == 0  # ab->a
                    assert np.max(np.abs(gtd[noa:, :noa, :nva])) == 0  # ba->a
                    assert np.max(np.abs(gtd[noa:, noa:, nva:])) == 0  # bb->b

                    assert np.max(np.abs(gts[noa:])) > 0  # has_beta
            # EA-ADC
            else:
                if is_alpha:
                    assert np.max(np.abs(gtd[noa:, :nva, nva:])) == 0  # b->ab
                    assert np.max(np.abs(gtd[noa:, nva:, :nva])) == 0  # b->ba
                    assert np.max(np.abs(gtd[:noa, :nva, :nva])) == 0  # a->aa

                    assert np.max(np.abs(gts[:nva])) > 0  # has_alpha
                else:
                    assert np.max(np.abs(gtd[:noa, :nva, nva:])) == 0  # a->ab
                    assert np.max(np.abs(gtd[:noa, nva:, :nva])) == 0  # a->ba
                    assert np.max(np.abs(gtd[noa:, nva:, nva:])) == 0  # b->bb

                    assert np.max(np.abs(gts[nva:])) > 0  # has_beta

        elif block in ["pph", "phh"]:
            if is_ip:
                if is_alpha:
                    assert np.max(np.abs(gts[:noa])) == 0
                    has_aaa = np.max(np.abs(gtd[:noa, :noa, :nva])) > 0
                    has_abb = np.max(np.abs(gtd[:noa, noa:, nva:])) > 0
                    has_bab = np.max(np.abs(gtd[noa:, :noa, nva:])) > 0
                    assert has_aaa or has_abb or has_bab
                else:
                    assert np.max(np.abs(gts[noa:])) == 0
                    has_aba = np.max(np.abs(gtd[:noa, noa:, :nva])) > 0
                    has_baa = np.max(np.abs(gtd[noa:, :noa, :nva])) > 0
                    has_bbb = np.max(np.abs(gtd[noa:, noa:, nva:])) > 0
                    assert has_aba or has_baa or has_bbb
            # EA-ADC
            else:
                if is_alpha:
                    assert np.max(np.abs(gts[:nva])) == 0
                    has_aaa = np.max(np.abs(gtd[:noa, :nva, :nva])) > 0
                    has_bab = np.max(np.abs(gtd[noa:, :nva, nva:])) > 0
                    has_bba = np.max(np.abs(gtd[noa:, nva:, :nva])) > 0
                    assert has_aaa or has_bab or has_bba
                else:
                    assert np.max(np.abs(gts[nva:])) == 0
                    has_aab = np.max(np.abs(gtd[:noa, :nva, nva:])) > 0
                    has_aba = np.max(np.abs(gtd[:noa, nva:, :nva])) > 0
                    has_bbb = np.max(np.abs(gtd[noa:, nva:, nva:])) > 0
                    assert has_aab or has_aba or has_bbb

    def assert_orthonormal(self, guesses):
        for (i, gi) in enumerate(guesses):
            for (j, gj) in enumerate(guesses):
                ref = 1 if i == j else 0
                assert adcc.dot(gi, gj) == approx(ref)

    def assert_guess_values(self, matrix, block, guesses, is_alpha):
        """
        Assert that the guesses correspond to the smallest
        diagonal values.
        """
        # Extract useful quantities
        mospaces = matrix.mospaces
        noa = mospaces.n_orbs_alpha("o1")
        nva = mospaces.n_orbs_alpha("v1")

        # Make a list of diagonal indices, ordered by the corresponding
        # diagonal values
        sidcs = None
        diagonal = matrix.diagonal().get(block).to_ndarray()
        # Build list of indices, which would sort the diagonal
        sidcs = np.dstack(np.unravel_index(np.argsort(diagonal.ravel()),
                                           diagonal.shape))
        assert sidcs.shape[0] == 1

        if block == "h":
            # IP-ADC singles
            if is_alpha:
                sidcs = [idx for idx in sidcs[0] if idx[0] < noa]
            else:
                sidcs = [idx for idx in sidcs[0] if idx[0] >= noa]
        elif block == "p":
            # EA-ADC singles
            if is_alpha:
                sidcs = [idx for idx in sidcs[0] if idx[0] < nva]
            else:
                sidcs = [idx for idx in sidcs[0] if idx[0] >= nva]
        elif block == "phh":
            # IP-ADC doubles
            if is_alpha:
                sidcs = [idx for idx in sidcs[0] if any(
                    (idx[0]  < noa and idx[1]  < noa and idx[2]  < nva,
                     idx[0]  < noa and idx[1] >= noa and idx[2] >= nva,
                     idx[0] >= noa and idx[1]  < noa and idx[2] >= nva))]
            else:
                sidcs = [idx for idx in sidcs[0] if any(
                    (idx[0]  < noa and idx[1] >= noa and idx[2]  < nva,
                     idx[0] >= noa and idx[1]  < noa and idx[2]  < nva,
                     idx[0] >= noa and idx[1] >= noa and idx[2] >= nva))]
            sidcs = [idx for idx in sidcs if idx[0] != idx[1]]
        elif block == "pph":
            # EA-ADC doubles
            if is_alpha:
                sidcs = [idx for idx in sidcs[0] if any(
                    (idx[0]  < noa and idx[1]  < nva and idx[2]  < nva,
                     idx[0] >= noa and idx[1]  < nva and idx[2] >= nva,
                     idx[0] >= noa and idx[1] >= nva and idx[2]  < nva))]
            else:
                sidcs = [idx for idx in sidcs[0] if any(
                    (idx[0]  < noa and idx[1]  < nva and idx[2] >= nva,
                     idx[0]  < noa and idx[1] >= nva and idx[2]  < nva,
                     idx[0] >= noa and idx[1] >= nva and idx[2] >= nva))]
            sidcs = [idx for idx in sidcs if idx[1] != idx[2]]

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

    def base_test_ip_ea(self, case, method, block, is_alpha, max_guesses=3):
        ground_state = adcc.LazyMp(cache.refstate[case])
        matmethod = method.split("-")[-1]
        matrix = adcc.AdcMatrix(matmethod, ground_state)

        is_ip = "ip" in method
        # Only spin changes of +- 0.5
        spin_change = -0.5 if is_ip else +0.5
        if not is_alpha:
            spin_change *= -1

        for n_guesses in range(1, max_guesses + 1):
            guesses = adcc.guess.guesses_from_diagonal(
                matrix, n_guesses, block=block, spin_change=spin_change,
                spin_block_symmetrisation="none"
            )
            assert len(guesses) == n_guesses
            for gs in guesses:
                self.assert_symmetry_ip_ea(matrix, gs, block, is_alpha)
            self.assert_orthonormal(guesses)
            self.assert_guess_values(matrix, block, guesses, is_alpha)

    def template_singles_h2o_ip_alpha(self, method):
        self.base_test_ip_ea("h2o_sto3g", "ip_" + method, "h", True)

    def template_singles_h2o_ea_alpha(self, method):
        self.base_test_ip_ea("h2o_sto3g", "ea_" + method, "p", True,
                             max_guesses=2)

    def template_singles_cn_ip_alpha(self, method):
        self.base_test_ip_ea("cn_sto3g", "alpha-ip_" + method, "h", True)

    def template_singles_cn_ip_beta(self, method):
        self.base_test_ip_ea("cn_sto3g", "beta-ip_" + method, "h", False)

    def template_singles_cn_ea_alpha(self, method):
        self.base_test_ip_ea("cn_sto3g", "alpha-ea_" + method, "p", True)

    def template_singles_cn_ea_beta(self, method):
        self.base_test_ip_ea("cn_sto3g", "beta-ea_" + method, "p", False)

    #
    # Tests against reference values
    #
    def base_reference(self, matrix, ref, is_alpha=True):
        is_ip = matrix.type == "ip"
        # Only spin changes of +- 0.5
        spin_change = -0.5 if is_ip else +0.5
        if not is_alpha:
            spin_change *= -1
        for block in matrix.axis_blocks:
            ref_sb = ref[(block, is_alpha)]
            guesses = adcc.guess.guesses_from_diagonal(
                matrix, len(ref_sb), block, spin_change=spin_change,
                spin_block_symmetrisation="none"
            )
            assert len(guesses) == len(ref_sb)
            for gs in guesses:
                self.assert_symmetry_ip_ea(matrix, gs, block, is_alpha)
            self.assert_orthonormal(guesses)

            for (i, guess) in enumerate(guesses):
                guess_b = guess[block].to_ndarray()
                nonzeros = np.dstack(np.where(guess_b != 0))
                assert nonzeros.shape[0] == 1
                nonzeros = [tuple(nzitem) for nzitem in nonzeros[0]]
                values = guess_b[guess_b != 0]
                assert nonzeros == ref_sb[i][0]
                assert_array_equal(values, np.array(ref_sb[i][1]))

    def test_reference_h2o_ip_adc2(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("ip_adc2", ground_state)
        self.base_reference(matrix, self.get_ref_h2o_ip_ea())

    def test_reference_h2o_ea_adc3(self):
        ground_state = adcc.LazyMp(cache.refstate["h2o_sto3g"])
        matrix = adcc.AdcMatrix("ea_adc3", ground_state)
        self.base_reference(matrix, self.get_ref_h2o_ip_ea())

    def test_reference_cn_ip_adc2_alpha(self):
        ground_state = adcc.LazyMp(cache.refstate["cn_sto3g"])
        matrix = adcc.AdcMatrix("ip_adc2", ground_state)
        self.base_reference(matrix, self.get_ref_cn_ip_ea())

    def test_reference_cn_ip_adc3_beta(self):
        ground_state = adcc.LazyMp(cache.refstate["cn_sto3g"])
        matrix = adcc.AdcMatrix("ip_adc2", ground_state)
        self.base_reference(matrix, self.get_ref_cn_ip_ea(), False)

    def test_reference_cn_ea_adc2_alpha(self):
        ground_state = adcc.LazyMp(cache.refstate["cn_sto3g"])
        matrix = adcc.AdcMatrix("ea_adc2", ground_state)
        self.base_reference(matrix, self.get_ref_cn_ip_ea())

    def test_reference_cn_ea_adc3_beta(self):
        ground_state = adcc.LazyMp(cache.refstate["cn_sto3g"])
        matrix = adcc.AdcMatrix("ea_adc3", ground_state)
        self.base_reference(matrix, self.get_ref_cn_ip_ea(), False)

    def get_ref_h2o_ip_ea(self):
        sq6 = 1 / np.sqrt(6)
        asymm = [1 / np.sqrt(2), -1 / np.sqrt(2)]
        return {
            ("h", True): [
                ([(4, )], [1]),
                ([(3, )], [1]),
                ([(2, )], [1]),
                ([(1, )], [1]),
                ([(0, )], [1])
            ],
            ("p", True): [
                ([(0, )], [1]),
                ([(1, )], [1])
            ],
            ("phh", True): [
                ([(4, 9, 2), (9, 4, 2)], asymm),
                ([(3, 4, 0), (3, 9, 2), (4, 3, 0),
                  (4, 8, 2), (8, 4, 2), (9, 3, 2)],
                 [-sq6, -sq6, sq6, -sq6, sq6, sq6]),
                ([(3, 8, 2), (8, 3, 2)], asymm),
                ([(4, 9, 3), (9, 4, 3)], asymm),
                ([(3, 4, 1), (3, 9, 3), (4, 3, 1),
                  (4, 8, 3), (8, 4, 3), (9, 3, 3)],
                 [-sq6, -sq6, sq6, -sq6, sq6, sq6])
            ],
            ("pph", True): [
                ([(9, 0, 2), (9, 2, 0)], asymm),
                ([(8, 0, 2), (8, 2, 0)], asymm),
                ([(4, 0, 1), (4, 1, 0), (9, 0, 3),
                  (9, 1, 2), (9, 2, 1), (9, 3, 0)],
                 [-sq6, sq6, -sq6, -sq6, sq6, sq6]),
                ([(3, 0, 1), (3, 1, 0), (8, 0, 3),
                  (8, 1, 2), (8, 2, 1), (8, 3, 0)],
                 [-sq6, sq6, -sq6, -sq6, sq6, sq6]),
                ([(7, 0, 2), (7, 2, 0)], asymm)
            ],
        }

    def get_ref_cn_ip_ea(self):
        asymm = [1 / np.sqrt(2), -1 / np.sqrt(2)]
        asymm1 = [-1 / np.sqrt(2), 1 / np.sqrt(2)]
        return {
            ("h", True): [
                ([(6, )], [1]),
                ([(4, )], [1]),  # occ. 4 and 5 are degenerate
                ([(5, )], [1]),  # occ. 4 and 5 are degenerate
                ([(3, )], [1]),
                ([(2, )], [1])
            ],
            ("h", False): [
                ([(11, )], [1]),
                ([(12, )], [1]),
                ([(10, )], [1]),
                ([(9, )],  [1]),
                ([(8, )],  [1])
            ],
            ("p", True): [
                ([(0, )], [1]),
                ([(1, )], [1]),
                ([(2, )], [1])
            ],
            ("p", False): [
                ([(3, )], [1]),
                ([(4, )], [1]),
                ([(5, )], [1]),
                ([(6, )], [1])
            ],
            ("phh", True): [
                ([(6, 11, 3), (11, 6, 3)], asymm1),
                ([(6, 12, 3), (12, 6, 3)], asymm1),
                ([(4, 11, 3), (11, 4, 3)], asymm),
                ([(4, 12, 3), (12, 4, 3)], asymm),
                ([(5, 11, 3), (11, 5, 3)], asymm1)
            ],
            ("phh", False): [
                ([(11, 12, 3), (12, 11, 3)], asymm),
                ([(10, 11, 3), (11, 10, 3)], asymm),
                ([(10, 12, 3), (12, 10, 3)], asymm),
                ([(6, 11, 0), (11, 6, 0)], asymm1),
                ([(6, 12, 0), (12, 6, 0)], asymm1)
            ],
            ("pph", True): [
                ([(11, 0, 3), (11, 3, 0)], asymm),
                ([(12, 0, 3), (12, 3, 0)], asymm),
                ([(11, 1, 3), (11, 3, 1)], asymm1),
                ([(12, 1, 3), (12, 3, 1)], asymm1),
                ([(10, 0, 3), (10, 3, 0)], asymm)
            ],
            ("pph", False): [
                ([(6, 0, 3), (6, 3, 0)], asymm),
                ([(6, 1, 3), (6, 3, 1)], asymm1),
                ([(4, 0, 3), (4, 3, 0)], asymm),  # occ. 4 and 5 are degenerate
                ([(5, 0, 3), (5, 3, 0)], asymm),  # occ. 4 and 5 are degenerate
                ([(4, 1, 3), (4, 3, 1)], asymm1)
            ],
        }
