#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2022 by the adcc authors
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
import re
import adcc
import unittest
import pytest
import itertools
import numpy as np
from numpy.testing import assert_allclose

from adcc.projection import Projector, SubspacePartitioning, transfer_cvs_to_full
from adcc.HfCounterData import HfCounterData

from .testdata_cache import testdata_cache
from . import testcases


class TestSubspacePartitioning(unittest.TestCase):
    def test_restricted(self):
        n_alpha = 5
        n_beta = 5
        n_bas = 20
        n_orbs_alpha = 10
        restricted = True
        data = HfCounterData(n_alpha, n_beta, n_bas, n_orbs_alpha, restricted)
        refstate = adcc.ReferenceState(data, import_all_below_n_orbs=None)

        partitioning = SubspacePartitioning(refstate.mospaces, core_orbitals=2,
                                            outer_virtuals=3)
        assert "v" in partitioning.aliases
        assert "w" in partitioning.aliases
        assert "o" in partitioning.aliases
        assert "c" in partitioning.aliases
        assert partitioning.keeps_spin_symmetry

        assert partitioning.list_space_partitions("o1") == ["c", "o"]
        assert partitioning.list_space_partitions("v1") == ["v", "w"]

        assert partitioning.get_partition("c") == [0, 1, 5, 6]
        assert partitioning.get_partition("o") == [2, 3, 4, 7, 8, 9]
        assert partitioning.get_partition("v") == [0, 1, 5, 6]
        assert partitioning.get_partition("w") == [2, 3, 4, 7, 8, 9]

    def test_unrestricted(self):
        n_alpha = 6
        n_beta = 5
        n_bas = 20
        n_orbs_alpha = 10
        restricted = False
        data = HfCounterData(n_alpha, n_beta, n_bas, n_orbs_alpha, restricted)
        refstate = adcc.ReferenceState(data, import_all_below_n_orbs=None)

        partitioning = SubspacePartitioning(refstate.mospaces, core_orbitals=2,
                                            outer_virtuals=2)
        assert "v" in partitioning.aliases
        assert "w" in partitioning.aliases
        assert "o" in partitioning.aliases
        assert "c" in partitioning.aliases
        assert not partitioning.keeps_spin_symmetry

        assert partitioning.list_space_partitions("o1") == ["c", "o"]
        assert partitioning.list_space_partitions("v1") == ["v", "w"]

        assert partitioning.get_partition("c") == [0, 1, 6, 7]
        assert partitioning.get_partition("o") == [2, 3, 4, 5, 8, 9, 10]
        assert partitioning.get_partition("v") == [0, 1, 4, 5, 6]
        assert partitioning.get_partition("w") == [2, 3, 7, 8]


def construct_nonzero_blocks(mospaces, n_core, n_virt):
    nva = mospaces.n_orbs_alpha("v1")
    nvb = mospaces.n_orbs_beta("v1")
    noa = mospaces.n_orbs_alpha("o1")
    nob = mospaces.n_orbs_beta("o1")
    c = dict(a=slice(0,         n_core),
             b=slice(noa, noa + n_core))
    o = dict(a=slice(n_core,             noa),
             b=slice(n_core + noa, noa + nob))
    v = dict(a=slice(0,         nva - n_virt),
             b=slice(nva, nvb + nva - n_virt))
    w = dict(a=slice(nva - n_virt,             nva),
             b=slice(nva - n_virt + nvb, nvb + nva))

    spaces_ph = ["cv", "ow"]
    nonzero_blocks_ph = []
    for (s1, s2) in itertools.product(["a", "b"], ["a", "b"]):
        nonzero_blocks_ph.append((c[s1], v[s2]))
        nonzero_blocks_ph.append((o[s1], w[s2]))

    spaces_pphh = ["covv", "ccvw", "covw"]
    nonzero_blocks_pphh = []
    for (s1, s2, s3, s4) in itertools.product(["a", "b"], ["a", "b"],
                                              ["a", "b"], ["a", "b"]):
        nonzero_blocks_pphh.append((c[s1], o[s2], v[s3], v[s4]))
        nonzero_blocks_pphh.append((o[s1], c[s2], v[s3], v[s4]))
        #
        nonzero_blocks_pphh.append((c[s1], c[s2], v[s3], w[s4]))
        nonzero_blocks_pphh.append((c[s1], c[s2], w[s3], v[s4]))
        #
        nonzero_blocks_pphh.append((c[s1], o[s2], v[s3], w[s4]))
        nonzero_blocks_pphh.append((o[s1], c[s2], v[s3], w[s4]))
        nonzero_blocks_pphh.append((c[s1], o[s2], w[s3], v[s4]))
        nonzero_blocks_pphh.append((o[s1], c[s2], w[s3], v[s4]))

    spaces = dict(ph=spaces_ph, pphh=spaces_pphh)
    nonzeros = dict(ph=nonzero_blocks_ph, pphh=nonzero_blocks_pphh)
    return spaces, nonzeros


def assert_nonzero_blocks(orig, projected, nonzero_blocks, zero_value=0, tol=0):
    diff = (orig - projected).to_ndarray()
    reset = projected.to_ndarray()
    for block in nonzero_blocks:
        assert np.max(np.abs(diff[block])) <= tol  # Values unchanged
        reset[block] = zero_value
    cond = np.max(np.abs(np.abs(reset) - zero_value)) <= tol
    if not cond:
        print(reset)
    # All zero blocks are equal to zero_value or -zero_value
    assert np.max(np.abs(np.abs(reset) - zero_value)) <= tol


def assert_equal_symmetry(sym1, sym2):
    def split_blocks(sym):
        str_sym = sym.describe_symmetry()

        # Perform some cleanup (effects only point group symmetry)
        str_sym = str_sym.replace("0(1)", "0(0)").replace("1(1)", "1(0)")

        blocks = []
        for line in str_sym.split("\n"):
            if re.match(r"^  [0-9]\.", line):
                blocks.append([])
                line = line[4:]
            blocks[-1].append(line)
        str_blocks = ["\n".join(b).strip() for b in blocks]
        return [b for b in str_blocks if not b.endswith("Mappings:")]

    blocks_sym1 = split_blocks(sym1)
    blocks_sym2 = split_blocks(sym2)
    assert len(blocks_sym1) == len(blocks_sym2)
    for block in blocks_sym1:
        assert block in blocks_sym2


class TestProjector(unittest.TestCase):
    def base_test(self, system: str, kind: str, n_core: int, n_virt: int):
        state = testdata_cache.adcc_states(
            system=system, method="adc3", kind=kind, case="gen"
        )
        mospaces = state.reference_state.mospaces

        partitioning = SubspacePartitioning(mospaces,
                                            core_orbitals=n_core,
                                            outer_virtuals=n_virt)
        spaces, nonzero_blocks = construct_nonzero_blocks(mospaces, n_core, n_virt)

        # Singles
        proj = Projector(["o1", "v1"], partitioning, spaces["ph"])
        rhs = state.excitation_vector[0].ph.copy().set_random()
        out = proj @ rhs
        assert_equal_symmetry(rhs, out)
        assert_nonzero_blocks(rhs, out, nonzero_blocks["ph"])

        #
        # Doubles
        partitioning = SubspacePartitioning(mospaces,
                                            core_orbitals=n_core,
                                            outer_virtuals=n_virt)
        proj = Projector(["o1", "o1", "v1", "v1"], partitioning, spaces["pphh"])
        rhs = state.excitation_vector[0].pphh.copy().set_random()
        out = proj @ rhs
        assert_equal_symmetry(rhs, out)
        assert_nonzero_blocks(rhs, out, nonzero_blocks["pphh"])

    def test_h2o_sto3g_singlet(self):
        self.base_test("h2o_sto3g", "singlet", n_core=2, n_virt=1)

    def test_h2o_sto3g_triplet(self):
        self.base_test("h2o_sto3g", "triplet", n_core=1, n_virt=1)

    def test_cn_sto3g(self):
        self.base_test("cn_sto3g", "any", n_core=2, n_virt=1)

    def test_h2o_def2tzvp_singlet(self):
        self.base_test("h2o_def2tzvp", "singlet", n_core=2, n_virt=5)

    def test_h2o_def2tzvp_triplet(self):
        self.base_test("h2o_def2tzvp", "triplet", n_core=1, n_virt=3)

    def test_cn_ccpvdz(self):
        self.base_test("cn_ccpvdz", "any", n_core=2, n_virt=4)


test_cases = testcases.get_by_filename(
    "h2o_sto3g", "h2o_def2tzvp", "cn_sto3g", "cn_ccpvdz"
)
cases = [(case.file_name, kind) for case in test_cases for kind in case.kinds["pp"]]


@pytest.mark.parametrize("system,kind", cases)
class TestCvsTransfer:
    def test_high_level(self, system: str, kind: str):
        state_cvs = testdata_cache.adcc_states(
            system=system, method="adc2x", kind=kind, case="cvs"
        )
        refstate = testdata_cache.refstate(system, case="gen")
        matrix = adcc.AdcMatrix("adc2x", refstate)

        orth = np.array([[v @ w for v in state_cvs.excitation_vector]
                         for w in state_cvs.excitation_vector])
        fullvecs = transfer_cvs_to_full(state_cvs, matrix)
        orthfull = np.array([[v @ w for v in fullvecs] for w in fullvecs])
        assert_allclose(orth, orthfull, atol=1e-16)

    def test_random(self, system: str, kind: str):
        state_cvs = testdata_cache.adcc_states(
            system=system, method="adc2x", kind=kind, case="cvs"
        )
        refstate = testdata_cache.refstate(system, case="gen")
        matrix = adcc.AdcMatrix("adc2x", refstate)
        vectors = [v.copy().set_random() for v in state_cvs.excitation_vector]

        orth = np.array([[v @ w for v in vectors] for w in vectors])
        fullvecs = transfer_cvs_to_full(state_cvs.matrix, matrix, vectors, kind)
        orthfull = np.array([[v @ w for v in fullvecs] for w in fullvecs])
        assert_allclose(orth, orthfull, atol=1e-16)
