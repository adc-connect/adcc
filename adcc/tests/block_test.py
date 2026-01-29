#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
import adcc
import pytest
from adcc.block import get_canonical_block
from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    inp: str
    sym: str
    expected: str
    factor: int
    transpose: tuple


cases_1p = [
    Block("o1v1", adcc.OperatorSymmetry.HERMITIAN, "o1v1", 1, (0, 1)),
    Block("v1o1", adcc.OperatorSymmetry.HERMITIAN, "o1v1", 1, (1, 0)),
    Block("o1o1", adcc.OperatorSymmetry.HERMITIAN, "o1o1", 1, (0, 1)),
    Block("v1v1", adcc.OperatorSymmetry.HERMITIAN, "v1v1", 1, (0, 1)),
    Block("o1v1", adcc.OperatorSymmetry.ANTIHERMITIAN, "o1v1",  1, (0, 1)),
    Block("v1o1", adcc.OperatorSymmetry.ANTIHERMITIAN, "o1v1", -1, (1, 0)),
    Block("o1o1", adcc.OperatorSymmetry.ANTIHERMITIAN, "o1o1",  1, (0, 1)),
    Block("v1v1", adcc.OperatorSymmetry.ANTIHERMITIAN, "v1v1",  1, (0, 1)),
    Block("o1v1", adcc.OperatorSymmetry.NOSYMMETRY, "o1v1",  1, (0, 1)),
    Block("v1o1", adcc.OperatorSymmetry.NOSYMMETRY, "v1o1",  1, (0, 1)),
    Block("o1o1", adcc.OperatorSymmetry.NOSYMMETRY, "o1o1",  1, (0, 1)),
    Block("v1v1", adcc.OperatorSymmetry.NOSYMMETRY, "v1v1",  1, (0, 1))
]

# test only subset
cases_2p = [
    Block("o1o1v1v1", adcc.OperatorSymmetry.HERMITIAN, "o1o1v1v1", 1, (0, 1, 2, 3)),
    Block("v1v1o1o1", adcc.OperatorSymmetry.HERMITIAN, "o1o1v1v1", 1, (2, 3, 0, 1)),
    Block("o1v1o1o1", adcc.OperatorSymmetry.HERMITIAN, "o1o1o1v1", 1, (2, 3, 0, 1)),
    Block("v1o1o1v1", adcc.OperatorSymmetry.HERMITIAN, "o1v1o1v1", -1,
          (1, 0, 2, 3)),

    Block("o1o1v1v1", adcc.OperatorSymmetry.ANTIHERMITIAN, "o1o1v1v1", 1,
          (0, 1, 2, 3)),
    Block("v1v1o1o1", adcc.OperatorSymmetry.ANTIHERMITIAN, "o1o1v1v1", -1,
          (2, 3, 0, 1)),
    Block("o1v1o1o1", adcc.OperatorSymmetry.ANTIHERMITIAN, "o1o1o1v1", -1,
          (2, 3, 0, 1)),
    Block("v1o1o1v1", adcc.OperatorSymmetry.ANTIHERMITIAN, "o1v1o1v1", -1,
          (1, 0, 2, 3)),

    Block("o1o1v1v1", adcc.OperatorSymmetry.NOSYMMETRY, "o1o1v1v1", 1,
          (0, 1, 2, 3)),
    Block("v1v1o1o1", adcc.OperatorSymmetry.NOSYMMETRY, "v1v1o1o1", 1,
          (0, 1, 2, 3)),
    Block("o1v1o1o1", adcc.OperatorSymmetry.NOSYMMETRY, "o1v1o1o1", 1,
          (0, 1, 2, 3)),
    Block("v1o1o1v1", adcc.OperatorSymmetry.NOSYMMETRY, "o1v1o1v1", -1,
          (1, 0, 2, 3)),
]


class TestBlock:
    @pytest.mark.parametrize("case", cases_1p,
                             ids=[f"{c.inp}_{c.sym.name}" for c in cases_1p])
    def test_get_canonical_block_1p(self, case):
        bra, ket = case.inp[:2], case.inp[2:]
        c_block, factor, transpose = get_canonical_block(bra, ket, case.sym)
        assert c_block == case.expected
        assert factor == case.factor
        assert transpose == case.transpose

    @pytest.mark.parametrize("case", cases_2p,
                             ids=[f"{c.inp}_{c.sym.name}" for c in cases_2p])
    def test_get_canonical_block_2p(self, case):
        bra, ket = case.inp[:4], case.inp[4:]
        c_block, factor, transpose = get_canonical_block(bra, ket, case.sym)
        assert c_block == case.expected
        assert factor == case.factor
        assert transpose == case.transpose
