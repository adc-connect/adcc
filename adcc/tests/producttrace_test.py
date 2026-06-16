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
import numpy as np

from adcc import OneParticleOperator
from adcc.NParticleOperator import product_trace, OperatorSymmetry
from adcc.OneParticleDensity import OneParticleDensity
from adcc.MoSpaces import split_spaces

from . import testcases
from adcc.backends import run_hf
from itertools import combinations_with_replacement
import string


operator_sym = [OperatorSymmetry.HERMITIAN, OperatorSymmetry.ANTIHERMITIAN,
                OperatorSymmetry.NOSYMMETRY]
op_syms_two_operators = list(combinations_with_replacement(operator_sym, 2))


def trace_contract(op_a, op_b):
    letters = string.ascii_lowercase
    indices = "".join([letters[i] for i in range(op_a.ndim)])  # "a", "b", "c", ...
    subscripts = f"{indices},{indices}->"
    return np.einsum(subscripts, op_a, op_b)


class TestProductTrace:
    @pytest.mark.parametrize("symmetries",
                             op_syms_two_operators,
                             ids=[f"{c[0].name}_{c[1].name}"
                                  for c in op_syms_two_operators])
    def test_product_trace(self, symmetries):
        system = "h2o_sto3g"
        system: testcases.TestCase = testcases.get_by_filename(system).pop()
        scfres = run_hf("pyscf", system.xyz, system.basis)
        ref = adcc.ReferenceState(scfres)

        sym_1 = symmetries[0]
        sym_2 = symmetries[1]

        op_a_mo = OneParticleOperator(ref, symmetry=sym_1)
        op_b_mo = OneParticleDensity(ref, symmetry=sym_2)

        op_a_mo.set_random()
        op_a_ao_a = op_a_mo.to_ao_basis(ref)[0].to_ndarray()
        op_a_ao_b = op_a_mo.to_ao_basis(ref)[1].to_ndarray()

        op_b_mo.set_random()
        op_b_ao_a = op_b_mo.to_ao_basis(ref)[0].to_ndarray()
        op_b_ao_b = op_b_mo.to_ao_basis(ref)[1].to_ndarray()

        ptrace_ao = (
            trace_contract(op_a_ao_a, op_b_ao_a)
            + trace_contract(op_a_ao_b, op_b_ao_b)
        )

        ptrace_mo_ref = 0
        if op_a_mo.symmetry == OperatorSymmetry.NOSYMMETRY:
            factors = op_a_mo.canonical_factors.copy()
        elif op_b_mo.symmetry == OperatorSymmetry.NOSYMMETRY:
            factors = op_b_mo.canonical_factors.copy()
        else:
            assert op_a_mo.canonical_factors == op_a_mo.canonical_factors
            factors = op_a_mo.canonical_factors.copy()
            if op_a_mo.symmetry is not op_b_mo.symmetry:
                to_remove = []
                for b in list(factors.keys()):
                    spaces = split_spaces(b)
                    n = op_a_mo.n_particle_op
                    if spaces[:n] != spaces[n:]:
                        to_remove.append(b)
                # remove non diagonals blocks
                for b in to_remove:
                    factors.pop(b)

        for b, factor in factors.items():
            ptrace_mo_ref += factor * np.sum(
                op_a_mo[b].to_ndarray() * op_b_mo[b].to_ndarray()
            )
        assert ptrace_mo_ref == pytest.approx(product_trace(op_a_mo, op_b_mo))
        assert product_trace(op_a_mo, op_b_mo) == pytest.approx(ptrace_ao)
        assert product_trace(op_b_mo, op_a_mo) == pytest.approx(ptrace_ao)
