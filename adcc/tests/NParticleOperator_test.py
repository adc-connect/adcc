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
import pytest
from numpy.testing import assert_array_almost_equal_nulp, assert_equal

from adcc.NParticleOperator import OperatorSymmetry, NParticleOperator

from .testdata_cache import testdata_cache
from itertools import combinations_with_replacement


operator_sym = [OperatorSymmetry.HERMITIAN, OperatorSymmetry.ANTIHERMITIAN,
                OperatorSymmetry.NOSYMMETRY]
op_syms_two_operators = list(combinations_with_replacement(operator_sym, 2))
n_particles = [1, 2]


class TestNParticleOperator:
    #
    # Test operators
    #
    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", operator_sym,
                             ids=[f"{c.name}" for c in operator_sym])
    def test_copy(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        op = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym)
        op.set_random()
        cpy = op.copy()

        assert cpy.blocks == op.blocks
        assert cpy.blocks_nonzero == op.blocks_nonzero
        assert cpy.reference_state == op.reference_state
        assert cpy.mospaces == op.mospaces

        for b in op.blocks:
            assert cpy.is_zero_block(b) == op.is_zero_block(b)
            if not op.is_zero_block(b):
                assert_equal(cpy.block(b).to_ndarray(),
                             op.block(b).to_ndarray())
                assert cpy.block(b) is not op.block(b)

    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", op_syms_two_operators,
                             ids=[f"{c[0].name}_{c[1].name}"
                                  for c in op_syms_two_operators])
    def test_add(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[0])
        a.set_random()
        b = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[1])
        b.set_random()

        if op_sym[0] != op_sym[1] and op_sym[1] != OperatorSymmetry.NOSYMMETRY \
                and op_sym[0] != OperatorSymmetry.NOSYMMETRY:
            with pytest.raises(ValueError):
                ref = a + b
        else:
            assert_array_almost_equal_nulp((a + b).to_ndarray(),
                                           a.to_ndarray() + b.to_ndarray())
            assert_array_almost_equal_nulp((b + a).to_ndarray(),
                                           a.to_ndarray() + b.to_ndarray())

    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", op_syms_two_operators,
                             ids=[f"{c[0].name}_{c[1].name}"
                                  for c in op_syms_two_operators])
    def test_iadd(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[0])
        a.set_random()
        b = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[1])
        b.set_random()

        ref = a.to_ndarray() + b.to_ndarray()
        if op_sym[0] != op_sym[1] and op_sym[1] == OperatorSymmetry.NOSYMMETRY:
            with pytest.raises(ValueError):
                a += b
        elif op_sym[0] != op_sym[1] and op_sym[1] != OperatorSymmetry.NOSYMMETRY \
                and op_sym[0] != OperatorSymmetry.NOSYMMETRY:
            with pytest.raises(ValueError):
                a += b
        else:
            a += b
            assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", op_syms_two_operators,
                             ids=[f"{c[0].name}_{c[1].name}"
                                  for c in op_syms_two_operators])
    def test_sub(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[0])
        a.set_random()

        b = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[1])
        b.set_random()

        if op_sym[0] != op_sym[1] and op_sym[1] != OperatorSymmetry.NOSYMMETRY \
                and op_sym[0] != OperatorSymmetry.NOSYMMETRY:
            with pytest.raises(ValueError):
                ref = a - b
        else:
            assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                           a.to_ndarray() - b.to_ndarray())
            assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                           b.to_ndarray() - a.to_ndarray())
            assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                           (a + (-1 * b)).to_ndarray())
            assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                           (b + (-1 * a)).to_ndarray())

    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", op_syms_two_operators,
                             ids=[f"{c[0].name}_{c[1].name}"
                                  for c in op_syms_two_operators])
    def test_isub(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[0])
        a.set_random()

        b = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym[1])
        b.set_random()

        ref = a.to_ndarray() - b.to_ndarray()
        if op_sym[0] != op_sym[1] and op_sym[1] == OperatorSymmetry.NOSYMMETRY:
            with pytest.raises(ValueError):
                a -= b
        elif op_sym[0] != op_sym[1] and op_sym[1] != OperatorSymmetry.NOSYMMETRY \
                and op_sym[0] != OperatorSymmetry.NOSYMMETRY:
            with pytest.raises(ValueError):
                a -= b
        else:
            a -= b
            assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", operator_sym,
                             ids=[f"{c.name}"
                                  for c in operator_sym])
    def test_mul(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym)
        a.set_random()
        assert_array_almost_equal_nulp((1.2 * a).to_ndarray(),
                                       1.2 * a.to_ndarray())

    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", operator_sym,
                             ids=[f"{c.name}"
                                  for c in operator_sym])
    def test_rmul(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym)
        a.set_random()
        assert_array_almost_equal_nulp((a * -1.8).to_ndarray(),
                                       -1.8 * a.to_ndarray())

    @pytest.mark.parametrize("n_particle_op", n_particles)
    @pytest.mark.parametrize("op_sym", operator_sym,
                             ids=[f"{c.name}"
                                  for c in operator_sym])
    def test_imul(self, n_particle_op, op_sym):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = NParticleOperator(ref, n_particle_op=n_particle_op, symmetry=op_sym)
        a.set_random()

        ref = 12 * a.to_ndarray()
        a *= 12
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)
