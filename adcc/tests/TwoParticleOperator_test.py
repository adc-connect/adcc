#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import unittest
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal_nulp, assert_equal

from adcc import TwoParticleOperator, zeros_like
from adcc.NParticleOperator import product_trace, OperatorSymmetry

from .testdata_cache import testdata_cache
from itertools import combinations_with_replacement

operator_sym = [OperatorSymmetry.HERMITIAN, OperatorSymmetry.ANTIHERMITIAN,
                OperatorSymmetry.NOSYMMETRY]
op_syms_two_operators = list(combinations_with_replacement(operator_sym, 2))                


class TestTwoParticleOperator:
    @pytest.mark.parametrize("symmetry", operator_sym,
                             ids=[f"{c.name}" for c in operator_sym])
    def test_to_ndarray_herm(self, symmetry):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)

        a.set_random()

        a_oooo = a.oooo.to_ndarray()
        a_ooov = a.ooov.to_ndarray()
        a_oovv = a.oovv.to_ndarray()
        a_ovov = a.ovov.to_ndarray()
        a_ovvv = a.ovvv.to_ndarray()
        a_vvvv = a.vvvv.to_ndarray()

        no = ref.mospaces.n_orbs("o1")
        nv = ref.mospaces.n_orbs("v1")
        n_orb = no + nv

        a_full = np.zeros((n_orb, n_orb, n_orb, n_orb))
        # oo oo 
        a_full[:no, :no, :no, :no] = a_oooo
        # oo ov
        a_full[:no, :no, :no, no:] = a_ooov
        # ov ov
        a_full[:no, no:, :no, no:] = a_ovov
        # oo vv
        a_full[:no, :no, no:, no:] = a_oovv
        # ov vv
        a_full[:no, no:, no:, no:] = a_ovvv
        # vv vv
        a_full[no:, no:, no:, no:] = a_vvvv

        # if symmetry != OperatorSymmetry.ANTIHERMITIAN:
        # oo vo
        a_full[:no, :no, no:, :no] = -a_ooov.transpose((0,1,3,2))
        # ov oo
        a_full[:no, no:, :no, :no] = a_ooov.transpose((2,3,0,1))
        # vo oo
        a_full[no:, :no, :no, :no] = -a_ooov.transpose((3,2,0,1))
        # ov vo
        a_full[:no, no:, no:, :no] = -a_ovov.transpose((0,1,3,2))
        # vo vo
        a_full[no:, :no, no:, :no] = a_ovov.transpose((1,0,3,2))
        # vo ov
        a_full[no:, :no, :no, no:] = -a_ovov.transpose((1,0,2,3))
        # vv oo
        a_full[no:, no:, :no, :no] = a_oovv.transpose((2,3,0,1))
        # vo vv
        a_full[no:, :no, no:, no:] = -a_ovvv.transpose((1,0,2,3))
        # vv vo
        a_full[no:, no:, no:, :no] = -a_ovvv.transpose((2,3,1,0))
        # vv ov
        a_full[no:, no:, :no, no:] = a_ovvv.transpose((2,3,0,1))

        # else:
            # # oo vo
            # a_full[:no, :no, no:, :no] = -a_ooov.transpose((0,1,3,2))
            # # ov oo
            # a_full[:no, no:, :no, :no] = -a_ooov.transpose((2,3,0,1))
            # # vo oo
            # a_full[no:, :no, :no, :no] = a_ooov.transpose((3,2,0,1))
            # # ov vo
            # a_full[:no, no:, no:, :no] = -a_ovov.transpose((0,1,3,2))
            # # vo vo
            # a_full[no:, :no, no:, :no] = a_ovov.transpose((1,0,3,2))
            # # vo ov
            # a_full[no:, :no, :no, no:] = -a_ovov.transpose((1,0,2,3))
            # # vv oo
            # a_full[no:, no:, :no, :no] = -a_oovv.transpose((2,3,0,1))
            # # vo vv
            # a_full[no:, :no, no:, no:] = -a_ovvv.transpose((1,0,2,3))
            # # vv vo
            # a_full[no:, no:, no:, :no] = a_ovvv.transpose((2,3,1,0))
            # # vv ov
            # a_full[no:, no:, :no, no:] = -a_ovvv.transpose((2,3,0,1))

        np.testing.assert_almost_equal(a_full, a.to_ndarray(),
                                       decimal=12)

    def test_block_functions(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        # no AO transformation with only zero blocks possible
        with pytest.raises(ValueError):
            a.to_ao_basis(ref)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()
        assert a.size == a.shape[0] * a.shape[1] * a.shape[2] * a.shape[3] 
        assert not a.is_zero_block("v1o1v1v1")
        a.set_zero_block("o1o1o1o1")
        assert a.is_zero_block("o1o1o1o1")
        # access to zero blocks forbidden via block function
        with pytest.raises(KeyError):
            a.block("o1o1o1o1")
        # invalid block names
        with pytest.raises(KeyError):
            a["xyz"]
        with pytest.raises(KeyError):
            a["xyz"] = a.oo
        with pytest.raises(KeyError):
            a.set_zero_block("xyz")
        # invalid tensor shape
        with pytest.raises(ValueError):
            a.oooo = a.ovoo
        # shortcuts
        np.testing.assert_allclose(a.oooo.to_ndarray(),
                                   a["o1o1o1o1"].to_ndarray())
