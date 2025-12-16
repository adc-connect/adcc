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


class TestTwoParticleOperator(unittest.TestCase):
    def test_to_ndarray_herm(self):
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
        # oo vo
        a_full[:no, :no, no:, :no] = -a_ooov.transpose((0,1,3,2))
        # ov oo
        a_full[:no, no:, :no, :no] = a_ooov.transpose((2,3,0,1))
        # vo oo
        a_full[no:, :no, :no, :no] = -a_ooov.transpose((3,2,0,1))
        # ov ov
        a_full[:no, no:, :no, no:] = a_ovov
        # ov vo
        a_full[:no, no:, no:, :no] = -a_ovov.transpose((0,1,3,2))
        # vo vo
        a_full[no:, :no, no:, :no] = a_ovov.transpose((1,0,3,2))
        # vo ov
        a_full[no:, :no, :no, no:] = -a_ovov.transpose((1,0,2,3))
        # oo vv
        a_full[:no, :no, no:, no:] = a_oovv
        # vv oo
        a_full[no:, no:, :no, :no] = a_oovv.transpose((2,3,0,1))
        # ov vv
        a_full[:no, no:, no:, no:] = a_ovvv
        # vo vv
        a_full[no:, :no, no:, no:] = -a_ovvv.transpose((1,0,2,3))
        # vv vo
        a_full[no:, no:, no:, :no] = -a_ovvv.transpose((2,3,1,0))
        # vv ov
        a_full[no:, no:, :no, no:] = a_ovvv.transpose((2,3,0,1))
        # vv vv
        a_full[no:, no:, no:, no:] = a_vvvv

        np.testing.assert_almost_equal(a_full, a.to_ndarray(),
                                       decimal=12)

    def test_to_ndarray_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        s.set_random()

        a_oooo = a.oooo.to_ndarray()
        a_ooov = a.ooov.to_ndarray()
        a_oovo = a.oovo.to_ndarray()
        a_ovoo = a.ovoo.to_ndarray()
        a_vooo = a.vooo.to_ndarray()
        a_oovv = a.oovv.to_ndarray()
        a_vvoo = a.vvoo.to_ndarray()
        a_ovov = a.ovov.to_ndarray()
        a_ovvo = a.ovvo.to_ndarray()
        a_vovo = a.vovo.to_ndarray()
        a_voov = a.voov.to_ndarray()
        a_ovvv = a.ovvv.to_ndarray()
        a_vovv = a.vovv.to_ndarray()
        a_vvov = a.vvov.to_ndarray()
        a_vvvo = a.vvvo.to_ndarray()
        a_vvvv = a.vvvv.to_ndarray()

        no = ref.mospaces.n_orbs("o1")
        nv = ref.mospaces.n_orbs("v1")
        n_orb = no + nv

        a_full = np.zeros((n_orb, n_orb, n_orb, n_orb))
        # oo oo 
        a_full[:no, :no, :no, :no] = a_oooo
        # oo ov
        a_full[:no, :no, :no, no:] = a_ooov
        # oo vo
        a_full[:no, :no, no:, :no] = a_oovo
        # ov oo
        a_full[:no, no:, :no, :no] = a_ovoo
        # vo oo
        a_full[no:, :no, :no, :no] = a_vooo
        # ov ov
        a_full[:no, no:, :no, no:] = a_ovov
        # ov vo
        a_full[:no, no:, no:, :no] = a_ovvo
        # vo vo
        a_full[no:, :no, no:, :no] = a_vovo
        # vo ov
        a_full[no:, :no, :no, no:] = a_voov
        # oo vv
        a_full[:no, :no, no:, no:] = a_oovv
        # vv oo
        a_full[no:, no:, :no, :no] = a_vvoo
        # ov vv
        a_full[:no, no:, no:, no:] = a_ovvv
        # vo vv
        a_full[no:, :no, no:, no:] = a_vovv
        # vv vo
        a_full[no:, no:, no:, :no] = a_vvvo
        # vv ov
        a_full[no:, no:, :no, no:] = a_vvov
        # vv vv
        a_full[no:, no:, no:, no:] = a_vvvv

        np.testing.assert_almost_equal(a_full, a.to_ndarray(),
                                       decimal=12)

    def test_to_ndarray_antiherm(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces,
                                 symmetry=OperatorSymmetry.ANTIHERMITIAN)
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
        # oo vo
        a_full[:no, :no, no:, :no] = -a_ooov.transpose((0,1,3,2))
        # ov oo
        a_full[:no, no:, :no, :no] = -a_ooov.transpose((2,3,0,1))
        # vo oo
        a_full[no:, :no, :no, :no] = a_ooov.transpose((3,2,0,1))
        # ov ov
        a_full[:no, no:, :no, no:] = a_ovov
        # ov vo
        a_full[:no, no:, no:, :no] = -a_ovov.transpose((0,1,3,2))
        # vo vo
        a_full[no:, :no, no:, :no] = a_ovov.transpose((1,0,3,2))
        # vo ov
        a_full[no:, :no, :no, no:] = -a_ovov.transpose((1,0,2,3))
        # oo vv
        a_full[:no, :no, no:, no:] = a_oovv
        # vv oo
        a_full[no:, no:, :no, :no] = -a_oovv.transpose((2,3,0,1))
        # ov vv
        a_full[:no, no:, no:, no:] = a_ovvv
        # vo vv
        a_full[no:, :no, no:, no:] = -a_ovvv.transpose((1,0,2,3))
        # vv vo
        a_full[no:, no:, no:, :no] = a_ovvv.transpose((2,3,1,0))
        # vv ov
        a_full[no:, no:, :no, no:] = -a_ovvv.transpose((2,3,0,1))
        # vv vv
        a_full[no:, no:, no:, no:] = a_vvvv

        np.testing.assert_almost_equal(a_full, a.to_ndarray(),
                                       decimal=12)

    def test_product_trace_herm(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")

        op_a = TwoParticleOperator(ref.mospaces, OperatorSymmetry.HERMITIAN)
        op_a.set_random()
        op_a_ao = op_a.to_ao_basis(ref)

        op_b = TwoParticleOperator(ref.mospaces, OperatorSymmetry.HERMITIAN)
        op_b.set_random()
        op_b_ao = op_b.to_ao_basis(ref)

        op_a_ao_a = op_a_ao[0].to_ndarray()
        op_a_ao_b = op_a_ao[1].to_ndarray()

        op_b_ao_a = op_b_ao[0].to_ndarray()
        op_b_ao_b = op_b_ao[1].to_ndarray()
        ref_ao = np.sum(op_a_ao_a * op_b_ao_a) + np.sum(op_a_ao_b * op_b_ao_b)

        oooo = np.sum(
            op_a.oooo.to_ndarray() * op_b.oooo.to_ndarray()
        )
        vvvv = np.sum(
            op_a.vvvv.to_ndarray() * op_b.vvvv.to_ndarray()
        )
        oovv = 2 * np.sum(
            op_a.oovv.to_ndarray() * op_b.oovv.to_ndarray()
        )
        ovov = 4 * np.sum(
            op_a.ovov.to_ndarray() * op_b.ovov.to_ndarray()
        )
        ovvv = 4 * np.sum(
            op_a.ovvv.to_ndarray() * op_b.ovvv.to_ndarray()
        )
        ooov = 4 * np.sum(
            op_a.ooov.to_ndarray() * op_b.ooov.to_ndarray()
        )

        ref_np = oooo + vvvv + oovv + ovov + ovvv + ooov

        assert ref_np == pytest.approx(product_trace(op_a, op_b))
        assert ref_np == pytest.approx(product_trace(op_b, op_a))
        # assert product_trace(op_a, op_b) == pytest.approx(ref_ao)
        # assert product_trace(op_b, op_a) == pytest.approx(ref_ao)

    def test_product_trace_antiherm(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")

        op_a = TwoParticleOperator(ref.mospaces, OperatorSymmetry.ANTIHERMITIAN)
        op_a.oooo = op_a.oooo.set_random()
        op_a.ooov = op_a.ooov.set_random()
        op_a.oovv = op_a.oovv.set_random()
        op_a.ovov = op_a.ovov.set_random()
        op_a.ovvv = op_a.ovvv.set_random()
        op_a.vvvv = op_a.vvvv.set_random()
        op_a_ao = op_a.to_ao_basis(ref)

        op_b = TwoParticleOperator(ref.mospaces, OperatorSymmetry.ANTIHERMITIAN)
        op_b.oooo = op_b.oooo.set_random()
        op_b.ooov = op_b.ooov.set_random()
        op_b.oovv = op_b.oovv.set_random()
        op_b.ovov = op_b.ovov.set_random()
        op_b.ovvv = op_b.ovvv.set_random()
        op_b.vvvv = op_b.vvvv.set_random()
        op_b_ao = op_b.to_ao_basis(ref)

        op_a_ao_a = op_a_ao[0].to_ndarray()
        op_a_ao_b = op_a_ao[1].to_ndarray()

        op_b_ao_a = op_b_ao[0].to_ndarray()
        op_b_ao_b = op_b_ao[1].to_ndarray()
        ref_ao = np.sum(op_a_ao_a * op_b_ao_a) + np.sum(op_a_ao_b * op_b_ao_b)

        oooo = np.sum(
            op_a.oooo.to_ndarray() * op_b.oooo.to_ndarray()
        )
        vvvv = np.sum(
            op_a.vvvv.to_ndarray() * op_b.vvvv.to_ndarray()
        )
        oovv = 2 * np.sum(
            op_a.oovv.to_ndarray() * op_b.oovv.to_ndarray()
        )
        ovov = 4 * np.sum(
            op_a.ovov.to_ndarray() * op_b.ovov.to_ndarray()
        )
        ovvv = 4 * np.sum(
            op_a.ovvv.to_ndarray() * op_b.ovvv.to_ndarray()
        )
        ooov = 4 * np.sum(
            op_a.ooov.to_ndarray() * op_b.ooov.to_ndarray()
        )

        ref_np = oooo + vvvv + oovv + ovov + ovvv + ooov

        assert ref_np == pytest.approx(product_trace(op_a, op_b))
        assert ref_np == pytest.approx(product_trace(op_b, op_a))
        # assert product_trace(op_a, op_b) == pytest.approx(ref_ao)
        # assert product_trace(op_b, op_a) == pytest.approx(ref_ao)

    def test_product_trace_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")

        op_a = TwoParticleOperator(ref.mospaces, OperatorSymmetry.NOSYMMETRY)
        op_a.oooo = op_a.oooo.set_random()
        op_a.ooov = op_a.ooov.set_random()
        op_a.oovv = op_a.oovv.set_random()
        op_a.ovov = op_a.ovov.set_random()
        op_a.ovvv = op_a.ovvv.set_random()
        op_a.vvvv = op_a.vvvv.set_random()
        op_a_ao = op_a.to_ao_basis(ref)

        op_b = TwoParticleOperator(ref.mospaces, OperatorSymmetry.NOSYMMETRY)
        op_b.oooo = op_b.oooo.set_random()
        op_b.ooov = op_b.ooov.set_random()
        op_b.oovv = op_b.oovv.set_random()
        op_b.ovov = op_b.ovov.set_random()
        op_b.ovvv = op_b.ovvv.set_random()
        op_b.vvvv = op_b.vvvv.set_random()
        op_b_ao = op_b.to_ao_basis(ref)

        op_a_ao_a = op_a_ao[0].to_ndarray()
        op_a_ao_b = op_a_ao[1].to_ndarray()

        op_b_ao_a = op_b_ao[0].to_ndarray()
        op_b_ao_b = op_b_ao[1].to_ndarray()
        ref_ao = np.sum(op_a_ao_a * op_b_ao_a) + np.sum(op_a_ao_b * op_b_ao_b)

        oooo = np.sum(
            op_a.oooo.to_ndarray() * op_b.oooo.to_ndarray()
        )
        vvvv = np.sum(
            op_a.vvvv.to_ndarray() * op_b.vvvv.to_ndarray()
        )
        oovv = np.sum(
            op_a.oovv.to_ndarray() * op_b.oovv.to_ndarray()
        )
        ovov = np.sum(
            op_a.ovov.to_ndarray() * op_b.ovov.to_ndarray()
        )
        ovvv = np.sum(
            op_a.ovvv.to_ndarray() * op_b.ovvv.to_ndarray()
        )
        ooov = np.sum(
            op_a.ooov.to_ndarray() * op_b.ooov.to_ndarray()
        )

        ref_np = oooo + vvvv + oovv + ovov + ovvv + ooov

        assert pytest.approx(ref_ao) == vvvv

        assert ref_np == pytest.approx(product_trace(op_a, op_b))
        assert ref_np == pytest.approx(product_trace(op_b, op_a))
        assert product_trace(op_a, op_b) == pytest.approx(ref_ao)
        assert product_trace(op_b, op_a) == pytest.approx(ref_ao)

    def test_product_trace_herm_antiherm(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")

        op_a = TwoParticleOperator(ref.mospaces, OperatorSymmetry.HERMITIAN)
        op_a.oooo = op_a.oooo.set_random()
        op_a.ooov = op_a.ooov.set_random()
        op_a.oovv = op_a.oovv.set_random()
        op_a.ovov = op_a.ovov.set_random()
        op_a.ovvv = op_a.ovvv.set_random()
        op_a.vvvv = op_a.vvvv.set_random()
        op_a_ao = op_a.to_ao_basis(ref)

        op_b = TwoParticleOperator(ref.mospaces, OperatorSymmetry.ANTIHERMITIAN)
        op_b.oooo = op_b.oooo.set_random()
        op_b.ooov = op_b.ooov.set_random()
        op_b.oovv = op_b.oovv.set_random()
        op_b.ovov = op_b.ovov.set_random()
        op_b.ovvv = op_b.ovvv.set_random()
        op_b.vvvv = op_b.vvvv.set_random()
        op_b_ao = op_b.to_ao_basis(ref)

        op_a_ao_a = op_a_ao[0].to_ndarray()
        op_a_ao_b = op_a_ao[1].to_ndarray()

        op_b_ao_a = op_b_ao[0].to_ndarray()
        op_b_ao_b = op_b_ao[1].to_ndarray()
        ref_ao = np.sum(op_a_ao_a * op_b_ao_a) + np.sum(op_a_ao_b * op_b_ao_b)

        oooo = np.sum(
            op_a.oooo.to_ndarray() * op_b.oooo.to_ndarray()
        )
        vvvv = np.sum(
            op_a.vvvv.to_ndarray() * op_b.vvvv.to_ndarray()
        )
        ovov = 4 * np.sum(
            op_a.ovov.to_ndarray() * op_b.ovov.to_ndarray()
        )

        ref_np = oooo + vvvv + ovov

        assert ref_np == pytest.approx(product_trace(op_a, op_b))
        assert ref_np == pytest.approx(product_trace(op_b, op_a))
        # assert product_trace(op_a, op_b) == pytest.approx(ref_ao)
        # assert product_trace(op_b, op_a) == pytest.approx(ref_ao)

    #
    # Test operators
    #
    def test_copy(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        op_a = TwoParticleOperator(ref, OperatorSymmetry.HERMITIAN)
        op_a.oooo = op_a.oooo.set_random()
        op_a.ooov = op_a.ooov.set_random()
        op_a.oovv = op_a.oovv.set_random()
        op_a.ovov = op_a.ovov.set_random()
        op_a.ovvv = op_a.ovvv.set_random()
        op_a.vvvv = op_a.vvvv.set_random()
        cpy = op_a.copy()

        assert cpy.blocks == op_a.blocks
        assert cpy.blocks_nonzero == op_a.blocks_nonzero
        assert cpy.reference_state == op_a.reference_state
        assert cpy.mospaces == op_a.mospaces

        for b in op_a.blocks:
            assert cpy.is_zero_block(b) == op_a.is_zero_block(b)
            if not op_a.is_zero_block(b):
                assert_equal(cpy.block(b).to_ndarray(),
                             op_a.block(b).to_ndarray())
                assert cpy.block(b) is not op_a.block(b)

    def test_copy_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        op_a = TwoParticleOperator(ref, OperatorSymmetry.NOSYMMETRY)
        op_a.oooo = op_a.oooo.set_random()
        op_a.ooov = op_a.ooov.set_random()
        op_a.oovv = op_a.oovv.set_random()
        op_a.ovov = op_a.ovov.set_random()
        op_a.ovvv = op_a.ovvv.set_random()
        op_a.vvvv = op_a.vvvv.set_random()
        cpy = op_a.copy()

        assert cpy.blocks == op_a.blocks
        assert cpy.blocks_nonzero == op_a.blocks_nonzero
        assert cpy.reference_state == op_a.reference_state
        assert cpy.mospaces == op_a.mospaces

        for b in op_a.blocks:
            assert cpy.is_zero_block(b) == op_a.is_zero_block(b)
            if not op_a.is_zero_block(b):
                assert_equal(cpy.block(b).to_ndarray(),
                             op_a.block(b).to_ndarray())
                assert cpy.block(b) is not op_a.block(b)

    def test_add(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref, OperatorSymmetry.NOSYMMETRY)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovo = a.oovo.set_random()
        a.ovoo = a.ovoo.set_random()
        a.vooo = a.vooo.set_random()
        a.oovv = a.oovv.set_random()
        a.vvoo = a.vvoo.set_random()
        a.ovov = a.ovov.set_random()
        a.voov = a.voov.set_random()
        a.ovvo = a.ovvo.set_random()
        a.vovo = a.vovo.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vovv = a.vovv.set_random()
        a.vvov = a.vvov.set_random()
        a.vvvo = a.vvvo.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref, OperatorSymmetry.HERMITIAN)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovv = b.oovv.set_random()
        b.ovov = b.ovov.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vvvv = b.vvvv.set_random()

        assert_array_almost_equal_nulp((a + b).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())
        assert_array_almost_equal_nulp((b + a).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())

    def test_add_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref, OperatorSymmetry.HERMITIAN)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref, OperatorSymmetry.NOSYMMETRY)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovo = b.oovo.set_random()
        # b.ovoo = b.ovoo.set_random()
        # b.vooo = b.vooo.set_random()
        # b.oovv = b.oovv.set_random()
        # b.vvoo = b.vvoo.set_random()
        # b.ovov = b.ovov.set_random()
        # b.voov = b.voov.set_random()
        # b.ovvo = b.ovvo.set_random()
        # b.vovo = b.vovo.set_random()
        # b.ovvv = b.ovvv.set_random()
        # b.vovv = b.vovv.set_random()
        # b.vvov = b.vvov.set_random()
        # b.vvvo = b.vvvo.set_random()
        # b.vvvv = b.vvvv.set_random()

        assert_array_almost_equal_nulp((a + b).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())
        assert_array_almost_equal_nulp((b + a).to_ndarray(),
                                       b.to_ndarray() + a.to_ndarray())

    def test_add_nosym_antiherm(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref, OperatorSymmetry.NOSYMMETRY)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovo = a.oovo.set_random()
        a.ovoo = a.ovoo.set_random()
        a.vooo = a.vooo.set_random()
        a.oovv = a.oovv.set_random()
        a.vvoo = a.vvoo.set_random()
        a.ovov = a.ovov.set_random()
        a.voov = a.voov.set_random()
        a.ovvo = a.ovvo.set_random()
        a.vovo = a.vovo.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vovv = a.vovv.set_random()
        a.vvov = a.vvov.set_random()
        a.vvvo = a.vvvo.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref, OperatorSymmetry.ANTIHERMITIAN)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovv = b.oovv.set_random()
        b.ovov = b.ovov.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vvvv = b.vvvv.set_random()

        assert_array_almost_equal_nulp((a + b).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())
        assert_array_almost_equal_nulp((b + a).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())

    def test_add_herm_antiherm(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref, OperatorSymmetry.HERMITIAN)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref, OperatorSymmetry.HERMITIAN)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovv = b.oovv.set_random()
        b.ovov = b.ovov.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vvvv = b.vvvv.set_random()

        assert_array_almost_equal_nulp((a + b).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())
        assert_array_almost_equal_nulp((b + a).to_ndarray(),
                                       a.to_ndarray() + b.to_ndarray())

    def test_iadd(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovo = a.oovo.set_random()
        a.ovoo = a.ovoo.set_random()
        a.vooo = a.vooo.set_random()
        a.oovv = a.oovv.set_random()
        a.vvoo = a.vvoo.set_random()
        a.ovov = a.ovov.set_random()
        a.voov = a.voov.set_random()
        a.ovvo = a.ovvo.set_random()
        a.vovo = a.vovo.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vovv = a.vovv.set_random()
        a.vvov = a.vvov.set_random()
        a.vvvo = a.vvvo.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovo = b.oovo.set_random()
        b.ovoo = b.ovoo.set_random()
        b.vooo = b.vooo.set_random()
        b.oovv = b.oovv.set_random()
        b.vvoo = b.vvoo.set_random()
        b.ovov = b.ovov.set_random()
        b.voov = b.voov.set_random()
        b.ovvo = b.ovvo.set_random()
        b.vovo = b.vovo.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vovv = b.vovv.set_random()
        b.vvov = b.vvov.set_random()
        b.vvvo = b.vvvo.set_random()
        b.vvvv = b.vvvv.set_random()

        ref = a.to_ndarray() + b.to_ndarray()
        a += b
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    def test_sub(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovv = b.oovv.set_random()
        b.ovov = b.ovov.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vvvv = b.vvvv.set_random()

        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       a.to_ndarray() - b.to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       b.to_ndarray() - a.to_ndarray())
        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       (a + (-1 * b)).to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       (b + (-1 * a)).to_ndarray())

    def test_sub_nosym(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovo = b.oovo.set_random()
        b.ovoo = b.ovoo.set_random()
        b.vooo = b.vooo.set_random()
        b.oovv = b.oovv.set_random()
        b.vvoo = b.vvoo.set_random()
        b.ovov = b.ovov.set_random()
        b.voov = b.voov.set_random()
        b.ovvo = b.ovvo.set_random()
        b.vovo = b.vovo.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vovv = b.vovv.set_random()
        b.vvov = b.vvov.set_random()
        b.vvvo = b.vvvo.set_random()
        b.vvvv = b.vvvv.set_random()

        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       a.to_ndarray() - b.to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       b.to_ndarray() - a.to_ndarray())
        assert_array_almost_equal_nulp((a - b).to_ndarray(),
                                       (a + (-1.0 * b)).to_ndarray())
        assert_array_almost_equal_nulp((b - a).to_ndarray(),
                                       (b + (-1.0 * a)).to_ndarray())

    def test_isub(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovo = a.oovo.set_random()
        a.ovoo = a.ovoo.set_random()
        a.vooo = a.vooo.set_random()
        a.oovv = a.oovv.set_random()
        a.vvoo = a.vvoo.set_random()
        a.ovov = a.ovov.set_random()
        a.voov = a.voov.set_random()
        a.ovvo = a.ovvo.set_random()
        a.vovo = a.vovo.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vovv = a.vovv.set_random()
        a.vvov = a.vvov.set_random()
        a.vvvo = a.vvvo.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovo = b.oovo.set_random()
        b.ovoo = b.ovoo.set_random()
        b.vooo = b.vooo.set_random()
        b.oovv = b.oovv.set_random()
        b.vvoo = b.vvoo.set_random()
        b.ovov = b.ovov.set_random()
        b.voov = b.voov.set_random()
        b.ovvo = b.ovvo.set_random()
        b.vovo = b.vovo.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vovv = b.vovv.set_random()
        b.vvov = b.vvov.set_random()
        b.vvvo = b.vvvo.set_random()
        b.vvvv = b.vvvv.set_random()

        ref = a.to_ndarray() - b.to_ndarray()
        a -= b
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    def test_isub_herm(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()

        b = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        b.oooo = b.oooo.set_random()
        b.ooov = b.ooov.set_random()
        b.oovv = b.oovv.set_random()
        b.ovov = b.ovov.set_random()
        b.ovvv = b.ovvv.set_random()
        b.vvvv = b.vvvv.set_random()

        ref = a.to_ndarray() - b.to_ndarray()
        a -= b
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)

    def test_mul(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()
        assert_array_almost_equal_nulp((1.2 * a).to_ndarray(),
                                       1.2 * a.to_ndarray())

    def test_rmul(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovv = a.oovv.set_random()
        a.ovov = a.ovov.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vvvv = a.vvvv.set_random()
        assert_array_almost_equal_nulp((a * -1.8).to_ndarray(),
                                       -1.8 * a.to_ndarray())

    def test_imul(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        a.oooo = a.oooo.set_random()
        a.ooov = a.ooov.set_random()
        a.oovo = a.oovo.set_random()
        a.ovoo = a.ovoo.set_random()
        a.vooo = a.vooo.set_random()
        a.oovv = a.oovv.set_random()
        a.vvoo = a.vvoo.set_random()
        a.ovov = a.ovov.set_random()
        a.voov = a.voov.set_random()
        a.ovvo = a.ovvo.set_random()
        a.vovo = a.vovo.set_random()
        a.ovvv = a.ovvv.set_random()
        a.vovv = a.vovv.set_random()
        a.vvov = a.vvov.set_random()
        a.vvvo = a.vvvo.set_random()
        a.vvvv = a.vvvv.set_random()

        ref = 12 * a.to_ndarray()
        a *= 12
        assert_array_almost_equal_nulp(a.to_ndarray(), ref)

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
