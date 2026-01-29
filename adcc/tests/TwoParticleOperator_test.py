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
import numpy as np

import adcc
from adcc import TwoParticleOperator
from adcc.NParticleOperator import OperatorSymmetry

from .testdata_cache import testdata_cache
from . import testcases
from adcc.backends import run_hf
from adcc.OperatorIntegrals import replicate_ao_block, transform_operator_ao2mo
from itertools import combinations_with_replacement

operator_sym = [OperatorSymmetry.HERMITIAN, OperatorSymmetry.ANTIHERMITIAN,
                OperatorSymmetry.NOSYMMETRY]
op_syms_two_operators = list(combinations_with_replacement(operator_sym, 2))


class TestTwoParticleOperator:
    @pytest.mark.parametrize("symmetry", operator_sym,
                             ids=[f"{c.name}" for c in operator_sym])
    def test_to_ndarray(self, symmetry):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=symmetry)

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

        # oo vo
        a_full[:no, :no, no:, :no] = -a_ooov.transpose((0, 1, 3, 2))
        # ov vo
        a_full[:no, no:, no:, :no] = -a_ovov.transpose((0, 1, 3, 2))
        # vo vo
        a_full[no:, :no, no:, :no] = a_ovov.transpose((1, 0, 3, 2))
        # vo ov
        a_full[no:, :no, :no, no:] = -a_ovov.transpose((1, 0, 2, 3))
        # vo vv
        a_full[no:, :no, no:, no:] = -a_ovvv.transpose((1, 0, 2, 3))

        if symmetry == OperatorSymmetry.HERMITIAN:
            # ov oo
            a_full[:no, no:, :no, :no] = a_ooov.transpose((2, 3, 0, 1))
            # vo oo
            a_full[no:, :no, :no, :no] = -a_ooov.transpose((3, 2, 0, 1))
            # vv oo
            a_full[no:, no:, :no, :no] = a_oovv.transpose((2, 3, 0, 1))
            # vv vo
            a_full[no:, no:, no:, :no] = -a_ovvv.transpose((2, 3, 1, 0))
            # vv ov
            a_full[no:, no:, :no, no:] = a_ovvv.transpose((2, 3, 0, 1))

        elif symmetry == OperatorSymmetry.ANTIHERMITIAN:
            # ov oo
            a_full[:no, no:, :no, :no] = -a_ooov.transpose((2, 3, 0, 1))
            # vo oo
            a_full[no:, :no, :no, :no] = a_ooov.transpose((3, 2, 0, 1))
            # vv oo
            a_full[no:, no:, :no, :no] = -a_oovv.transpose((2, 3, 0, 1))
            # vv vo
            a_full[no:, no:, no:, :no] = a_ovvv.transpose((2, 3, 1, 0))
            # vv ov
            a_full[no:, no:, :no, no:] = -a_ovvv.transpose((2, 3, 0, 1))

        else:
            a_ovoo = a.ovoo.to_ndarray()
            a_vvov = a.vvov.to_ndarray()
            a_vvoo = a.vvoo.to_ndarray()
            # ov oo
            a_full[:no, no:, :no, :no] = a_ovoo
            # vo oo
            a_full[no:, :no, :no, :no] = -a_ovoo.transpose((1, 0, 2, 3))
            # vv oo
            a_full[no:, no:, :no, :no] = a_vvoo
            # vv ov
            a_full[no:, no:, :no, no:] = a_vvov
            # vv vo
            a_full[no:, no:, no:, :no] = -a_vvov.transpose((0, 1, 3, 2))

        np.testing.assert_almost_equal(a_full, a.to_ndarray(),
                                       decimal=12)

    def test_block_functions(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = TwoParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        # no AO transformation with only zero blocks possible
        with pytest.raises(NotImplementedError):
            a.to_ao_basis(ref)
        a.set_random()
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
            a["xyz"] = a.oooo
        with pytest.raises(KeyError):
            a.set_zero_block("xyz")
        # invalid tensor shape
        with pytest.raises(ValueError):
            a.oooo = a.ovoo
        # shortcuts
        np.testing.assert_allclose(a.oooo.to_ndarray(),
                                   a["o1o1o1o1"].to_ndarray())

    def test_import_2p_op(self):
        system = "h2o_sto3g"
        system: testcases.TestCase = testcases.get_by_filename(system).pop()
        scfres = run_hf("pyscf", system.xyz, system.basis)
        ref = adcc.ReferenceState(scfres)

        # get AO integral in physicist notation
        int2e = scfres.mol.intor('int2e', comp=1, aosym=1).transpose((0, 2, 1, 3))
        # from the integrals construct a TwoParticleOperator
        dip_bb = replicate_ao_block(ref.mospaces, int2e,
                                    symmetry=OperatorSymmetry.HERMITIAN)
        eri_operator = TwoParticleOperator(ref, symmetry=OperatorSymmetry.HERMITIAN)
        transform_operator_ao2mo(dip_bb, eri_operator, ref.orbital_coefficients,
                                 ref.conv_tol)

        # compare constructed TwoParticleOperator with ERIs
        for block in eri_operator.blocks:
            np.testing.assert_allclose(ref.eri(block).to_ndarray(),
                                       eri_operator[block].to_ndarray(), atol=1e-12)
