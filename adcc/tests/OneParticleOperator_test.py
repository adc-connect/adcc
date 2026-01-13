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
import pytest
import numpy as np

from adcc import OneParticleOperator
from adcc.NParticleOperator import OperatorSymmetry

from .testdata_cache import testdata_cache
from . import testcases
from adcc.backends import run_hf
from itertools import combinations_with_replacement


operator_sym = [OperatorSymmetry.HERMITIAN, OperatorSymmetry.ANTIHERMITIAN,
                OperatorSymmetry.NOSYMMETRY]
op_syms_two_operators = list(combinations_with_replacement(operator_sym, 2))


class TestOneParticleOperator:
    def test_to_ao_basis_hermitian(self):
        system = "h2o_sto3g"
        system: testcases.TestCase = testcases.get_by_filename(system).pop()
        scfres = run_hf("pyscf", system.xyz, system.basis)
        ref = adcc.ReferenceState(scfres)
        dipx_mo = ref.operators.electric_dipole[0]
        dipx_ao_ref = ref.operators.provider_ao.electric_dipole[0]

        dipx_ao_a = dipx_mo.to_ao_basis(ref)[0].to_ndarray()
        dipx_ao_b = dipx_mo.to_ao_basis(ref)[1].to_ndarray()

        np.testing.assert_allclose(dipx_ao_ref, dipx_ao_a, atol=1e-12)
        np.testing.assert_allclose(dipx_ao_ref, dipx_ao_b, atol=1e-12)

    def test_to_ao_basis_antihermitian(self):
        system = "h2o_sto3g"
        system: testcases.TestCase = testcases.get_by_filename(system).pop()
        scfres = run_hf("pyscf", system.xyz, system.basis)
        ref = adcc.ReferenceState(scfres)
        dipx_mo = ref.operators.magnetic_dipole()[0]
        dipx_ao_ref = ref.operators.provider_ao.magnetic_dipole()[0]

        dipx_ao_a = dipx_mo.to_ao_basis(ref)[0].to_ndarray()
        dipx_ao_b = dipx_mo.to_ao_basis(ref)[1].to_ndarray()

        np.testing.assert_allclose(dipx_ao_ref, dipx_ao_a, atol=1e-12)
        np.testing.assert_allclose(dipx_ao_ref, dipx_ao_b, atol=1e-12)

    @pytest.mark.parametrize("symmetry", operator_sym,
                             ids=[f"{c.name}" for c in operator_sym])
    def test_to_ndarray(self, symmetry):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        dm = OneParticleOperator(ref.mospaces, symmetry=symmetry)
        dm.set_random()

        dm_oo = dm.oo.to_ndarray()
        dm_ov = dm.ov.to_ndarray()
        dm_vv = dm.vv.to_ndarray()

        if symmetry == OperatorSymmetry.HERMITIAN:
            dm_vo = dm_ov.conj().T
        elif symmetry == OperatorSymmetry.ANTIHERMITIAN:
            dm_vo = -dm_ov.conj().T
        else:
            dm_vo = dm.vo.to_ndarray()

        dm_o = np.hstack((dm_oo, dm_ov))
        dm_v = np.hstack((dm_vo, dm_vv))
        dm_full = np.vstack((dm_o, dm_v))

        np.testing.assert_almost_equal(dm_full, dm.to_ndarray(), decimal=12)

    def test_block_functions(self):
        ref = testdata_cache.refstate("h2o_sto3g", "gen")
        a = OneParticleOperator(ref.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        # no AO transformation with only zero blocks possible
        with pytest.raises(ValueError):
            a.to_ao_basis(ref)
        a.set_random()
        assert a.size == a.shape[0] * a.shape[1]
        assert not a.is_zero_block("v1o1")
        a.set_zero_block("o1o1")
        assert a.is_zero_block("o1o1")
        # access to zero blocks forbidden via block function
        with pytest.raises(KeyError):
            a.block("o1o1")
        # invalid block names
        with pytest.raises(KeyError):
            a["xyz"]
        with pytest.raises(KeyError):
            a["xyz"] = a.oo
        with pytest.raises(KeyError):
            a.set_zero_block("xyz")
        # invalid tensor shape
        with pytest.raises(ValueError):
            a.oo = a.ov
        # shortcuts
        np.testing.assert_allclose(a.oo.to_ndarray(),
                                   a["o1o1"].to_ndarray())
