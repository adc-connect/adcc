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
import numpy as np
import pytest

from adcc.AdcMethod import IsrMethod
from adcc.functions import einsum
from adcc.adc_pp.state_diffdm_2p import state_diffdm_2p
from adcc.adc_pp.state_diffdm import state_diffdm
from adcc.MoSpaces import split_spaces

from .. import testcases
from ..testdata_cache import testdata_cache


test_cases = testcases.get_by_filename("h2o_sto3g", "cn_sto3g")
cases = [(case.file_name, c, kind)
         for case in test_cases for c in ["gen"] for kind in case.kinds.pp]
methods = [
    ("adc0", None),
    ("adc1", None),
    ("adc2", None),
    ("adc3", 3),
]


@pytest.mark.parametrize("adc_method, isr_order", methods)
@pytest.mark.parametrize("system,case,kind", cases)
class TestStateDiffDm:
    def calculate_adcn_excitation_energy(self, state):
        hf = state.reference_state
        mp = state.ground_state
        n_states = len(state.excitation_energy)
        excitation_energy = np.zeros((n_states))

        method = state.property_method
        level = method.level.to_int()

        method_order_minus_one = None
        if level - 1 >= 0:
            method_order_minus_one = IsrMethod("isr" + str(level - 1))

        for es in range(n_states):
            evec = state.excitation_vector[es]
            dens_1p = state_diffdm(method, mp, evec)

            # one particle
            # fock operator part
            excitation_energy[es] = einsum("pq,pq", hf.foo, dens_1p.oo)
            excitation_energy[es] += einsum("pq,pq", hf.fvv, dens_1p.vv)

            if method_order_minus_one is not None:
                # two particle part
                dens_2p = state_diffdm_2p(method_order_minus_one, mp, evec)
                dens_1p = state_diffdm(method_order_minus_one, mp, evec)
                for block in dens_1p.blocks:
                    s1, s2 = split_spaces(block)
                    eri_1p = np.einsum(
                        "piqi->pq", hf.eri(f"{s1}o1{s2}o1").to_ndarray()
                    )
                    excitation_energy[es] -= np.einsum(
                        "pq,pq->",
                        eri_1p,
                        dens_1p[block].to_ndarray()
                    )
                for block in dens_2p.blocks:
                    # the full 2e part
                    excitation_energy[es] += 0.25 * einsum(
                        "pqrs,pqrs", dens_2p[block], hf.eri(block)
                    )

                # reconstruct the 1-particle density from the 2-particle density
                n_occ = hf.foo.shape[1]
                d_oo = 1 / (n_occ - 1) * (
                    np.einsum("ikjk->ij", dens_2p.oooo.to_ndarray())
                    + np.einsum("icjc->ij", dens_2p.ovov.to_ndarray())
                )
                np.testing.assert_allclose(
                    dens_1p.oo.to_ndarray(), d_oo, atol=1e-14
                )
                d_ov = 1 / (n_occ - 1) * (
                    np.einsum("ikak->ia", dens_2p.oovo.to_ndarray())
                    + np.einsum("icac->ia", dens_2p.ovvv.to_ndarray())
                )
                np.testing.assert_allclose(
                    dens_1p.ov.to_ndarray(), d_ov, atol=1e-14
                )
                d_vo = 1 / (n_occ - 1) * (
                    np.einsum("ikak->ia", dens_2p.vooo.to_ndarray())
                    + np.einsum("icac->ia", dens_2p.vvov.to_ndarray())
                )
                np.testing.assert_allclose(
                    dens_1p.vo.to_ndarray(), d_vo, atol=1e-14
                )
                d_vv = 1 / (n_occ - 1) * (
                    np.einsum("ikak->ia", dens_2p.vovo.to_ndarray())
                    + np.einsum("icac->ia", dens_2p.vvvv.to_ndarray())
                )
                np.testing.assert_allclose(
                    dens_1p.vv.to_ndarray(), d_vv, atol=1e-14
                )
        return excitation_energy

    def test_adcn(self, adc_method: str, system: str, isr_order: str,
                  case: str, kind: str):
        state = testdata_cache.adcc_states(
            system=system, method=adc_method, kind=kind, case=case,
            isr_order=isr_order
        )
        ref = state.excitation_energy_uncorrected
        adcn = self.calculate_adcn_excitation_energy(state)
        np.testing.assert_allclose(adcn, ref, atol=1e-14)
