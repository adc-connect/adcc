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
methods = ["adc0", "adc1", "adc2", "adc3"]


@pytest.mark.parametrize("method", methods)
@pytest.mark.parametrize("system,case,kind", cases)
class TestStateDiffDm:
    def calculate_adcn_excitation_energy(self, state):
        hf = state.reference_state
        mp = state.ground_state
        n_states = len(state.excitation_energy)
        excitation_energy = np.zeros((n_states))
        if state.method.name == "adc3":
            method = IsrMethod("isr3")
        else:
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
                for block in dens_2p.blocks:
                    # compute
                    # 1/4 [(1 - P_pq) (1 - P_rs) 1 / (n_occ - 1) <pi||ri> delta_qs]
                    #   * D^pq_rs
                    # = 1 / (n_occ - 1) <pi||ri> D^pq_rq
                    s1, s2, s3, s4 = split_spaces(block)
                    if s2 == s4:
                        eri_1p = np.einsum(
                            "piqi->pq", hf.eri(f"{s1}o1{s3}o1").to_ndarray()
                        )
                        n_occ = hf.foo.shape[1]
                        excitation_energy[es] -= 1 / (n_occ - 1) * np.einsum(
                            "pr,pqrq->",
                            eri_1p,
                            dens_2p[block].to_ndarray()
                        )
                    # and the full 2e part
                    excitation_energy[es] += 0.25 * einsum(
                        "pqrs,pqrs", dens_2p[block], hf.eri(block)
                    )
        return excitation_energy

    def test_adcn(self, method: str, system: str, case: str, kind: str):
        state = testdata_cache.adcc_states(
            system=system, method=method, kind=kind, case=case
        )
        ref = state.excitation_energy_uncorrected
        adcn = self.calculate_adcn_excitation_energy(state)
        np.testing.assert_allclose(adcn, ref, atol=1e-12)
