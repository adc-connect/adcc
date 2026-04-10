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

from adcc import block as b
from adcc.AdcMethod import IsrMethod
from adcc.functions import evaluate, einsum
from adcc.OneParticleDensity import OneParticleDensity
from adcc.NParticleOperator import OperatorSymmetry
from adcc.adc_pp.state_diffdm_2p import state_diffdm_2p
from adcc.adc_pp.state_diffdm import state_diffdm
from adcc.adc_pp.util import check_doubles_amplitudes, check_singles_amplitudes
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
            # TODO switch to ISR(3) implemntation
            # so we don't forget to switch to the actual implementation
            with pytest.raises(NotImplementedError):
                method = IsrMethod("isr3")
            method = IsrMethod("isr3", validate_level=False)
        else:
            method = state.property_method
        level = method.level

        method_order_minus_one = None
        if level - 1 >= 0:
            method_order_minus_one = IsrMethod("isr" + str(level - 1))

        for es in range(n_states):
            evec = state.excitation_vector[es]
            # TODO switch to ISR(3) implemntation
            if method.level == 3:
                # so we don't forget to switch to the actual implementation
                with pytest.raises(NotImplementedError):
                    state_diffdm(method, mp, evec)
                dens_1p = self.state_diffdm_adc3(mp, evec)
            else:
                dens_1p = state_diffdm(method, mp, evec)

            # one particle
            # fock operator part
            excitation_energy[es] = einsum("pq,pq", hf.foo, dens_1p.oo)
            excitation_energy[es] += einsum("pq,pq", hf.fvv, dens_1p.vv)

            if method_order_minus_one is not None:
                # two particle part
                dens_2p = state_diffdm_2p(method_order_minus_one, mp, evec)
                # go for ISR(1)-d for ADC(2)
                if method_order_minus_one.level == 1:
                    dens_2p.ooov += (
                        - 2.0 * einsum("kb,ijab->ijka", evec.ph, evec.pphh)
                    )
                    dens_2p.ovvv += (
                        - 2.0 * einsum("ja,ijbc->iabc", evec.ph, evec.pphh)
                    )
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

    def mp3_diffdm(self, mp) -> OneParticleDensity:
        # Only calculate the oo and vv block since the hf.fov block is zero anyways.
        hf = mp.reference_state
        ret = OneParticleDensity(hf.mospaces, symmetry=OperatorSymmetry.HERMITIAN)

        mp2_dm = mp.mp2_diffdm
        ret.oo = (
            mp2_dm.oo  # 2nd order
            # 3rd order
            - einsum("ikab,jkab->ij", mp.t2oo, mp.td2(b.oovv)).symmetrise(0, 1)
        )
        ret.vv = (
            mp2_dm.vv  # 2nd order
            # 3rd order
            + einsum("ijac,ijbc->ab", mp.t2oo, mp.td2(b.oovv)).symmetrise(0, 1)
        )
        return evaluate(ret)

    def state_diffdm_adc3(self, mp, amplitude) -> OneParticleDensity:
        # Only calculate the oo and vv block since the hf.fov block is zero anyways.
        check_singles_amplitudes([b.o, b.v], amplitude)
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
        hf = mp.reference_state
        dm = OneParticleDensity(hf.mospaces, symmetry=OperatorSymmetry.HERMITIAN)

        ur1, ur2 = amplitude.ph, amplitude.pphh

        t2_1 = mp.t2oo
        t2_2 = mp.td2(b.oovv)

        p0_2 = mp.mp2_diffdm
        p0_2_ov = p0_2.ov
        p0 = self.mp3_diffdm(mp)
        p0_oo, p0_vv = p0.oo, p0.vv

        # NOTE: In the equations below only MP densities have been factored.
        #       They can be further simplified by factoring 0'th order
        #       contributions,
        #       ru1 and a similar intermediate with td2 (t2_2)

        # The scaling in the comments is given as: [comp_scaling] / [mem_scaling]
        dm.oo += (
            - 1 * einsum('ia,ja->ij', ur1, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
            - 2 * einsum('ikab,jkab->ij', ur2, ur2)  # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum('jlbc,ilbc->ij', t2_1,  # N^5: O^3V^2 / N^4: O^2V^2
                           einsum('ikbc,kl->ilbc', t2_1,
                                  einsum('ka,la->kl', ur1, ur1)))
            - 1 * einsum('jlbc,ilbc->ij', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                         einsum('ilab,ac->ilbc', t2_1,
                                einsum('ka,kc->ac', ur1, ur1)))
            - 1 * einsum('ia,ja->ij',
                         einsum('kb,ikab->ia', ur1, t2_1),
                         einsum('lc,jlac->ja',
                                ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * (  # factor 2: revert the 1/2 from symmetrise
                - 2 * einsum('jb,ib->ij', p0_2_ov,  # N^4: O^2V^2 / N^4: O^2V^2
                             einsum('ka,ikab->ib', ur1, ur2))
                - 0.5 * einsum('jk,ik->ij', p0_oo,  # N^3: O^2V^1 / N^2: O^1V^1
                               einsum('ia,ka->ik', ur1, ur1))
                + 1 * einsum('ib,jb->ij',  # N^4: O^2V^2 / N^4: O^2V^2
                             einsum('ka,ikab->ib', ur1, t2_1),
                             einsum('lc,jlbc->jb', ur1, t2_2))
                + 1 * einsum('jkac,ikac->ij', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                             einsum('ikbc,ab->ikac', t2_1,
                                    einsum('la,lb->ab', ur1, ur1)))
                + 0.5 * einsum('jlbc,ilbc->ij', t2_2,  # N^5: O^3V^2 / N^4: O^2V^2
                               einsum('ikbc,kl->ilbc', t2_1,
                                      einsum('ka,la->kl', ur1, ur1)))
                + 0.5 * einsum('ic,jc->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('jlbc,lb->jc', t2_1,
                                      einsum('ka,klab->lb', ur1, t2_1)))
                + 0.5 * einsum('ib,jb->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('jkbc,kc->jb', t2_2,
                                      einsum('la,klac->kc', ur1, t2_1)))
                + 0.5 * einsum('ib,jb->ij', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('jkbc,kc->jb', t2_1,
                                      einsum('la,klac->kc', ur1, t2_2)))
            )
        ).symmetrise()
        dm.vv = (
            + 1 * einsum('ia,ib->ab', ur1, ur1)  # N^3: O^1V^2 / N^2: V^2
            + 2 * einsum('ijac,ijbc->ab', ur2, ur2)  # N^5: O^2V^3 / N^4: O^2V^2
            + 1 * einsum('ka,kb->ab',
                         einsum('ic,ikac->ka', ur1, t2_1),
                         einsum('jd,jkbd->kb',
                                ur1, t2_1))  # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum('jkbd,jkad->ab', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                         einsum('ikad,ij->jkad',
                                t2_1, einsum('ic,jc->ij', ur1, ur1)))
            - 0.5 * einsum('jkbd,jkad->ab', t2_1,  # N^5: O^2V^3 / N^4: O^2V^2
                           einsum('jkac,cd->jkad',
                                  t2_1, einsum('ic,id->cd', ur1, ur1)))
            + 2 * (  # factor 2: revert the 1/2 from symmetrise
                + 2 * einsum('jb,ja->ab', p0_2_ov,  # N^4: O^2V^2 / N^4: O^2V^2
                             einsum('ic,ijac->ja', ur1, ur2))
                - 0.5 * einsum('ia,ib->ab', ur1,  # N^3: O^1V^2 / N^2: V^2
                               einsum('ic,bc->ib', ur1, p0_vv))
                + 1 * einsum('jkbc,jkac->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                             einsum('ijac,ik->jkac', t2_1,
                                    einsum('id,kd->ik', ur1, ur1)))
                + 0.5 * einsum('ia,ib->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('ikbd,kd->ib', t2_1,
                                      einsum('jc,jkcd->kd', ur1, t2_1)))
                + 0.5 * einsum('ja,jb->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('jkbd,kd->jb', t2_1,
                                      einsum('ic,ikcd->kd', ur1, t2_2)))
                + 0.5 * einsum('ja,jb->ab', ur1,  # N^4: O^2V^2 / N^4: O^2V^2
                               einsum('jkbd,kd->jb', t2_2,
                                      einsum('ic,ikcd->kd', ur1, t2_1)))
                - 1 * einsum('jb,ja->ab',  # N^4: O^2V^2 / N^4: O^2V^2
                             einsum('ic,ijbc->jb', ur1, t2_2),
                             einsum('kd,jkad->ja', ur1, t2_1))
                - 0.5 * einsum('jkbc,jkac->ab', t2_2,  # N^5: O^2V^3 / N^4: O^2V^2
                               einsum('jkad,cd->jkac', t2_1,
                                      einsum('ic,id->cd', ur1, ur1)))
            )
        ).symmetrise()
        return dm

    def test_adcn(self, method: str, system: str, case: str, kind: str):
        state = testdata_cache.adcc_states(
            system=system, method=method, kind=kind, case=case
        )
        ref = state.excitation_energy_uncorrected
        adcn = self.calculate_adcn_excitation_energy(state)
        np.testing.assert_allclose(adcn, ref, atol=1e-12)
