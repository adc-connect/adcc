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
from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def s2s_tdm_adc0(mp, amplitude_l, amplitude_r, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    dm = OneParticleOperator(mp, is_symmetric=False)
    dm.oo = -einsum('ja,ia->ij', ul1, ur1)
    dm.vv = einsum('ia,ib->ab', ul1, ur1)
    return dm


def s2s_tdm_adc2(mp, amplitude_l, amplitude_r, intermediates):
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_adc0(mp, amplitude_l, amplitude_r, intermediates)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_oo = dm.oo.evaluate()  # ADC(1) tdm
    p1_vv = dm.vv.evaluate()  # ADC(1) tdm

    # ADC(2) ISR intermediate (TODO Move to intermediates)
    rul1 = einsum('ijab,jb->ia', t2, ul1).evaluate()
    rur1 = einsum('ijab,jb->ia', t2, ur1).evaluate()

    dm.oo = (
        p1_oo - 2.0 * einsum('ikab,jkab->ij', ur2, ul2)
        + 0.5 * einsum('ik,kj->ij', p1_oo, p0.oo)
        + 0.5 * einsum('ik,kj->ij', p0.oo, p1_oo)
        - 0.5 * einsum('ikcd,lk,jlcd->ij', t2, p1_oo, t2)
        + 1.0 * einsum('ikcd,jkcb,db->ij', t2, t2, p1_vv)
        - 0.5 * einsum('ia,jkac,kc->ij', ur1, t2, rul1)
        - 0.5 * einsum('ikac,kc,ja->ij', t2, rur1, ul1)
        - 1.0 * einsum('ia,ja->ij', rul1, rur1)
    )
    dm.vv = (
        p1_vv + 2.0 * einsum('ijac,ijbc->ab', ul2, ur2)
        - 0.5 * einsum("ac,cb->ab", p1_vv, p0.vv)
        - 0.5 * einsum("ac,cb->ab", p0.vv, p1_vv)
        - 0.5 * einsum("klbc,klad,cd->ab", t2, t2, p1_vv)
        + 1.0 * einsum("klbc,jk,jlac->ab", t2, p1_oo, t2)
        + 0.5 * einsum("ikac,kc,ib->ab", t2, rul1, ur1)
        + 0.5 * einsum("ia,ikbc,kc->ab", ul1, t2, rur1)
        + 1.0 * einsum("ia,ib->ab", rur1, rul1)
    )

    p1_ov = -2.0 * einsum("jb,ijab->ia", ul1, ur2).evaluate()
    p1_vo = -2.0 * einsum("ijab,jb->ai", ul2, ur1).evaluate()

    dm.ov = (
        p1_ov
        - einsum("ijab,bj->ia", t2, p1_vo)
        - einsum("ib,ba->ia", p0.ov, p1_vv)
        + einsum("ij,ja->ia", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ia", ur1, t2, ul2)
        - einsum("ikcd,jkcd,ja->ia", t2, ul2, ur1)
    )
    dm.vo = (
        p1_vo
        - einsum("ijab,jb->ai", t2, p1_ov)
        - einsum("ib,ab->ai", p0.ov, p1_vv)
        + einsum("ji,ja->ai", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ai", ul1, t2, ur2)
        - einsum("ikcd,jkcd,ja->ai", t2, ur2, ul1)
    )
    return dm


def s2s_tdm_adc3(mp, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_adc2(mp, amplitude_l, amplitude_r, intermediates)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    t2 = mp.t2(b.oovv)
    p0_initial = mp.mp2_diffdm
    p0 = p0_initial.ov
    td2 = mp.td2(b.oovv)
    tt2 = mp.tt2(b.ooovvv)
    ts3 = mp.ts3(b.ov)

    # intermediates
    rul1 = einsum('ijab,jb->ia', t2, ul1).evaluate()
    rur1 = einsum('ijab,jb->ia', t2, ur1).evaluate()
    p1_ov = -2.0 * einsum("jb,ijab->ia", ul1, ur2).evaluate()
    p1_vo = -2.0 * einsum("ijab,jb->ai", ul2, ur1).evaluate()
    t2t2_vv = einsum('klad,klcd->ac', t2, t2).evaluate()
    t2t2_oo = einsum('ilcd,klcd->ik', t2, t2).evaluate()
    t2t2_oovv = einsum('ilad,klcd->ikac', t2, t2).evaluate()
    tt2t2_ov = einsum('ijkabc,jkbc->ia', tt2, t2).evaluate()
    td2t2_oo = einsum('ikcd,jkcd->ij', td2, t2).evaluate()
    td2t2_vv = einsum('ijbd,ijcd->bc', td2, t2).evaluate()
    td2t2_oovv = einsum('ikac,jkbc->ijab', td2, t2).evaluate()

    dm.oo += (
        - 0.5 * einsum('jlbc,lc,ib->ij', td2, rul1, ur1)
        - 0.5 * einsum('liba,ja,lb->ij', td2t2_oovv, ul1, ur1)
        + 1.0 * einsum('jkbc,ic,kb->ij', td2, rul1, ur1)
        - 0.5 * einsum('ilac,ja,lc->ij', td2, ul1, rur1)
        - 0.5 * einsum('ljab,la,ib->ij', td2t2_oovv, ul1, ur1)
        + 1.0 * einsum('ikac,ka,jc->ij', td2, ul1, rur1)
        + 0.25 * einsum('li,ja,la->ij', td2t2_oo, ul1, ur1)
        + 0.25 * einsum('jk,ka,ia->ij', td2t2_oo, ul1, ur1)
        + 0.5 * einsum('jkcd,ilcd,la,ka->ij', td2, t2, ul1, ur1)
        + 0.25 * einsum('lj,la,ia->ij', td2t2_oo, ul1, ur1)
        + 0.25 * einsum('ik,ja,ka->ij', td2t2_oo, ul1, ur1)
        + 0.5 * einsum('jlcd,ikcd,ka,la->ij', t2, td2, ul1, ur1)
        + 1.0 * einsum('jlbc,ilac,ka,kb->ij', td2, t2, ul1, ur1)
        + 1.0 * einsum('jlbc,ilac,ka,kb->ij', t2, td2, ul1, ur1)
        - 1.0 * einsum('jc,ic->ij', p0, p1_ov)
        - 1.0 * einsum('ib,bj->ij', p0, p1_vo)
    )
    dm.vv += (
        + 0.5 * einsum('jidb,ia,jd->ab', td2t2_oovv, ul1, ur1)
        + 0.5 * einsum('jkad,kd,jb->ab', td2, rul1, ur1)
        - 1.0 * einsum('jkad,kb,jd->ab', td2, rul1, ur1)
        + 0.5 * einsum('ijda,id,jb->ab', td2t2_oovv, ul1, ur1)
        + 0.5 * einsum('ikbd,ia,kd->ab', td2, ul1, rur1)
        - 1.0 * einsum('ikbd,id,ka->ab', td2, ul1, rur1)
        - 0.25 * einsum('db,ia,id->ab', td2t2_vv, ul1, ur1)
        - 0.25 * einsum('ac,ic,ib->ab', td2t2_vv, ul1, ur1)
        - 0.5 * einsum('klad,klbc,ic,id->ab', td2, t2, ul1, ur1)
        - 0.25 * einsum('da,id,ib->ab', td2t2_vv, ul1, ur1)
        - 0.25 * einsum('bc,ia,ic->ab', td2t2_vv, ul1, ur1)
        - 0.5 * einsum('klac,klbd,id,ic->ab', t2, td2, ul1, ur1)
        - 1.0 * einsum('jiab,ic,jc->ab', td2t2_oovv, ul1, ur1)
        - 1.0 * einsum('ijba,ic,jc->ab', td2t2_oovv, ul1, ur1)
        + 1.0 * einsum('ka,kb->ab', p0, p1_ov)
        + 1.0 * einsum('jb,aj->ab', p0, p1_vo)
    )
    dm.ov += (
        - 1.0 * einsum('ib,jb,ja->ia', ts3, ul1, ur1)
        - 1.0 * einsum('ja,jb,ib->ia', ts3, ul1, ur1)
        + 1.0 * einsum('kb,ka,ib->ia', p0, rul1, ur1)
        + 1.0 * einsum('jc,ic,ja->ia', p0, rul1, ur1)
        + 1.0 * einsum('kc,jkac,jb,ib->ia', p0, t2, ul1, ur1)
        + 1.0 * einsum('jc,ikac,kb,jb->ia', p0, t2, ul1, ur1)
        + 1.0 * einsum('kc,ikbc,jb,ja->ia', p0, t2, ul1, ur1)
        + 1.0 * einsum('kb,ikac,jc,jb->ia', p0, t2, ul1, ur1)
        - 0.5 * einsum('jlcd,iklbcd,kb,ja->ia', t2, tt2, ul1, ur1)
        - 0.5 * einsum('klbd,jklacd,jc,ib->ia', t2, tt2, ul1, ur1)
        - 1.0 * einsum('iklacd,ld,kc->ia', tt2, ul1, rur1)
        + 0.25 * einsum('ja,jb,ib->ia', tt2t2_ov, ul1, ur1)
        - 0.5 * einsum('jkcd,iklacd,lb,jb->ia', t2, tt2, ul1, ur1)
        + 0.25 * einsum('ib,jb,ja->ia', tt2t2_ov, ul1, ur1)
        + 0.5 * einsum('jkbd,ijkacd,lc,lb->ia', t2, tt2, ul1, ur1)
        - 0.25 * einsum('ki,ka->ia', t2t2_oo, p1_ov)
        - 0.25 * einsum('ca,ic->ia', t2t2_vv, p1_ov)
        + 0.5 * einsum('ikac,kc->ia', t2t2_oovv, p1_ov)
        - 1.0 * einsum('kibc,mc,kmab->ia', t2t2_oovv, ul1, ur2)
        + 1.0 * einsum('jkca,kb,ijbc->ia', t2t2_oovv, ul1, ur2)
        - 0.5 * einsum('jkcd,ilad,lb,jkbc->ia', t2, t2, ul1, ur2)
        - 0.25 * einsum('jkcd,ilcd,lb,jkab->ia', t2, t2, ul1, ur2)
        - 0.5 * einsum('jkbd,id,jkab->ia', t2, rul1, ur2)
        + 0.5 * einsum('klbc,ilad,md,kmbc->ia', t2, t2, ul1, ur2)
        - 0.25 * einsum('lmbc,lmad,kd,ikbc->ia', t2, t2, ul1, ur2)
        - 0.5 * einsum('jlbc,la,ijbc->ia', t2, rul1, ur2)
        - 1.0 * einsum('ijab,bj->ia', td2, p1_vo)
        + 1.0 * einsum('jkab,jkbc,ic->ia', td2, ul2, ur1)
        + 1.0 * einsum('ijbc,jkbc,ka->ia', td2, ul2, ur1)
    )
    dm.vo += (
        - 1.0 * einsum('ib,ja,jb->ai', ts3, ul1, ur1)
        - 1.0 * einsum('ja,ib,jb->ai', ts3, ul1, ur1)
        + 1.0 * einsum('kc,ic,ka->ai', p0, ul1, rur1)
        + 1.0 * einsum('kc,ka,ic->ai', p0, ul1, rur1)
        + 1.0 * einsum('jkac,kc,ib,jb->ai', t2, p0, ul1, ur1)
        + 1.0 * einsum('ijac,kc,kb,jb->ai', t2, p0, ul1, ur1)
        + 1.0 * einsum('ikbc,kc,ja,jb->ai', t2, p0, ul1, ur1)
        + 1.0 * einsum('ikab,kc,jc,jb->ai', t2, p0, ul1, ur1)
        - 0.5 * einsum('ijlbcd,klcd,ka,jb->ai', tt2, t2, ul1, ur1)
        - 0.5 * einsum('jklabd,klcd,ic,jb->ai', tt2, t2, ul1, ur1)
        - 1.0 * einsum('ijkabc,kc,jb->ai', tt2, rul1, ur1)
        + 0.25 * einsum('ja,ib,jb->ai', tt2t2_ov, ul1, ur1)
        - 0.5 * einsum('ijkacd,klcd,lb,jb->ai', tt2, t2, ul1, ur1)
        + 0.25 * einsum('ib,ja,jb->ai', tt2t2_ov, ul1, ur1)
        + 0.5 * einsum('iklabd,klcd,jc,jb->ai', tt2, t2, ul1, ur1)
        - 1.0 * einsum('ikac,kc->ai', td2, p1_ov)
        - 1.0 * einsum('jkac,ib,jkbc->ai', td2, ul1, ur2)
        + 1.0 * einsum('ikbc,la,klbc->ai', td2, ul1, ur2)
        - 0.25 * einsum('ij,aj->ai', t2t2_oo, p1_vo)
        - 0.25 * einsum('ab,bi->ai', t2t2_vv, p1_vo)
        + 0.5 * einsum('ijab,bj->ai', t2t2_oovv, p1_vo)
        - 1.0 * einsum('ijcb,jkab,kc->ai', t2t2_oovv, ul2, ur1)
        - 1.0 * einsum('kjab,ijbc,kc->ai', t2t2_oovv, ul2, ur1)
        + 0.5 * einsum('ikad,jlbd,jlbc,kc->ai', t2, t2, ul2, ur1)
        - 0.25 * einsum('ikcd,jlcd,jlab,kb->ai', t2, t2, ul2, ur1)
        - 0.5 * einsum('jlbd,jlab,id->ai', t2, ul2, rur1)
        + 0.5 * einsum('ilac,jlbd,jkbd,kc->ai', t2, t2, ul2, ur1)
        - 0.25 * einsum('lmac,lmbd,ijbd,jc->ai', t2, t2, ul2, ur1)
        - 0.5 * einsum('jlbd,ijbd,la->ai', t2, ul2, rur1)
    )
    return dm


# Ref: https://doi.org/10.1080/00268976.2013.859313
DISPATCH = {"adc0": s2s_tdm_adc0,
            "adc1": s2s_tdm_adc0,       # same as ADC(0)
            "adc2": s2s_tdm_adc2,
            "adc2x": s2s_tdm_adc2,      # same as ADC(2)
            "adc3": s2s_tdm_adc3,
            }


def state2state_transition_dm(method, ground_state, amplitude_from,
                              amplitude_to, intermediates=None):
    """
    Compute the state to state transition density matrix
    state in the MO basis using the intermediate-states representation.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude_from : AmplitudeVector
        The amplitude vector of the state to start from
    amplitude_to : AmplitudeVector
        The amplitude vector of the state to excite to
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state2state_transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        # final state is on the bra side/left (complex conjugate)
        # see ref https://doi.org/10.1080/00268976.2013.859313, appendix A2
        ret = DISPATCH[method.name](ground_state, amplitude_to, amplitude_from,
                                    intermediates)
        return ret.evaluate()
