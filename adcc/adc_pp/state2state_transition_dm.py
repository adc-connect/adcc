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
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

import libadcc


def s2s_tdm_adc0(mp, amplitude_l, amplitude_r, intermediates):
    x1 = amplitude_l["s"]
    y1 = amplitude_r["s"]

    dm = OneParticleOperator(mp, is_symmetric=False)
    dm[b.oo] = -einsum('ja,ia->ij', x1, y1)
    dm[b.vv] = einsum('ia,ib->ab', x1, y1)
    return dm


def s2s_tdm_adc2(mp, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_adc0(mp, amplitude_l, amplitude_r, intermediates)

    x1 = amplitude_l["s"]
    y1 = amplitude_r["s"]
    x2 = amplitude_l["d"]
    y2 = amplitude_r["d"]

    t2 = mp.t2(b.oovv)
    p0_ov = mp.mp2_diffdm[b.ov]
    p0_oo = mp.mp2_diffdm[b.oo]
    p0_vv = mp.mp2_diffdm[b.vv]
    p1_oo = dm[b.oo].evaluate()  # ADC(1) tdm
    p1_vv = dm[b.vv].evaluate()  # ADC(1) tdm

    rx1 = einsum('ijab,jb->ia', t2, x1)
    ry1 = einsum('ijab,jb->ia', t2, y1)

    dm[b.oo] = (
        p1_oo - 2.0 * einsum('ikab,jkab->ij', y2, x2) + (
            + 0.5 * (
                einsum('ik,kj->ij', p1_oo, p0_oo)
                + einsum('ik,kj->ij', p0_oo, p1_oo)
            )
            - einsum(
                'ikcd,jkcd->ij', t2,
                einsum('lk,jlcd->jkcd', 0.5 * p1_oo, t2)
                - einsum('jkcb,db->jkcd', t2, p1_vv)
            )
            - 0.5 * (
                einsum('ia,ja->ij', y1, einsum('jkac,kc->ja', t2, rx1))
                + einsum('ikac,kc,ja->ij', t2, ry1, x1)
            )
            - einsum('ia,ja->ij', rx1, ry1)
        )
    )
    dm[b.vv] = (
        p1_vv + 2.0 * einsum('ijac,ijbc->ab', x2, y2)
        + (
            - 0.5 * (
                einsum("ac,cb->ab", p1_vv, p0_vv)
                + einsum("ac,cb->ab", p0_vv, p1_vv)
            )
            - einsum(
                "klbc,klac->ab", t2,
                0.5 * einsum("klad,cd->klac", t2, p1_vv)
                - einsum("jk,jlac->klac", p1_oo, t2)
            )
            + 0.5 * (
                einsum("ikac,kc,ib->ab", t2, rx1, y1)
                + einsum(
                    "ia,ib->ab", x1,
                    einsum("ikbc,kc->ib", t2, ry1)
                )
            )
            + einsum("ia,ib->ab", ry1, rx1)
        )
    )

    p1_ov = -2.0 * einsum("jb,ijab->ia", x1, y2)
    p1_vo = -2.0 * einsum("ijab,jb->ai", x2, y1)

    dm[b.ov] = (
        p1_ov
        - einsum("ijab,bj->ia", t2, p1_vo)
        - einsum("ib,ba->ia", p0_ov, p1_vv)
        + einsum("ij,ja->ia", p1_oo, p0_ov)
        - einsum(
            "ib,ab->ia", y1,
            einsum("klca,klcb->ab", t2, x2)
        )
        - einsum("ikcd,jkcd,ja->ia", t2, x2, y1)
    )
    dm[b.vo] = (
        p1_vo
        - einsum("ijab,jb->ai", t2, p1_ov)
        - einsum("ib,ab->ai", p0_ov, p1_vv)
        + einsum("ji,ja->ai", p1_oo, p0_ov)
        - einsum(
            "ib,ab->ai", x1,
            einsum("klca,klcb->ab", t2, y2)
        )
        - einsum("ikcd,jkcd,ja->ai", t2, y2, x1)
    )
    return dm


# Ref: https://doi.org/10.1080/00268976.2013.859313
DISPATCH = {"adc0": s2s_tdm_adc0,
            "adc1": s2s_tdm_adc0,       # same as ADC(0)
            "adc2": s2s_tdm_adc2,
            "adc2x": s2s_tdm_adc2,      # same as ADC(2)
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
    intermediates : AdcIntermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, libadcc.LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = libadcc.AdcIntermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state2state_transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude_from, amplitude_to,
                                    intermediates)
        return ret.evaluate()
