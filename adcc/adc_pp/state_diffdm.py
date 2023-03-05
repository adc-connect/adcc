#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from math import sqrt

from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def diffdm_adc0(mp, amplitude, intermediates):
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o
    check_singles_amplitudes([C, b.v], amplitude)
    u1 = amplitude.ph

    dm = OneParticleOperator(mp, is_symmetric=True)
    dm[C + C] = -einsum("ia,ja->ij", u1, u1)
    dm.vv = einsum("ia,ib->ab", u1, u1)
    return dm


def diffdm_adc2(mp, amplitude, intermediates):
    dm = diffdm_adc0(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.ph, amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_oo = dm.oo.evaluate()  # ADC(1) diffdm
    p1_vv = dm.vv.evaluate()  # ADC(1) diffdm

    # Zeroth order doubles contributions
    p2_oo = -einsum("ikab,jkab->ij", u2, u2)
    p2_vv = einsum("ijac,ijbc->ab", u2, u2)
    p2_ov = -2 * einsum("jb,ijab->ia", u1, u2).evaluate()

    # ADC(2) ISR intermediate (TODO Move to intermediates)
    ru1 = einsum("ijab,jb->ia", t2, u1).evaluate()

    # Compute second-order contributions to the density matrix
    dm.oo = (  # adc2_p_oo
        p1_oo + 2 * p2_oo - einsum("ia,ja->ij", ru1, ru1) + (
            + einsum("ik,kj->ij", p1_oo, p0.oo)
            - einsum("ikcd,jkcd->ij", t2,
                     + 0.5 * einsum("lk,jlcd->jkcd", p1_oo, t2)
                     - einsum("jkcb,db->jkcd", t2, p1_vv))
            - einsum("ia,jkac,kc->ij", u1, t2, ru1)
        ).symmetrise()
    )

    dm.vv = (  # adc2_p_vv
        p1_vv + 2 * p2_vv + einsum("ia,ib->ab", ru1, ru1) - (
            + einsum("ac,cb->ab", p1_vv, p0.vv)
            + einsum("klbc,klac->ab", t2,
                     + 0.5 * einsum("klad,cd->klac", t2, p1_vv)
                     - einsum("jk,jlac->klac", p1_oo, t2))
            - einsum("ikac,kc,ib->ab", t2, ru1, u1)
        ).symmetrise()
    )

    dm.ov = (  # adc2_p_ov
        + p2_ov
        - einsum("ijab,jb->ia", t2, p2_ov)
        - einsum("ib,ba->ia", p0.ov, p1_vv)
        + einsum("ij,ja->ia", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ia", u1, t2, u2)
        - einsum("ikcd,jkcd,ja->ia", t2, u2, u1)
    )
    return dm


def diffdm_adc3(mp, amplitude, intermediates):
    dm = diffdm_adc2(mp, amplitude, intermediates)
    u1, u2 = amplitude.ph, amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    td2 = mp.td2(b.oovv)
    tt2 = mp.tt2(b.ooovvv)
    ts3 = mp.ts3(b.ov)

    # Zeroth order doubles contributions
    p2_ov = -2 * einsum("jb,ijab->ia", u1, u2).evaluate()

    # ADC(2) ISR intermediate (TODO Move to intermediates)
    ru1 = einsum("ijab,jb->ia", t2, u1).evaluate()

    dm.vv += (
        - 2.0 * einsum('ia,ka,jkcd,ijbc->db', u1, u1, t2, td2)
        + 0.5 * einsum('ia,ib,jkac,jkcd->db', u1, u1, t2, td2)
        + 0.5 * einsum('ia,ib,jkcd,jkac->db', u1, u1, t2, td2)
        - 1.0 * einsum('ia,ib,jkbd,jkac->dc', u1, u1, t2, td2)
        + 1.0 * einsum('ia,jd,ijcd->ca', u1, ru1, td2)
        - 1.0 * einsum('ia,kb,jkcd,ijad->cb', u1, u1, t2, td2)
        + 2.0 * einsum('ia,jd,ijac->dc', u1, ru1, td2)
        + 2.0 * einsum('ib,ic->cb', p2_ov, p0.ov)
    )
    dm.oo += (
        + 0.5 * einsum('ia,ja,klbc,jlbc->ki', u1, u1, t2, td2)
        - 0.5 * einsum('ia,la,klbc,jkbc->ji', u1, u1, t2, td2)
        - 1.0 * einsum('ia,la,ijbc,klbc->kj', u1, u1, t2, td2)
        - 2.0 * einsum('ia,ib,klbc,jkac->lj', u1, u1, t2, td2)
        - 1.0 * einsum('ia,jb,klac,jlbc->ki', u1, u1, t2, td2)
        - 1.0 * einsum('ia,lc,klac->ki', u1, ru1, td2)
        - 2.0 * einsum('ia,lc,ijac->lj', u1, ru1, td2)
        - 2.0 * einsum('ib,kb->ki', p2_ov, p0.ov)
    )
    dm.ov += (
        - 1.0 * einsum('ia,ja,jb->ib', u1, u1, ts3)
        + 1.0 * einsum('ia,ja,jkbc,ib->kc', u1, u1, t2, p0.ov)
        - 0.5 * einsum('ia,ja,ilbc,jklbcd->kd', u1, u1, t2, tt2)
        + 1.0 * einsum('ia,ka,jkbc,jb->ic', u1, u1, t2, p0.ov)
        - 0.25 * einsum('ia,ka,jlbc,jklbcd->id', u1, u1, t2, tt2)
        - 1.0 * einsum('ia,ib,jb->ja', u1, u1, ts3)
        - 0.5 * einsum('ia,ib,klbd,jklacd->jc', u1, u1, t2, tt2)
        - 1.0 * einsum('ia,ib,jkbc,jc->ka', u1, u1, t2, p0.ov)
        - 0.25 * einsum('ia,ib,jlcd,jklacd->kb', u1, u1, t2, tt2)
        + 1.0 * einsum('ia,ib,jkbc,ja->kc', u1, u1, t2, p0.ov)
        - 0.5 * einsum('ia,jb,klbc,iklacd->jd', u1, u1, t2, tt2)
        + 1.0 * einsum('ia,kc,ic->ka', u1, ru1, p0.ov)
        + 1.0 * einsum('ia,jc,ja->ic', u1, ru1, p0.ov)
        - 0.5 * einsum('ia,lb,klcd,ijkacd->jb', u1, u1, t2, tt2)
        + 1.0 * einsum('ia,kc,ijkacd->jd', u1, ru1, tt2)
        + 0.5 * einsum('jb,jlbd,klcd->kc', p2_ov, t2, t2)
        - 1.0 * einsum('jb,jkbc->kc', p2_ov, td2)
        - 0.25 * einsum('kb,jlcd,klcd->jb', p2_ov, t2, t2)
        + 1.0 * einsum('ia,jkab,jkbc->ic', u1, u2, td2)
        - 0.25 * einsum('ia,klab,ijcd,klcd->jb', u1, u2, t2, t2)
        - 0.25 * einsum('jc,klbd,klcd->jb', p2_ov, t2, t2)
        + 1.0 * einsum('ia,jkac,ilbd,klcd->jb', u1, u2, t2, t2)
        + 0.5 * einsum('ia,klac,ijbd,klcd->jb', u1, u2, t2, t2)
        + 1.0 * einsum('ia,ijbc,jkbc->ka', u1, u2, td2)
        - 1.0 * einsum('ia,ijbc,klad,jlbd->kc', u1, u2, t2, t2)
        - 0.5 * einsum('klbc,jd,klcd->jb', u2, ru1, t2)
        - 0.25 * einsum('ia,ijbd,klac,klbd->jc', u1, u2, t2, t2)
        - 0.5 * einsum('jkbd,lc,klbd->jc', u2, ru1, t2)
        + 0.5 * einsum('jc,ijab,ikab,klcd->ld', u1, u2, t2, t2)
    )
    return dm


def diffdm_cvs_adc2(mp, amplitude, intermediates):
    dm = diffdm_adc0(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
    u1, u2 = amplitude.ph, amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = intermediates.cvs_p0
    p1_vv = dm.vv.evaluate()  # ADC(1) diffdm

    # Zeroth order doubles contributions
    p2_ov = -sqrt(2) * einsum("jb,ijab->ia", u1, u2)
    p2_vo = -sqrt(2) * einsum("ijab,jb->ai", u2, u1)
    p2_oo = -einsum("ljab,kjab->kl", u2, u2)
    p2_vv = 2 * einsum("ijac,ijbc->ab", u2, u2)

    # Second order contributions
    # cvs_adc2_dp_oo
    dm.oo = p2_oo + einsum("ab,ikac,jkbc->ij", p1_vv, t2, t2)

    dm.ov = p2_ov + (  # cvs_adc2_dp_ov
        - einsum("ka,ab->kb", p0.ov, p1_vv)
        - einsum("lkdb,dl->kb", t2, p2_vo)
        + 1 / sqrt(2) * einsum("ib,klad,liad->kb", u1, t2, u2)
    )

    dm.vv = p1_vv + p2_vv - 0.5 * (  # cvs_adc2_dp_vv
        + einsum("cb,ac->ab", p1_vv, p0.vv)
        + einsum("cb,ac->ab", p0.vv, p1_vv)
        + einsum("ijbc,ijad,cd->ab", t2, t2, p1_vv)
    )

    # Add 2nd order correction to CVS-ADC(1) diffdm
    dm.cc -= einsum("kIab,kJab->IJ", u2, u2)
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "adc0": diffdm_adc0,
    "adc1": diffdm_adc0,       # same as ADC(0)
    "adc2": diffdm_adc2,
    "adc2x": diffdm_adc2,
    "adc3": diffdm_adc3,
    "cvs-adc0": diffdm_adc0,
    "cvs-adc1": diffdm_adc0,   # same as ADC(0)
    "cvs-adc2": diffdm_cvs_adc2,
    "cvs-adc2x": diffdm_cvs_adc2,
}


def state_diffdm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle difference density matrix of an excited state
    in the MO basis.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state_diffdm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
