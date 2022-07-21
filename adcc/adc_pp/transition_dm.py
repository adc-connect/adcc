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


def tdm_adc0(mp, amplitude, intermediates):
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o
    check_singles_amplitudes([C, b.v], amplitude)
    u1 = amplitude.ph

    # Transition density matrix for (CVS-)ADC(0)
    dm = OneParticleOperator(mp, is_symmetric=False)
    dm[b.v + C] = u1.transpose()
    return dm


def tdm_adc1(mp, amplitude, intermediates):
    dm = tdm_adc0(mp, amplitude, intermediates)  # Get ADC(0) result
    # adc1_dp0_ov
    dm.ov = -einsum("ijab,jb->ia", mp.t2(b.oovv), amplitude.ph)
    return dm


def tdm_cvs_adc2(mp, amplitude, intermediates):
    # Get CVS-ADC(1) result (same as CVS-ADC(0))
    dm = tdm_adc0(mp, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = intermediates.cvs_p0

    # Compute CVS-ADC(2) tdm
    dm.oc = (  # cvs_adc2_dp0_oc
        - einsum("ja,Ia->jI", p0.ov, u1)
        + (1 / sqrt(2)) * einsum("kIab,jkab->jI", u2, t2)
    )

    # cvs_adc2_dp0_vc
    dm.vc = u1.transpose() - einsum("ab,Ib->aI", p0.vv, u1)
    return dm


def tdm_adc2(mp, amplitude, intermediates):
    dm = tdm_adc1(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    # Compute ADC(2) tdm
    dm.oo = (  # adc2_dp0_oo
        - einsum("ia,ja->ij", p0.ov, u1)
        - einsum("ikab,jkab->ij", u2, t2)
    )
    dm.vv = (  # adc2_dp0_vv
        + einsum("ia,ib->ab", u1, p0.ov)
        + einsum("ijac,ijbc->ab", u2, t2)
    )
    dm.ov -= einsum("ijab,jb->ia", td2, u1)  # adc2_dp0_ov
    dm.vo += 0.5 * (  # adc2_dp0_vo
        + einsum("ijab,jkbc,kc->ai", t2, t2, u1)
        - einsum("ab,ib->ai", p0.vv, u1)
        + einsum("ja,ij->ai", u1, p0.oo)
    )
    return dm

def tdm_adc3_raw(mp, amplitude, intermediates):
    dm = tdm_adc2(mp, amplitude, intermediates)
    #check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm
    tt2 = mp.tt2(b.ooovvv)
    ts3 = mp.ts3(b.ov)
    td3 = mp.td3(b.oovv)

    dm.vv += (
        + einsum('ib,ia->ab', ts3, u1)
        - einsum('ijbc,jc,ia->ab', t2, p0.ov, u1)
        + einsum('ijbc,ja,ic->ab', t2, p0.ov, u1)
        - 0.5 * einsum('ijkbcd,jkac,id->ab', tt2, t2, u1)
        - 0.25 * einsum('ijkbcd,jkcd,ia->ab', tt2, t2, u1)
        + einsum('ijbc,ijac->ab', td2, u2)
    )
    dm.oo += (
        - einsum('ia,ja->ij',ts3, u1)
        - einsum('ikab,jb,ka->ij', t2, p0.ov, u1)
        + einsum('ikab,kb,ja->ij', t2, p0.ov, u1)
        - 0.5 * einsum('iklabc,jkbc,la->ij', tt2, t2, u1)
        + 0.25 * einsum('iklabc,klbc,ja->ij', tt2, t2, u1)

        - einsum('ikab,jkab->ij', td2, u2)
    )
    dm.vo += (
        - 0.25 * einsum('jkbc,ikbc,ja->ai', t2, td2, u1)
        - 0.25 * einsum('jkbc,jkac,ib->ai', t2, td2, u1)
        + 0.5 * einsum('jkbc,ijab,kc->ai', t2, td2, u1)
        - 0.25 * einsum('jkbc,ikbc,ja->ai', td2, t2, u1)
        - 0.25 * einsum('jkbc,jkac,ib->ai', td2, t2, u1)
        + 0.5 * einsum('jkbc,ijab,kc->ai', td2, t2, u1)
    )
    dm.ov += (
        - 0.25 * einsum('ijbc,klad,klcd,jb->ia', t2, t2, t2, u1) #quadruples factorization and cancellation
        + 0.25 * einsum('ijbd,klac,klcd,jb->ia', t2, t2, t2, u1)
        - 0.25 * einsum('ijad,klbc,klcd,jb->ia', t2, t2, t2, u1)
        - 0.25 * einsum('ijcd,klab,klcd,jb->ia', t2, t2, t2, u1)
        - 0.25 * einsum('jkab,ilcd,klcd,jb->ia', t2, t2, t2, u1)
        - 0.25 * einsum('jkbc,ilad,klcd,jb->ia', t2, t2, t2, u1)
        + 0.25 * einsum('jkbd,ilac,klcd,jb->ia', t2, t2, t2, u1)
        + 0.25 * einsum('jkac,ilbd,klcd,jb->ia', t2, t2, t2, u1)
        - 0.25 * einsum('jkad,ilbc,klcd,jb->ia', t2, t2, t2, u1)
        - 0.25 * einsum('jkcd,ilab,klcd,jb->ia', t2, t2, t2, u1)
        + 0.25 * einsum('jlab,ikcd,klcd,jb->ia', t2, t2, t2, u1)
        + 0.25 * einsum('jlbc,ikad,klcd,jb->ia', t2, t2, t2, u1)
        - 0.25 * einsum('jlac,ikbd,klcd,jb->ia', t2, t2, t2, u1)
        + 0.25 * einsum('jlad,ikbc,klcd,jb->ia', t2, t2, t2, u1)
        + 0.25 * einsum('jlbd,ikac,klcd,jb->ia', t2, t2, t2, u1)
        - einsum('ijab,jb->ia', td3, u1)
        - 0.5 * einsum('ijkabc,jkbc->ia', tt2, u2)
    )
    return dm


def tdm_adc3(mp, amplitude, intermediates):
    dm = tdm_adc2(mp, amplitude, intermediates)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm
    tt2 = mp.tt2(b.ooovvv)
    ts3 = mp.ts3(b.ov)
    td3 = mp.td3(b.oovv)
    t2t2_vv = einsum('klad,klcd->ac', t2, t2).evaluate()
    t2t2_oo = einsum('ilcd,klcd->ik', t2, t2).evaluate()
    t2t2_oovv = einsum('ilad,klcd->ikac', t2, t2).evaluate()
    t2u_ov = einsum('ijab,ia->jb', t2, u1).evaluate()
    t2u_ovvv = einsum('ijbc,ia->jabc', t2, u1).evaluate()
    tt2t2_ov = einsum('ijkabc,jkbc->ia', tt2, t2).evaluate()

    dm.vv += (
        + einsum('ib,ia->ab', ts3, u1)
        - einsum('jabc,jc->ab', t2u_ovvv, p0.ov)
        - einsum('jb,ja->ab', t2u_ov, p0.ov)
        - 0.5 * einsum('jkac,ijkbcd,id->ab',t2, tt2, u1)
        - 0.25 * einsum('ib,ia->ab', tt2t2_ov, u1)
        + einsum('ijbc,ijac->ab', td2, u2)
    )
    dm.oo += (
        - einsum('ia,ja->ij',ts3, u1)
        + einsum('ib,jb->ij', t2u_ov, p0.ov)
        + einsum('ikab,kb,ja->ij', t2, p0.ov, u1)
        - 0.5 * einsum('jkbc,iklabc,la->ij', t2, tt2, u1)
        + 0.25 * einsum('ia,ja->ij', tt2t2_ov, u1)
        - einsum('ikab,jkab->ij', td2, u2)
    )
    dm.vo += (
        - 0.25 * einsum('kabc,ikbc->ai', t2u_ovvv, td2)
        - 0.25 * einsum('ib,jkbc,jkac->ai', u1, t2, td2)
        + 0.5 * einsum('jb,ijab->ai', t2u_ov, td2)
        - 0.25 * einsum('ja,jkbc,ikbc->ai', u1, td2, t2)
        - 0.25 * einsum('ib,jkbc,jkac->ai', u1, td2, t2)
        + 0.5 * einsum('ijab,jkbc,kc->ai', t2, td2, u1)
    )
    dm.ov += (
        + 0.25 * einsum('ic,ac->ia', t2u_ov, t2t2_vv) #quadruples factorization and cancellation
        + 0.25 * einsum('id,ad->ia', t2u_ov, t2t2_vv)
        - 0.25 * einsum('ibad,bd->ia', t2u_ovvv, t2t2_vv)
        + 0.25 * einsum('ibcd,klab,klcd->ia', t2u_ovvv, t2, t2)
        + 0.25 * einsum('ka,ik->ia', t2u_ov, t2t2_oo)
        - 0.25 * einsum('kc,ikac->ia', t2u_ov, t2t2_oovv)
        - 0.25 * einsum('kd,ikad->ia', t2u_ov, t2t2_oovv)
        + 0.25 * einsum('kbac,ikbc->ia', t2u_ovvv, t2t2_oovv)
        + 0.25 * einsum('kbad,ikbd->ia', t2u_ovvv, t2t2_oovv)
        + 0.25 * einsum('jl,ilab,jb->ia', t2t2_oo, t2, u1)
        + 0.25 * einsum('la,il->ia', t2u_ov, t2t2_oo)
        - 0.25 * einsum('lc,ilac->ia', t2u_ov, t2t2_oovv)
        + 0.25 * einsum('lbac,ilbc->ia', t2u_ovvv, t2t2_oovv)
        + 0.25 * einsum('lbad,ilbd->ia', t2u_ovvv, t2t2_oovv)
        + 0.25 * einsum('ld,ilad->ia', t2u_ov, t2t2_oovv)
        - einsum('ijab,jb->ia', td3, u1)
        - 0.5 * einsum('ijkabc,jkbc->ia', tt2, u2)
    )
    return dm    



DISPATCH = {
    "adc0": tdm_adc0,
    "adc1": tdm_adc1,
    "adc2": tdm_adc3,
    "adc2x": tdm_adc2,
    "adc3": tdm_adc3,
    "cvs-adc0": tdm_adc0,
    "cvs-adc1": tdm_adc0,  # No extra contribs for CVS-ADC(1)
    "cvs-adc2": tdm_cvs_adc2,
    "cvs-adc2x": tdm_cvs_adc2,
}


def transition_dm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle transition density matrix from ground to excited
    state in the MO basis.

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
        raise NotImplementedError("transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
