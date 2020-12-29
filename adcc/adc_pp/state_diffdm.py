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
