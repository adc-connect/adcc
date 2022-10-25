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
from math import sqrt

from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum, direct_sum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def s2s_tdm_ea_adc0(mp, amplitude_l, amplitude_r, intermediates):
    check_singles_amplitudes([b.v], amplitude_l, amplitude_r)
    ul1 = amplitude_l.p
    ur1 = amplitude_r.p

    dm = OneParticleOperator(mp, is_symmetric=False)
    dm.vv = einsum("a,b->ab", ul1, ur1)
    return dm


def s2s_tdm_ea_adc2(mp, amplitude_l, amplitude_r, intermediates):
    check_doubles_amplitudes([b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_ea_adc0(mp, amplitude_l, amplitude_r, intermediates)

    ul1, ul2 = amplitude_l.p, amplitude_l.pph
    ur1, ur2 = amplitude_r.p, amplitude_r.pph

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_vv = dm.vv.evaluate()  # ADC(1) diffdm
    
    # Zeroth order doubles contributions
    p2_oo = -einsum("jab,iab->ij", ul2, ur2)
    p2_vv = 2 * einsum("iac,ibc->ab", ul2, ur2)
    p_ov = sqrt(2) * einsum("b,iba->ia", ul1, ur2)
    p_vo = sqrt(2) * einsum("iba,b->ai", ul2, ur1)
    
    # ADC(2) ISR intermediate (TODO Move to intermediates)
#    ru1 = einsum("i,ijab->jab", u1, t2).evaluate()

    # Compute second-order contributions to the density matrix
    dm.oo = (  # ea_adc2_p_oo
        + p2_oo 
        + einsum("ikc,jkc->ij", einsum("a,ikac->ikc", ul1, t2),
                 einsum("b,jkbc->jkc", ur1, t2))
    )

    dm.vv = (  # ea_adc2_p_vv
        + p1_vv + p2_vv 
        - 0.5 * einsum("c,ac,b->ab", ul1, p0.vv, ur1)
        - 0.5 * einsum("a,bc,b->ab", ul1, p0.vv, ur1)
        + 0.5 * einsum("ijb,ija->ab", einsum("c,ijcb->ijb", ul1, t2), 
                       einsum("d,ijad->ija", ur1, t2))
    )

    dm.ov = (  # ea_adc2_p_ov
        + p_ov
        + 1/sqrt(2) * (
            + einsum("jbc,ijbc,a->ia", ul2, t2, ur1)
            + 2 * einsum("jc,ijac->ia", einsum("jcb,b->jc", ul2, ur1), t2))
        - einsum("b,ib,a->ia", ul1, p0.ov, ur1)
    )
    
    dm.vo = (  # ea_adc2_p_vo
        + p_vo
        + 1/sqrt(2) * (
            + einsum("a,ijbc,jbc->ai", ul1, t2, ur2)
            + 2 * einsum("jc,ijac->ai", einsum("b,jcb->jc", ul1, ur2), t2))
        - einsum("b,ib,a->ai", ur1, p0.ov, ul1)
        # switched indices because p0_ov is used instead of p0_vo
    )
    return dm


# Ref: https://doi.org/10.1080/00268976.2013.859313
DISPATCH = {"ea_adc0": s2s_tdm_ea_adc0,
            "ea_adc1": s2s_tdm_ea_adc0,       # same as ADC(0)
            "ea_adc2": s2s_tdm_ea_adc2,
            "ea_adc2x": s2s_tdm_ea_adc2,      # same as ADC(2)
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