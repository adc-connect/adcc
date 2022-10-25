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
from adcc.functions import einsum, direct_sum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

from .util import check_doubles_amplitudes, check_singles_amplitudes


def diffdm_ea_adc0(mp, amplitude, intermediates):
    check_singles_amplitudes([b.v], amplitude)
    u1 = amplitude.p
    
    dm = OneParticleOperator(mp, is_symmetric=True)
    dm.vv = einsum("a,b->ab", u1, u1)
    return dm


def diffdm_ea_adc2(mp, amplitude, intermediates):
    dm = diffdm_ea_adc0(mp, amplitude, intermediates)  # Get ADC(0/1) result
    check_doubles_amplitudes([b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.p, amplitude.pph
    
    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_vv = dm.vv.evaluate()  # ADC(1) diffdm
    
    # Zeroth order doubles contributions
    p2_oo = -einsum("jab,iab->ij", u2, u2)
    p2_vv = 2 * einsum("iac,ibc->ab", u2, u2)
    p_ov = sqrt(2) * einsum("b,iba->ia", u1, u2)
    
    # ADC(2) ISR intermediate (TODO Move to intermediates)
    # ru1 = einsum("i,ijab->jab", u1, t2).evaluate()

    # Compute second-order contributions to the density matrix
    dm.oo = (  # ea_adc2_p_oo
        + p2_oo 
        + einsum("ikc,jkc->ij", einsum("a,ikac->ikc", u1, t2),
                 einsum("b,jkbc->jkc", u1, t2))
    )

    dm.vv = (  # ea_adc2_p_vv
        + p1_vv + p2_vv 
        - 0.5 * einsum("c,ac,b->ab", u1, p0.vv, u1)
        - 0.5 * einsum("a,bc,c->ab", u1, p0.vv, u1)
        + 0.5 * einsum("ijb,ija->ab", einsum("c,ijcb->ijb", u1, t2), 
                       einsum("d,ijad->ija", u1, t2))
    )

    dm.ov = (  # ea_adc2_p_ov
        + p_ov
        + 1/sqrt(2) * (
            + einsum("jbc,ijbc,a->ia", u2, t2, u1)
            + 2 * einsum("jc,ijac->ia", einsum("jcb,b->jc", u2, u1), t2))
        - einsum("b,ib,a->ia", u1, p0.ov, u1)
    )
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "ea_adc0": diffdm_ea_adc0,
    "ea_adc1": diffdm_ea_adc0,       # same as ADC(0)
    "ea_adc2": diffdm_ea_adc2
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
