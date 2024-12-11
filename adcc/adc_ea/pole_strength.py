#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
from adcc.functions import einsum, zeros_like, dot, direct_sum
from adcc.Intermediates import Intermediates, register_as_intermediate

from .util import check_doubles_amplitudes, check_singles_amplitudes


def pole_strength_ea_adc0(mp, amplitude, intermediates):
    check_singles_amplitudes([b.v], amplitude)
    # Build Kronecker delta
    hf = mp.reference_state
    d_vv = zeros_like(hf.fvv)
    d_vv.set_mask("aa", 1.0)
    f11 = d_vv

    # Calculate the spectroscopic amplitude x
    xa = einsum("b,ba->a", amplitude.p, f11)

    return dot(xa, xa)


def pole_strength_ea_adc2(mp, amplitude, intermediates):
    check_singles_amplitudes([b.v], amplitude)
    check_doubles_amplitudes([b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.p, amplitude.pph

    f11 = intermediates.ea_adc2_f11
    f12 = - mp.mp2_diffdm.ov        # -t_ia
    f22 = intermediates.ea_adc2_f22

    # Calculate the spectroscopic amplitude x
    xi = einsum("ia,a->i", f12, u1) + einsum("ijab,jab->i", f22, u2)
    xa = einsum("b,ba->a", u1, f11)

    return dot(xi, xi) + dot(xa, xa)


def pole_strength_ea_adc3(mp, amplitude, intermediates):
    check_singles_amplitudes([b.v], amplitude)
    check_doubles_amplitudes([b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.p, amplitude.pph

    f11 = intermediates.ea_adc3_f11
    # TODO: when mp3_diffdm is implemented, intermediate can be directly reused
    # to avoid redundancy
    # f12 = - intermediates.mp3_diffdm.ov
    f12 = intermediates.ea_adc3_f12
    f22 = intermediates.ea_adc3_f22

    # Calculate the spectroscopic amplitude x
    xi = einsum("ia,a->i", f12, u1) + einsum("ijab,jab->i", f22, u2)
    xa = einsum("b,ba->a", u1, f11)

    return dot(xi, xi) + dot(xa, xa)


#
# Intermediates
#

@register_as_intermediate
def ea_adc2_f11(hf, mp, intermediates):
    # effective transition moments, vv part f_ab
    # Build Kronecker delta
    d_vv = zeros_like(hf.fvv)
    d_vv.set_mask("aa", 1.0)

    t2 = mp.t2(b.oovv)

    return d_vv - 0.25 * einsum("ijbc,ijac->ab", t2, t2)


@register_as_intermediate
def ea_adc2_f22(hf, mp, intermediates):
    # effective transition moments, oovv part f_ijab
    return 1/sqrt(2) * mp.t2(b.oovv)


@register_as_intermediate
def ea_adc3_f11(hf, mp, intermediates):
    # effective transition moments, vv part f_ab
    # Build Kronecker delta
    d_vv = zeros_like(hf.fvv)
    d_vv.set_mask("aa", 1.0)

    df = mp.df(b.o + b.v)
    df2 = direct_sum("ib+jc->ijbc", df, df).symmetrise((2, 3))

    t2 = mp.t2(b.oovv)

    return (d_vv
            - 0.25 * einsum("ijbc,ijac->ab", t2, t2)
            + (+ 0.25 * einsum("ijac,ijbc->ab", t2,
                               mp.t2eri(b.oovv, b.vv) / df2)
               + 0.25 * einsum("ijac,ijbc->ab", t2,
                               mp.t2eri(b.oovv, b.oo) / df2)
               + einsum("ijac,ijbc->ab", t2, mp.t2eri(b.oovv, b.ov) / df2)
               - einsum("ijac,jibc->ab", t2, mp.t2eri(b.oovv, b.ov) / df2))
            )


@register_as_intermediate
def ea_adc3_f12(hf, mp, intermediates):
    # effective transition moments, ov part f_ia
    # Intermediates are defined in /adcc/adc_ip/pole_strength.py
    return - mp.mp2_diffdm.ov + (intermediates.sigma_ov
                                 + intermediates.m_3_plus
                                 + intermediates.m_3_minus
                                 ) / mp.df(b.o + b.v)


@register_as_intermediate
def ea_adc3_f22(hf, mp, intermediates):
    # effective transition moments, oovv part f_ijab
    df = mp.df(b.o + b.v)
    df2 = direct_sum("ia+jb->ijab", df, df).symmetrise((2, 3))

    return (1/sqrt(2) * mp.t2(b.oovv)
            + 1/sqrt(2) * (0.5 * (mp.t2eri(b.oovv, b.oo)
                           + mp.t2eri(b.oovv, b.vv))
                           + ((mp.t2eri(b.oovv, b.ov)
                               ).antisymmetrise(2, 3)).antisymmetrise(0, 1)
                           ) / df2
            )


DISPATCH = {
    "ea_adc0": pole_strength_ea_adc0,
    "ea_adc1": pole_strength_ea_adc0,
    "ea_adc2": pole_strength_ea_adc2,
    "ea_adc3": pole_strength_ea_adc3,
}


def pole_strength(method, ground_state, amplitude, intermediates=None):
    """Compute the pole strength of the electron attached state for the
    provided ADC method from the spectroscopic amplitude x.

    Parameters
    ----------
    method: adc.Method
        Provide a method at which to compute the MTMs
    ground_state : adcc.LazyMp
        The MP ground state
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse

    Returns
    -------
    Scalar
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)
    if method.name not in DISPATCH:
        raise NotImplementedError("pole_strength is not "
                                  f"implemented for {method.name}.")

    ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
    return ret
