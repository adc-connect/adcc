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


def pole_strength_ip_adc0(mp, amplitude, intermediates):
    check_singles_amplitudes([b.o], amplitude)
    # Build Kronecker delta
    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1.0)
    f11 = d_oo

    # Calculate the spectroscopic amplitude x
    xi = einsum("j,ji->i", amplitude.h, f11)

    return dot(xi, xi)


def pole_strength_ip_adc2(mp, amplitude, intermediates):
    check_singles_amplitudes([b.o], amplitude)
    check_doubles_amplitudes([b.o, b.o, b.v], amplitude)
    u1, u2 = amplitude.h, amplitude.phh

    f11 = intermediates.ip_adc2_f11
    f12 = mp.mp2_diffdm.ov          # t_ia
    f22 = intermediates.ip_adc2_f22

    # Calculate the spectroscopic amplitude x
    xi = einsum("j,ji->i", u1, f11)
    xa = einsum("j,ja->a", u1, f12) + einsum("ijb,ijba->a", u2, f22)

    return dot(xi, xi) + dot(xa, xa)


def pole_strength_ip_adc3(mp, amplitude, intermediates):
    check_singles_amplitudes([b.o], amplitude)
    check_doubles_amplitudes([b.o, b.o, b.v], amplitude)
    u1, u2 = amplitude.h, amplitude.phh

    f11 = intermediates.ip_adc3_f11
    f12 = intermediates.ip_adc3_f12
    f22 = intermediates.ip_adc3_f22

    # Calculate the spectroscopic amplitude x
    xi = einsum("j,ji->i", u1, f11)
    xa = einsum("j,ja->a", u1, f12) + einsum("ijb,ijba->a", u2, f22)

    return dot(xi, xi) + dot(xa, xa)


#
# Intermediates
#

@register_as_intermediate
def ip_adc2_f11(hf, mp, intermediates):
    # effective transition moments, oo part f_ij
    # Build Kronecker delta
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1.0)

    t2 = mp.t2(b.oovv)

    return d_oo - 0.25 * einsum("ilab,jlab->ij", t2, t2)


@register_as_intermediate
def ip_adc2_f22(hf, mp, intermediates):
    # effective transition moments, oovv part f_ijab
    return - 1/sqrt(2) * mp.t2(b.oovv)


@register_as_intermediate
def ip_adc3_f11(hf, mp, intermediates):
    # effective transition moments, oo part f_ij
    # Build Kronecker delta
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1.0)

    df = mp.df(b.ov)
    df2 = direct_sum("ia+kb->ikab", df, df).symmetrise((0, 1))

    t2 = mp.t2(b.oovv)

    return (d_oo
            - 0.25 * einsum("ilab,jlab->ij", t2, t2)
            + (+ 0.25 * einsum("jkab,ikab->ij", t2,
                               mp.t2eri(b.oovv, b.vv) / df2)
               + 0.25 * einsum("jkab,ikab->ij", t2,
                               mp.t2eri(b.oovv, b.oo) / df2)
               + einsum("jkab,ikab->ij", t2, mp.t2eri(b.oovv, b.ov) / df2)
               - einsum("jkab,kiab->ij", t2, mp.t2eri(b.oovv, b.ov) / df2))
            )


@register_as_intermediate
def ip_adc3_f12(hf, mp, intermediates):
    # effective transition moments, ov part f_ia
    return mp.mp2_diffdm.ov - (intermediates.sigma_ov
                               + intermediates.m_3_plus
                               + intermediates.m_3_minus
                               ) / mp.df(b.ov)


@register_as_intermediate
def ip_adc3_f22(hf, mp, intermediates):
    # effective transition moments, oovv part f_ijab

    df = mp.df(b.ov)
    df2 = direct_sum("ia+jb->ijab", df, df).symmetrise((2, 3))

    return (- 1/sqrt(2) * mp.t2(b.oovv)
            + 1/sqrt(2) * (0.5 * (mp.t2eri(b.oovv, b.oo)
                           + mp.t2eri(b.oovv, b.vv))
                           + ((mp.t2eri(b.oovv, b.ov)
                               ).antisymmetrise(2, 3)).antisymmetrise(0, 1)
                           ) / df2
            )


# TODO: Intermediates are also necessary in LazyMP, avoid redundancy
# TODO: Proper testing against Q-Chem of the pole strengths
@register_as_intermediate
def sigma_ov(hf, mp, intermediates):
    # Static self-energy, oo part \Sigma_{ij}(\infty)
    p0 = mp.mp2_diffdm
    return (+ einsum("jika,jk->ia", hf.ooov, p0.oo)
            + einsum("ijab,jb->ia", hf.oovv, p0.ov)
            - einsum("ibja,jb->ia", hf.ovov, p0.ov)
            + einsum("ibac,bc->ia", hf.ovvv, p0.vv))


@register_as_intermediate
def m_3_plus(hf, mp, intermediates):
    # Intermediate M_ia^{(3)+}, parts of the dynamic self-energy
    return (+ 1 * einsum("ijbc,jabc->ia", mp.t2oo, mp.t2eri(b.ovvv, b.ov))
            + 0.5 * einsum("ijbc,jabc->ia", mp.td2(b.oovv), hf.ovvv)
            - 0.25 * einsum(
                "ijbc,jabc->ia", mp.t2oo, mp.t2eri(b.ovvv, b.oo))
            )


@register_as_intermediate
def m_3_minus(hf, mp, intermediates):
    # Intermediate M_ia^{(3)-}, parts of the dynamic self-energy
    return (+ 0.5 * einsum("jkab,jkib->ia", mp.td2(b.oovv), hf.ooov)
            - 1 * einsum("jkab,jkib->ia", mp.t2oo, mp.t2eri(b.ooov, b.ov))
            - 0.25 * einsum(
                "jkab,jkib->ia", mp.t2oo, mp.t2eri(b.ooov, b.vv)
                )
            )

DISPATCH = {
    "ip_adc0": pole_strength_ip_adc0,
    "ip_adc1": pole_strength_ip_adc0,
    "ip_adc2": pole_strength_ip_adc2,
    "ip_adc3": pole_strength_ip_adc3,
}


def pole_strength(method, ground_state, amplitude, intermediates=None):
    """Compute the pole strength of the ionized state for the
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
