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
from collections import namedtuple

from adcc import block as b
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate
from adcc.AmplitudeVector import AmplitudeVector

__all__ = ["block"]

# TODO One thing one could still do to improve timings is implement a "fast einsum"
#      that does not call opt_einsum, but directly dispatches to libadcc. This could
#      lower the call overhead in the applies for the cases where we have only a
#      trivial einsum to do. For the moment I'm not convinced that is worth the
#      effort ... I suppose it only makes a difference for the cheaper ADC variants
#      (ADC(0), ADC(1), CVS-ADC(0-2)-x), but then on the other hand they are not
#      really so much our focus.


#
# Dispatch routine
#
"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix. `diagonal` is an
`AmplitudeVector` containing the expression to the diagonal of the ADC matrix
from this block.
"""
AdcBlock = namedtuple("AdcBlock", ["apply", "diagonal"])


def block(ground_state, spaces, order, variant=None, intermediates=None):
    """
    Gets ground state, potentially intermediates, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied
    space, while one would probably want in the long run that one can have an
    "o2" space, but not do CVS.
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError("Cannot run a general (non-core-valence approximated)"
                         " ADC method on top of a ground state with a "
                         "core-valence separation.")
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError("Cannot run a core-valence approximated ADC method on"
                         " top of a ground state without a "
                         "core-valence separation.")

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError("Could not dispatch: "
                         f"spaces={spaces} order={order} variant=variant")
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def block_h_h_0(hf, mp, intermediates):
    # M_{11}
    def apply(ampl):
        return AmplitudeVector(h=-einsum("ij,j->i", hf.foo, ampl.h))
    diagonal = AmplitudeVector(h=-hf.foo.diagonal())
    return AdcBlock(apply, diagonal)


def diagonal_phh_phh_0(hf):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum("-i-J+a->iJa",
                     hf.foo.diagonal(), fCC.diagonal(), hf.fvv.diagonal())
    return AmplitudeVector(phh=res)


def block_phh_phh_0(hf, mp, intermediates):
    # M_{22}
    def apply(ampl):
        return AmplitudeVector(phh=(
            + einsum("ab,ijb->ija", hf.fvv, ampl.phh)
            - 2 * einsum("ik,kja->ija", hf.foo, ampl.phh).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_phh_phh_0(hf))


#
# 1st order main
#
def block_h_h_1(hf, mp, intermediates):
    # M_{11}, same as ADC(0)
    return block_h_h_0(hf, mp, intermediates)


def diagonal_phh_phh_1(hf):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    i1 = direct_sum("-i-J+a->iJa",
                    hf.foo.diagonal(), fCC.diagonal(), hf.fvv.diagonal())

    # Build Kronecker delta
    d_vv = zeros_like(hf.fvv)
    d_vv.set_mask("aa", 1.0)

    i2 = einsum("ijij,aa->ija", hf.oooo, d_vv)
    res = i1 + i2
    return AmplitudeVector(phh=res.symmetrise(0, 1))


def block_phh_phh_1(hf, mp, intermediates):
    # M_{22}
    def apply(ampl):
        return AmplitudeVector(phh=(
            + einsum("ac,ijc->ija", hf.fvv, ampl.phh)
            - 2 * einsum("ik,kja->ija", hf.foo, ampl.phh).antisymmetrise(0, 1)
            + 0.5 * einsum("ijkl,kla->ija", hf.oooo, ampl.phh)
            - 2 * einsum("kaic,kjc->ija", hf.ovov, ampl.phh
                         ).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_phh_phh_1(hf))


#
# 1st order coupling
#
def block_h_phh_1(hf, mp, intermediates):
    # M_{12}
    def apply(ampl):
        return AmplitudeVector(h=(
           + 1 / sqrt(2) * einsum("jkib,jkb->i", hf.ooov, ampl.phh)))
    return AdcBlock(apply, 0)


def block_phh_h_1(hf, mp, intermediates):
    # M_{21}
    def apply(ampl):
        return AmplitudeVector(phh=(
            + 1 / sqrt(2) * einsum("ijka,k->ija", hf.ooov, ampl.h)))
    return AdcBlock(apply, 0)


#
# 2nd order main
#
def block_h_h_2(hf, mp, intermediates):
    # M_{11}
    i1 = intermediates.adc2_ip_i1
    diagonal = AmplitudeVector(h=i1.diagonal())

    def apply(ampl):
        return AmplitudeVector(h=einsum("ij,j->i", i1, ampl.h))
    return AdcBlock(apply, diagonal)


#
# 2nd order coupling
#
def block_h_phh_2(hf, mp, intermediates):
    # M_{12}
    i2 = intermediates.adc3_ip_i2

    def apply(ampl):
        return AmplitudeVector(h=(
           + 1 / sqrt(2) * einsum("jkib,jkb->i", i2, ampl.phh)))
    return AdcBlock(apply, 0)


def block_phh_h_2(hf, mp, intermediates):
    # M_{21}
    i2 = intermediates.adc3_ip_i2

    def apply(ampl):
        return AmplitudeVector(phh=(
            + 1 / sqrt(2) * einsum("ijka,k->ija", i2, ampl.h)))
    return AdcBlock(apply, 0)


#
# 3rd order main
#
def block_h_h_3(hf, mp, intermediates):
    # M_{11}
    i1 = intermediates.adc3_ip_i1
    diagonal = AmplitudeVector(h=i1.diagonal())

    def apply(ampl):
        return AmplitudeVector(h=einsum("ij,j->i", i1, ampl.h))
    return AdcBlock(apply, diagonal)


#
# Intermediates
#

@register_as_intermediate
def adc2_ip_i1(hf, mp, intermediates):
    return - hf.foo + 0.5 * einsum("ikab,jkab->ij",
                                   mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def adc3_ip_i1(hf, mp, intermediates):
    return (
        - hf.foo + (
            + 0.5 * einsum("ikab,jkab->ij", mp.t2oo, hf.oovv)
            - 0.25 * einsum("jkab,ikab->ij", mp.t2oo, mp.t2eri(b.oovv, b.vv))
            + 0.5 * einsum("jkab,ikba->ij", mp.t2oo, mp.t2eri(b.oovv, b.oo))
            + einsum("jkab,kiab->ij", mp.t2oo, mp.t2eri(b.oovv, b.ov))
            - 2 * einsum("jkab,ikab->ij", mp.t2oo, mp.t2eri(b.oovv, b.ov))
        ).symmetrise()
        - intermediates.sigma_oo
    )


@register_as_intermediate
def adc3_ip_i2(hf, mp, intermediates):
    return hf.ooov - intermediates.adc3_pia.antisymmetrise(0, 1)


@register_as_intermediate
def adc3_pia(hf, mp, intermediates):
    return (2 * mp.t2eri(b.ooov, b.ov).antisymmetrise(0, 1)
            + 0.5 * mp.t2eri(b.ooov, b.vv))


@register_as_intermediate
def sigma_oo(hf, mp, intermediates):
    # Static self-energy, oo part \Sigma_{ij}(\infty)
    p0 = mp.mp2_diffdm
    return (einsum("ikjl,kl->ij", hf.oooo, p0.oo)
            + 2 * einsum("ikja,ka->ij", hf.ooov, p0.ov)
            + einsum("iajb,ab->ij", hf.ovov, p0.vv)).symmetrise()
