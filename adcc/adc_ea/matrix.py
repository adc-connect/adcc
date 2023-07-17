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
from adcc.functions import direct_sum, einsum
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

    It is assumed largely, that CVS is equivalent to
    mp.has_core_occupied_space, while one would probably want in the long run
    that one can have an "o2" space, but not do CVS.
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
def block_p_p_0(hf, mp, intermediates):
    # M_{11}
    def apply(ampl):
        return AmplitudeVector(p=einsum("ab,b->a", hf.fvv, ampl.p))
    diagonal = AmplitudeVector(p=hf.fvv.diagonal())
    return AdcBlock(apply, diagonal)


def diagonal_pph_pph_0(hf):
    res = direct_sum("-i+a+b->iab",
                     hf.foo.diagonal(), hf.fvv.diagonal(), hf.fvv.diagonal())
    return AmplitudeVector(pph=res.symmetrise(1, 2))


def block_pph_pph_0(hf, mp, intermediates):
    # M_{22}
    def apply(ampl):
        return AmplitudeVector(pph=(
            - einsum("jab,ij->iab", ampl.pph, hf.foo)
            + 2 * einsum("ac,icb->iab", hf.fvv, ampl.pph).antisymmetrise(1, 2)
        ))
    return AdcBlock(apply, diagonal_pph_pph_0(hf))


#
# 1st order main
#
def block_p_p_1(hf, mp, intermediates):
    # M_{11}, same as ADC(0)
    return block_p_p_0(hf, mp, intermediates)


def block_pph_pph_1(hf, mp, intermediates):
    # M_{22}
    def apply(ampl):
        return AmplitudeVector(pph=(
            - einsum("jab,ij->iab", ampl.pph, hf.foo)
            + 2 * einsum("ac,icb->iab", hf.fvv, ampl.pph).antisymmetrise(1, 2)
            + 0.5 * einsum("abcd,icd->iab", hf.vvvv, ampl.pph)
            - 2 * einsum("icka,kcb->iab", hf.ovov, ampl.pph
                         ).antisymmetrise(1, 2)
        ))
    return AdcBlock(apply, diagonal_pph_pph_0(hf))


#
# 1st order coupling
#
def block_p_pph_1(hf, mp, intermediates):
    # M_{12}
    def apply(ampl):
        return AmplitudeVector(p=(
           - 1 / sqrt(2) * einsum("jabc,jbc->a", hf.ovvv, ampl.pph)))
    return AdcBlock(apply, 0)


def block_pph_p_1(hf, mp, intermediates):
    # M_{21}
    def apply(ampl):
        return AmplitudeVector(pph=(
            - 1 / sqrt(2) * einsum("icab,c->iab", hf.ovvv, ampl.p)))
    return AdcBlock(apply, 0)


#
# 2nd order main
#
def block_p_p_2(hf, mp, intermediates):
    # M_{11}
    i1 = intermediates.adc2_ea_i1
    diagonal = AmplitudeVector(p=i1.diagonal())

    def apply(ampl):
        return AmplitudeVector(p=einsum("ab,b->a", i1, ampl.p))
    return AdcBlock(apply, diagonal)


#
# 2nd order coupling
#
def block_p_pph_2(hf, mp, intermediates):
    # M_{12}
    i2 = intermediates.adc3_ea_i2

    def apply(ampl):
        return AmplitudeVector(p=(
           + 1 / sqrt(2) * einsum("jabc,jbc->a", i2, ampl.pph)))
    return AdcBlock(apply, 0)


def block_pph_p_2(hf, mp, intermediates):
    # M_{21}
    i2 = intermediates.adc3_ea_i2

    def apply(ampl):
        return AmplitudeVector(pph=(
            + 1 / sqrt(2) * einsum("icab,c->iab", i2, ampl.p)))
    return AdcBlock(apply, 0)


#
# 3rd order main
#
def block_p_p_3(hf, mp, intermediates):
    # M_{11}
    i1 = intermediates.adc3_ea_i1
    diagonal = AmplitudeVector(p=i1.diagonal())

    def apply(ampl):
        return AmplitudeVector(p=einsum("ab,b->a", i1, ampl.p))
    return AdcBlock(apply, diagonal)


#
# Intermediates
#

@register_as_intermediate
def adc2_ea_i1(hf, mp, intermediates):
    return hf.fvv + 0.5 * einsum("ijac,ijbc->ab",
                                 mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def adc3_ea_i1(hf, mp, intermediates):
    return (
        hf.fvv + (
            + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, hf.oovv)
            - 0.25 * einsum("ijbc,ijac->ab", mp.t2oo, mp.t2eri(b.oovv, b.oo))
            + 0.5 * einsum("ijbc,ijca->ab", mp.t2oo, mp.t2eri(b.oovv, b.vv))
            + einsum("ijbc,jiac->ab", mp.t2oo, mp.t2eri(b.oovv, b.ov))
            - 2 * einsum("ijbc,jica->ab", mp.t2oo, mp.t2eri(b.oovv, b.ov))
        ).symmetrise()
        + intermediates.sigma_vv
    )


@register_as_intermediate
def adc3_ea_i2(hf, mp, intermediates):
    return - hf.ovvv - intermediates.adc3_pib.antisymmetrise(2, 3)


@register_as_intermediate
def adc3_pib(hf, mp, intermediates):
    return (2 * mp.t2eri(b.ovvv, b.ov).antisymmetrise(2, 3)
            - 0.5 * mp.t2eri(b.ovvv, b.oo))


@register_as_intermediate
def sigma_vv(hf, mp, intermediates):
    # Static self-energy, oo part \Sigma_{ij}(\infty)
    p0 = mp.mp2_diffdm
    return (einsum("iajb,ij->ab", hf.ovov, p0.oo)
            + 2 * einsum("iacb,ic->ab", hf.ovvv, p0.ov)
            + einsum("acbd,cd->ab", hf.vvvv, p0.vv)).symmetrise()
