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
block to the result of applying the ADC matrix. `diagonal` is an `AmplitudeVector`
containing the expression to the diagonal of the ADC matrix from this block.
"""
AdcBlock = namedtuple("AdcBlock", ["apply", "diagonal"])


def block(ground_state, spaces, order, variant=None, intermediates=None):
    """
    Gets ground state, potentially intermediates, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied_space,
    while one would probably want in the long run that one can have an "o2" space,
    but not do CVS.
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError("Cannot run a general (non-core-valence approximated) "
                         "ADC method on top of a ground state with a "
                         "core-valence separation.")
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError("Cannot run a core-valence approximated ADC method on "
                         "top of a ground state without a "
                         "core-valence separation.")

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError("Could not dispatch: "
                         f"spaces={spaces} order={order} variant=variant")
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def block_ph_ph_0(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                             fCC.diagonal()))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)
            - einsum("IJ,Ja->Ia", fCC, ampl.ph)
        ))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_0 = block_ph_ph_0


def diagonal_pphh_pphh_0(hf):
    # Note: adcman similarly does not symmetrise the occupied indices
    #       (for both CVS and general ADC)
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum("-i-J+a+b->iJab",
                     hf.foo.diagonal(), fCC.diagonal(),
                     hf.fvv.diagonal(), hf.fvv.diagonal())
    return AmplitudeVector(pphh=res.symmetrise(2, 3))


def block_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_cvs_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


#
# 0th order coupling
#
def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_cvs_ph_pphh_0 = block_ph_pphh_0
block_cvs_pphh_ph_0 = block_pphh_ph_0


#
# 1st order main
#
def block_ph_ph_1(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
        - einsum("IaIa->Ia", CvCv)  # order 1
    ))

    def apply(ampl):
        return AmplitudeVector(ph=(                 # PT order
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
            - einsum("IJ,Ja->Ia", fCC, ampl.ph)     # 0
            - einsum("JaIb,Jb->Ia", CvCv, ampl.ph)  # 1
        ))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_1 = block_ph_ph_1


def diagonal_pphh_pphh_1(hf):
    # Fock matrix and ovov diagonal term (sometimes called "intermediate diagonal")
    dinterm_ov = (direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
                  - 2.0 * einsum("iaia->ia", hf.ovov)).evaluate()

    if hf.has_core_occupied_space:
        dinterm_Cv = (direct_sum("a-I->Ia", hf.fvv.diagonal(), hf.fcc.diagonal())
                      - 2.0 * einsum("IaIa->Ia", hf.cvcv)).evaluate()
        diag_oC = einsum("iJiJ->iJ", hf.ococ)
    else:
        dinterm_Cv = dinterm_ov
        diag_oC = einsum("ijij->ij", hf.oooo).symmetrise()

    diag_vv = einsum("abab->ab", hf.vvvv).symmetrise()
    return AmplitudeVector(pphh=(
        + direct_sum("ia+Jb->iJab", dinterm_ov, dinterm_Cv).symmetrise(2, 3)
        + direct_sum("iJ+ab->iJab", diag_oC, diag_vv)
    ))


def block_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(  # 0th order
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
            # 1st order
            + (
                -4 * einsum("ikac,kbjc->ijab", ampl.pphh, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijkl,klab->ijab", hf.oooo, ampl.pphh)
            + 0.5 * einsum("ijcd,abcd->ijab", ampl.pphh, hf.vvvv)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


def block_cvs_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            # 0th order
            + 2.0 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 1.0 * einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - 1.0 * einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
            # 1st order
            + (
                - 2.0 * einsum("iKac,KbJc->iJab", ampl.pphh, hf.cvcv)
                + 2.0 * einsum("icka,kJbc->iJab", hf.ovov, ampl.pphh)
            ).antisymmetrise(2, 3)
            + 1.0 * einsum("iJlK,lKab->iJab", hf.ococ, ampl.pphh)
            + 0.5 * einsum("iJcd,abcd->iJab", ampl.pphh, hf.vvvv)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


#
# 1st order coupling
#
def block_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + sqrt(2) * einsum("jKIb,jKab->Ia", hf.occv, ampl.pphh)
            - 1 / sqrt(2) * einsum("jIbc,jabc->Ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + sqrt(2) * einsum("jIKb,Ka->jIab",
                               hf.occv, ampl.ph).antisymmetrise(2, 3)
            - 1 / sqrt(2) * einsum("Ic,jcab->jIab", ampl.ph, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


#
# 2nd order main
#
def block_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
        - einsum("IaIa->Ia", hf.ovov)
        - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
    ))

    # Not used anywhere else, so kept as an anonymous intermediate
    term_t2_eri = (
        + einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
        + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo)
    ).evaluate()

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", i2, ampl.ph)
            - einsum("jaib,jb->ia", hf.ovov, ampl.ph)    # 1
            - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph)  # 2
        ))
    return AdcBlock(apply, diagonal)


def block_cvs_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), hf.fcc.diagonal())
        - einsum("IaIa->Ia", hf.cvcv)
    ))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", hf.fcc, ampl.ph)
            - einsum("JaIb,Jb->Ia", hf.cvcv, ampl.ph)
        ))
    return AdcBlock(apply, diagonal)


#
# 2nd order coupling
#
def block_ph_pphh_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", pia_ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, pib_ovvv)
            + einsum("icab,jkcd,jkbd->ia", hf.ovvv, ampl.pphh, mp.t2oo)  # 2nd
            + einsum("ijka,jlbc,klbc->ia", hf.ooov, mp.t2oo, ampl.pphh)  # 2nd
        ))
    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(1 / sqrt(2)) * (
            + 2.0 * einsum("lKIc,lKac->Ia", pia_occv, ampl.pphh)
            - einsum("lIcd,lacd->Ia", ampl.pphh, pib_ovvv)
            - einsum("jIKa,ljcd,lKcd->Ia", hf.occv, mp.t2oo, ampl.pphh)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(
            (
                + einsum("ic,jcab->ijab", ampl.ph, pib_ovvv)
                + einsum("lkic,kc,jlab->ijab", hf.ooov, ampl.ph, mp.t2oo)  # 2st
            ).antisymmetrise(0, 1)
            + (
                - einsum("ijka,kb->ijab", pia_ooov, ampl.ph)
                - einsum("ijac,kbcd,kd->ijab", mp.t2oo, hf.ovvv, ampl.ph)  # 2st
            ).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(1 / sqrt(2)) * (
            - 2.0 * einsum("jIKa,Kb->jIab", pia_occv, ampl.ph).antisymmetrise(2, 3)
            - einsum("Ic,jcab->jIab", ampl.ph, pib_ovvv)
            - einsum("lKIc,Kc,jlab->jIab", hf.occv, ampl.ph, mp.t2oo)
        ))
    return AdcBlock(apply, 0)


#
# 3rd order main
#
def block_ph_ph_3(hf, mp, intermediates):
    if hf.has_core_occupied_space:
        m11 = intermediates.cvs_adc3_m11
    else:
        m11 = intermediates.adc3_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_3 = block_ph_ph_3


#
# Intermediates
#

@register_as_intermediate
def adc2_i1(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.fvv term.
    return hf.fvv + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def adc2_i2(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.foo term.
    return hf.foo - 0.5 * einsum("ikab,jkab->ij", mp.t2oo, hf.oovv).symmetrise()


def adc3_i1(hf, mp, intermediates):
    # Used for both CVS and general
    td2 = mp.td2(b.oovv)
    p0 = intermediates.cvs_p0 if hf.has_core_occupied_space else mp.mp2_diffdm

    t2eri_sum = (
        + einsum("jicb->ijcb", mp.t2eri(b.oovv, b.ov))  # t2eri4
        - 0.25 * mp.t2eri(b.oovv, b.vv)                 # t2eri5
    )
    return (
        (  # symmetrise a<>b
            + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ijac,ijcb->ab", mp.t2oo, t2eri_sum)
            - 2.0 * einsum("iabc,ic->ab", hf.ovvv, p0.ov)
        ).symmetrise()
        + einsum("iajb,ij->ab", hf.ovov, p0.oo)
        + einsum("acbd,cd->ab", hf.vvvv, p0.vv)
    )


def adc3_i2(hf, mp, intermediates):
    # Used only for general
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    # t2eri4 + t2eri3 / 4
    t2eri_sum = mp.t2eri(b.oovv, b.ov) + 0.25 * mp.t2eri(b.oovv, b.oo)
    return (
        (  # symmetrise i<>j
            + 0.5 * einsum("ikab,jkab->ij", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ikab,jkab->ij", mp.t2oo, t2eri_sum)
            + 2.0 * einsum("kija,ka->ij", hf.ooov, p0.ov)
        ).symmetrise()
        - einsum("ikjl,kl->ij", hf.oooo, p0.oo)
        - einsum("iajb,ab->ij", hf.ovov, p0.vv)
    )


def cvs_adc3_i2(hf, mp, intermediates):
    cvs_p0 = intermediates.cvs_p0
    return (
        + 2.0 * einsum("kIJa,ka->IJ", hf.occv, cvs_p0.ov).symmetrise()
        - 1.0 * einsum("kIlJ,kl->IJ", hf.ococ, cvs_p0.oo)
        - 1.0 * einsum("IaJb,ab->IJ", hf.cvcv, cvs_p0.vv)
    )


@register_as_intermediate
def adc3_m11(hf, mp, intermediates):
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    t2eri_sum = (
        + 2.0 * mp.t2eri(b.oovv, b.ov).symmetrise((0, 1), (2, 3))  # t2eri4
        + 0.5 * mp.t2eri(b.oovv, b.vv)                             # t2eri5
        + 0.5 * mp.t2eri(b.oovv, b.oo)                             # t2eri3
    )
    return (
        + einsum("ij,ab->iajb", d_oo, hf.fvv + i1)
        - einsum("ij,ab->iajb", hf.foo - i2, d_vv)
        - einsum("jaib->iajb", hf.ovov)
        - (  # symmetrise i<>j and a<>b
            + einsum("jkbc,ikac->iajb", hf.oovv, mp.t2oo + td2)
            - einsum("jkbc,ikac->iajb", mp.t2oo, t2eri_sum)
            - einsum("ibac,jc->iajb", hf.ovvv, 2.0 * p0.ov)
            - einsum("ikja,kb->iajb", hf.ooov, 2.0 * p0.ov)
            - einsum("jaic,bc->iajb", hf.ovov, p0.vv)
            + einsum("ik,jakb->iajb", p0.oo, hf.ovov)
            + einsum("ibkc,kajc->iajb", hf.ovov, 2.0 * t2sq)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("icjd,klac,klbd->iajb", hf.ovov, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("icjd,acbd->iajb", hf.ovov,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        + 0.5 * einsum("ikcd,jlcd,kalb->iajb", mp.t2oo, mp.t2oo, hf.ovov)
        - einsum("iljk,kalb->iajb", hf.oooo, t2sq)
        - einsum("idjc,acbd->iajb", t2sq, hf.vvvv)
    )


@register_as_intermediate
def cvs_adc3_m11(hf, mp, intermediates):
    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = cvs_adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()

    # Build two Kronecker deltas
    d_cc = zeros_like(hf.fcc)
    d_vv = zeros_like(hf.fvv)
    d_cc.set_mask("II", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (
        + einsum("IJ,ab->IaJb", d_cc, hf.fvv + i1)
        - einsum("IJ,ab->IaJb", hf.fcc - i2, d_vv)
        - einsum("JaIb->IaJb", hf.cvcv)
        + (  # symmetrise I<>J and a<>b
            + einsum("JaIc,bc->IaJb", hf.cvcv, intermediates.cvs_p0.vv)
            - einsum("kIJa,kb->IaJb", hf.occv, 2.0 * intermediates.cvs_p0.ov)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("IcJd,klac,klbd->IaJb", hf.cvcv, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("IcJd,acbd->IaJb", hf.cvcv,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        - einsum("lIkJ,kalb->IaJb", hf.ococ, t2sq)
    )


@register_as_intermediate
def adc3_pia(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ooov term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ooov                                            # 1st order
        - 2.0 * mp.t2eri(b.ooov, b.ov).antisymmetrise(0, 1)  # 2nd order
        - 0.5 * mp.t2eri(b.ooov, b.vv)                       # 2nd order
    )


@register_as_intermediate
def cvs_adc3_pia(hf, mp, intermediates):
    # Perturbation theory in CVS-ADC coupling block:
    #       1st                     2nd
    return hf.occv - einsum("jlac,lKIc->jIKa", mp.t2oo, hf.occv)


@register_as_intermediate
def adc3_pib(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ovvv term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ovvv                                            # 1st order
        + 2.0 * mp.t2eri(b.ovvv, b.ov).antisymmetrise(2, 3)  # 2nd order
        - 0.5 * mp.t2eri(b.ovvv, b.oo)                       # 2nd order
    )
