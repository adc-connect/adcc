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
                         f"spaces={spaces} order={order} variant={variant}. "
                         "Probably the secular matrix is not implemented for "
                         "the requested method.")
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
    # since we use orbital energies to construct the diagonal,
    # we have to symmetrise both occupied and virtual indices!
    # (for dfs a single symmetrisation would be fine)
    # -> distinct treatment of CVS-ADC
    # If the occupied indices are not symmetrised the preconditioned
    # residuals do no longer have the correct symmetry in the davidson:
    # the antisymmetry of the occupied indices is missing
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum(
        "-i-J+a+b->iJab",
        hf.foo.diagonal(), fCC.diagonal(), hf.fvv.diagonal(), hf.fvv.diagonal()
    ).symmetrise(2, 3)
    if not hf.has_core_occupied_space:
        res = res.symmetrise(0, 1)
    return AmplitudeVector(pphh=res)


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


def diagonal_ppphhh_ppphhh_0(hf):
    res = direct_sum("-i-j-k+a+b+c->ijkabc",
                     hf.foo.diagonal(), hf.foo.diagonal(), hf.foo.diagonal(),
                     hf.fvv.diagonal(), hf.fvv.diagonal(), hf.fvv.diagonal())
    return AmplitudeVector(ppphhh=res.symmetrise(3, 4, 5))


def block_ppphhh_ppphhh_0(hf, mp, intermediates):
    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(ppphhh=(
            # Prefactor of 3: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/12,
            # yielding effectively 3 distinct terms.
            3 * (
                # N^7: O^3V^4 / N^6: O^3V^3
                + 1 * einsum("ijkabd,cd->ijkabc", ur3, hf.fvv)
                # N^7: O^4V^3 / N^6: O^3V^3
                + 1 * einsum("iklabc,jl->ijkabc", ur3, hf.foo)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
        ))
    return AdcBlock(apply, diagonal_ppphhh_ppphhh_0(hf))


#
# 0th order coupling
#
def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_cvs_ph_pphh_0 = block_ph_pphh_0
block_cvs_pphh_ph_0 = block_pphh_ph_0


def block_ph_ppphhh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ppphhh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ppphhh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ppphhh_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


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
        # Both objects are 'df like' and thus it is sufficient to either symmetrise
        # the occupied or virtual indices
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


def block_ph_ppphhh_1(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ppphhh_ph_1(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ppphhh_1(hf, mp, intermediates):
    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(pphh=(
            2 * (
                # N^7: O^3V^4 / N^6: O^3V^3
                - 3 / 2 * einsum("ijkacd,kbcd->ijab",
                                 ur3, hf.ovvv)
            ).antisymmetrise(2, 3)
            + 2 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 3 / 2 * einsum("iklabc,kljc->ijab",
                                 ur3, hf.ooov)
            ).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, 0)


def block_ppphhh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(ppphhh=(
            # Prefactor of 9: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/4,
            # yielding effectively 9 distinct terms.
            9 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 3 * einsum("ilab,jklc->ijkabc", ur2, hf.ooov)
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 3 * einsum("ijad,kdbc->ijkabc", ur2, hf.ovvv)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
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


def diagonal_pphh_pphh_2(hf, mp):
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

    # 2nd order intermediates
    t2_1 = mp.t2(b.oovv)
    dinterm_o = 0.5 * einsum("ikcd,ikcd->i", t2_1, hf.oovv).evaluate()
    dinterm_ovv = einsum("jkab,jkab->jab", t2_1, hf.oovv).evaluate()

    dinterm_v = 0.5 * einsum("klbc,klbc->b", t2_1, hf.oovv).evaluate()
    dinterm_oov = einsum("ijac,ijac->ija", t2_1, hf.oovv).evaluate()

    dinterm_oo = - 0.5 * einsum("ijcd,ijcd->ij", t2_1, hf.oovv).evaluate()
    dinterm_vv = - 0.5 * einsum("klab,klab->ab", t2_1, hf.oovv).evaluate()

    dinterm_ov_2 = - einsum("ikac,ikac->ia", t2_1, hf.oovv).evaluate()

    return AmplitudeVector(pphh=(
        # 0th + 1st order
        + direct_sum("ia+Jb->iJab", dinterm_ov, dinterm_Cv).symmetrise(2, 3)
        + direct_sum("iJ+ab->iJab", diag_oC, diag_vv)
        # 2nd order
        + 2 * direct_sum("i+jab->ijab", dinterm_o, dinterm_ovv).symmetrise(0, 1)
        + 2 * direct_sum("ija+b->ijab", dinterm_oov, dinterm_v).symmetrise(2, 3)
        + direct_sum("ij+ab->ijab", dinterm_oo, dinterm_vv)
        + 2 * direct_sum("ia+jb->ijab", dinterm_ov_2, dinterm_ov_2).symmetrise(2, 3)
    ))


def block_pphh_pphh_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(pphh=(  # 0th order
            + 2 * einsum("ijac,bc->ijab", ur2, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ur2).antisymmetrise(0, 1)
            # 1st order
            + (
                -4 * einsum("ikac,kbjc->ijab", ur2, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijkl,klab->ijab", hf.oooo, ur2)
            + 0.5 * einsum("ijcd,abcd->ijab", ur2, hf.vvvv)
            # 2nd order
            + 2 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", ur2, hf.oovv), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("jl,ilab->ijab",
                                einsum("jkcd,klcd->jl", ur2, t2_1), hf.oovv)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ik,jkab->ijab",
                                einsum("ilcd,klcd->ik", t2_1, hf.oovv), ur2)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ik,jkab->ijab",
                                einsum("klcd,ilcd->ik", t2_1, hf.oovv), ur2)
            ).antisymmetrise(0, 1)
            + (
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, hf.oovv), t2_1)
                # N^6:  O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", ur2, t2_1), hf.oovv)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijcd,klcd->ijkl", t2_1, hf.oovv), ur2)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("klcd,ijcd->ijkl", t2_1, hf.oovv), ur2)
            )
            + 2 * (
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("ac,ijbc->ijab",
                                einsum("klad,klcd->ac", t2_1, hf.oovv), ur2)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("ac,ijbc->ijab",
                                einsum("klcd,klad->ac", t2_1, hf.oovv), ur2)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", ur2, hf.oovv), t2_1)
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("klbc,klcd->bd", ur2, t2_1), hf.oovv)
            ).antisymmetrise(2, 3)
            + 4 * (
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("jlad,ilbd->ijab",
                               einsum("jkac,klcd->jlad", ur2, hf.oovv), t2_1)
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("jlad,ilbd->ijab",
                               einsum("jkac,klcd->jlad", ur2, t2_1), hf.oovv)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_2(hf, mp))


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


def block_ph_ppphhh_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(ph=(
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 * einsum("ijld,jald->ia",
                         einsum("ijkbcd,klbc->ijld", ur3, t2_1), hf.ovov)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 / 2 * einsum("ibde,aebd->ia",
                             einsum("ijkbcd,jkce->ibde", ur3, t2_1), hf.vvvv)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 * einsum("kacd,ickd->ia",
                         einsum("jklabc,jlbd->kacd", ur3, t2_1), hf.ovov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 / 2 * einsum("jlma,imjl->ia",
                             einsum("jklabc,kmbc->jlma", ur3, t2_1), hf.oooo)
        ))
    return AdcBlock(apply, 0)


def block_ppphhh_ph_2(hf, mp, intermediates):
    t2_1 = mp.t2(b.oovv)

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(ppphhh=(
            # Prefactor of 18: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/2,
            # yielding effectively 18 distinct terms.
            18 * (
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 6 * einsum("iklc,jlab->ijkabc",
                                 einsum("id,kdlc->iklc", ur1, hf.ovov), t2_1)
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 6 * einsum("kacd,ijbd->ijkabc",
                                 einsum("la,kdlc->kacd", ur1, hf.ovov), t2_1)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
            # Prefactor of 9: the antisymmetrisation generates 36 terms with
            # an overall factor of 1/36, so each term appears with weight 1/4,
            # yielding effectively 9 distinct terms.
            + 9 * (
                # N^7: O^3V^4 / N^6: O^3V^3
                - 1 / 6 * einsum("ibce,jkae->ijkabc",
                                 einsum("id,bcde->ibce", ur1, hf.vvvv), t2_1)
                # N^7: O^4V^3 / N^6: O^3V^3
                - 1 / 6 * einsum("jkma,imbc->ijkabc",
                                 einsum("la,jklm->jkma", ur1, hf.oooo), t2_1)
            ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
        ))
        return
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
# 3rd order coupling
#
def block_ph_pphh_3(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    p0_2 = mp.diffdm(level=2)
    p0_2_oo, p0_2_vv, t1_2 = p0_2.oo, p0_2.vv, p0_2.ov

    t2sq = intermediates.t2sq
    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    t2eri_1 = mp.t2eri(b.ooov, b.vv).evaluate()
    t2eri_2 = mp.t2eri(b.ooov, b.ov).evaluate()
    t2eri_6 = mp.t2eri(b.ovvv, b.oo).evaluate()
    t2eri_7 = mp.t2eri(b.ovvv, b.ov).evaluate()

    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", pia_ooov, ur2)
            + einsum("ijbc,jabc->ia", ur2, pib_ovvv)
            + einsum("icab,jkcd,jkbd->ia", hf.ovvv, ur2, t2_1)  # 2nd
            + einsum("ijka,jlbc,klbc->ia", hf.ooov, t2_1, ur2)  # 2nd
            # 3rd
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 * einsum("iklb,kbla->ia",
                         einsum("jkbc,jlic->iklb", ur2, hf.ooov), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("kl,ikla->ia",
                         einsum("jlbc,jkbc->kl", ur2, t2_2), hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("cd,idac->ia",
                         einsum("jkbd,jkbc->cd", ur2, t2_2), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("kl,lika->ia",
                           einsum("jkbc,jlbc->kl", ur2, t2_1), t2eri_2)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("bd,ibda->ia",
                           einsum("jkbc,jkcd->bd", ur2, t2_1), t2eri_7)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.5 * einsum("lc,ilac->ia",
                           einsum("jkbd,jklbcd->lc", ur2, t3_2), hf.oovv)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 1 * einsum("ijbd,jabd->ia",
                         einsum("jkbc,idkc->ijbd", ur2, t2sq), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("lb,ialb->ia",
                           einsum("jkbc,jklc->lb", ur2, hf.ooov), t2sq)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("jd,iajd->ia",
                           einsum("jkbc,kdbc->jd", ur2, hf.ovvv), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("kl,ilka->ia",
                            einsum("jkbc,jlbc->kl", ur2, t2_1), t2eri_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("cd,icad->ia",
                            einsum("jkbc,jkbd->cd", ur2, t2_1), t2eri_6)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ilmd,lmad->ia",
                             einsum("jklm,jkid->ilmd",
                                    einsum("jkbc,lmbc->jklm", ur2, t2_1),
                                    hf.ooov), t2_1)
            # N^6: O^4V^2 / N^4: O^1V^3
            + 1 / 8 * einsum("ilbc,labc->ia",
                             einsum("ijkl,jkbc->ilbc",
                                    einsum("ilde,jkde->ijkl", t2_1, t2_1),
                                    ur2), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 1 * einsum("ikbd,kbda->ia",
                         einsum("ijbc,jkcd->ikbd", ur2, t2_1), t2eri_7)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("id,ad->ia",
                           einsum("ijbc,jdbc->id", ur2, hf.ovvv), p0_2_vv)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ikbc,kabc->ia",
                           einsum("ijbc,jk->ikbc", ur2, p0_2_oo), hf.ovvv)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.5 * einsum("iklc,klac->ia",
                           einsum("ijbd,jklbcd->iklc", ur2, t3_2), hf.oovv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ijbd,jabd->ia",
                         einsum("ijbc,cd->ijbd", ur2, p0_2_vv), hf.ovvv)
            # N^5: O^1V^4 / N^4: V^4
            - 1 * einsum("ibcd,abcd->ia",
                         einsum("ijcd,jb->ibcd", ur2, t1_2), hf.vvvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkc,jcka->ia",
                         einsum("ikbc,jb->ijkc", ur2, t1_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 2 * einsum("ikcd,kdac->ia",
                         einsum("ijbd,jkbc->ikcd", ur2, t2_2), hf.ovvv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkl,jkla->ia",
                           einsum("ilbc,jkbc->ijkl", ur2, t2_2), hf.ooov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ijkl,klja->ia",
                             einsum("ijbc,klbc->ijkl", ur2, t2_1), t2eri_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("ka,ik->ia",
                            einsum("klde,lade->ka", t2_1, hf.ovvv),
                            einsum("ijbc,jkbc->ik", ur2, t2_1))
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.5 * einsum("ijkc,jkac->ia",
                           einsum("jklbcd,ilbd->ijkc", t3_2, hf.oovv), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,jkab->ia",
                         einsum("klbc,lijc->ijkb", t2_1, t2eri_2), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,jkab->ia",
                         einsum("klib,jl->ijkb", hf.ooov, p0_2_oo), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("jkla,ijkl->ia",
                         einsum("klab,jb->jkla", ur2, t1_2), hf.oooo)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkc,jkac->ia",
                         einsum("jb,ickb->ijkc", t1_2, hf.ovov), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 2 * einsum("ijlb,jlab->ia",
                         einsum("jkbc,iklc->ijlb", t2_2, hf.ooov), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkb,jkab->ia",
                           einsum("jkic,bc->ijkb", hf.ooov, p0_2_vv), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("la,il->ia",
                           einsum("jkab,jklb->la", ur2, hf.ooov), p0_2_oo)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijkd,jkad->ia",
                           einsum("jkbc,idbc->ijkd", t2_2, hf.ovvv), ur2)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 1 / 8 * einsum("ijkb,jkab->ia",
                             einsum("jkcd,ibcd->ijkb", t2_1, t2eri_6), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkb,jkab->ia",
                            einsum("ic,jkbc->ijkb",
                                   einsum("lmcd,lmid->ic", t2_1, hf.ooov),
                                   t2_1), ur2)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_3(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib
    p0_2 = mp.diffdm(level=2)
    p0_2_oo, p0_2_vv, t1_2 = p0_2.oo, p0_2.vv, p0_2.ov

    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    t2sq = intermediates.t2sq
    t2eri_1 = mp.t2eri(b.ooov, b.vv).evaluate()
    t2eri_2 = mp.t2eri(b.ooov, b.ov).evaluate()
    t2eri_6 = mp.t2eri(b.ovvv, b.oo).evaluate()
    t2eri_7 = mp.t2eri(b.ovvv, b.ov).evaluate()

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(pphh=(
            (
                + einsum("ic,jcab->ijab", ur1, pib_ovvv)
                + einsum("lkic,kc,jlab->ijab", hf.ooov, ur1, t2_1)  # 2st
            ).antisymmetrise(0, 1)
            + (
                - einsum("ijka,kb->ijab", pia_ooov, ur1)
                - einsum("ijac,kbcd,kd->ijab", t2_1, hf.ovvv, ur1)  # 2st
            ).antisymmetrise(2, 3)
            # 3rd order
            + 2 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ik,jkab->ijab",
                               einsum("lc,klic->ik", ur1, hf.ooov), t2_2)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("il,jlab->ijab",
                                einsum("kc,lkic->il", ur1, t2eri_2), t2_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("id,jdab->ijab",
                                einsum("kc,idkc->id", ur1, t2sq), hf.ovvv)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("jl,ilab->ijab",
                                 einsum("kc,kljc->jl", ur1, t2eri_1), t2_1)
                # N^5: O^1V^4 / N^4: V^4
                + 0.5 * einsum("iabc,jc->ijab",
                               einsum("id,abcd->iabc", ur1, hf.vvvv), t1_2)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkl,klab->ijab",
                                einsum("ic,kljc->ijkl", ur1, hf.ooov), t2_2)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("jd,idab->ijab",
                                einsum("jc,cd->jd", ur1, p0_2_vv), hf.ovvv)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("jkab,ik->ijab",
                                einsum("jc,kcab->jkab", ur1, hf.ovvv), p0_2_oo)
                # N^7: O^4V^3 / N^6: O^3V^3
                - 0.25 * einsum("jklc,iklabc->ijab",
                                einsum("jd,klcd->jklc", ur1, hf.oovv), t3_2)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 16 * einsum("ijkl,klab->ijab",
                                  einsum("jc,klic->ijkl", ur1, t2eri_1), t2_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 1 / 8 * einsum("jk,ikab->ijab",
                                 einsum("kc,jc->jk",
                                        einsum("klde,lcde->kc", t2_1, hf.ovvv),
                                        ur1), t2_1)
            ).antisymmetrise(0, 1)
            + 2 * (
                # N^5: O^2V^3 / N^4: O^1V^3
                + 0.5 * einsum("ac,ijbc->ijab",
                               einsum("kd,kacd->ac", ur1, hf.ovvv), t2_2)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("bd,ijad->ijab",
                                einsum("kc,kbdc->bd", ur1, t2eri_7), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("la,ijlb->ijab",
                                einsum("kc,kcla->la", ur1, t2sq), hf.ooov)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 1 / 8 * einsum("bd,ijad->ijab",
                                 einsum("kc,kbcd->bd", ur1, t2eri_6), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijka,kb->ijab",
                               einsum("la,ijkl->ijka", ur1, hf.oooo), t1_2)
                # N^6: O^3V^3 / N^4: O^1V^3
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("ijcd,kbcd->ijkb", t2_2, hf.ovvv), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("ijkc,bc->ijkb", hf.ooov, p0_2_vv), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("la,ijlb->ijab",
                                einsum("ka,kl->la", ur1, p0_2_oo), hf.ooov)
                # N^7: O^4V^3 / N^6: O^3V^3
                - 0.25 * einsum("ijla,lb->ijab",
                                einsum("ijkacd,klcd->ijla", t3_2, hf.oovv), ur1)
                # N^6: O^3V^3 / N^4: O^1V^3
                - 1 / 16 * einsum("ijka,kb->ijab",
                                  einsum("ijcd,kacd->ijka", t2_1, t2eri_6), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("ijka,kb->ijab",
                                 einsum("kc,ijac->ijka",
                                        einsum("lmcd,lmkd->kc", t2_1, hf.ooov),
                                        t2_1), ur1)
            ).antisymmetrise(2, 3)
            + (
                # N^6: O^3V^3 / N^6: O^3V^3
                - 0.5 * einsum("kc,ijkabc->ijab",
                               einsum("ld,klcd->kc", ur1, hf.oovv), t3_2)
                # N^6: O^4V^2 / N^4: O^2V^2
                + 1 / 8 * einsum("ijlm,lmab->ijab",
                                 einsum("klmd,ijkd->ijlm",
                                        einsum("kc,lmcd->klmd", ur1, t2_1),
                                        hf.ooov), t2_1)
                # N^6: O^4V^2 / N^4: O^1V^3
                + 1 / 8 * einsum("ijkl,klab->ijab",
                                 einsum("ijde,klde->ijkl", t2_1, t2_1),
                                 einsum("kc,lcab->klab", ur1, hf.ovvv))
            )
            + 4 * (
                # N^6: O^3V^3 / N^4: O^1V^3
                - 0.25 * einsum("ikad,jbkd->ijab",
                                einsum("kc,icad->ikad", ur1, hf.ovvv), t2sq)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum("jklb,ilka->ijab",
                                einsum("kc,jblc->jklb", ur1, t2sq), hf.ooov)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijka,kb->ijab",
                               einsum("ic,jcka->ijka", ur1, hf.ovov), t1_2)
                # N^6: O^3V^3 / N^4: O^1V^3
                + 0.5 * einsum("ikac,jkbc->ijab",
                               einsum("id,kacd->ikac", ur1, hf.ovvv), t2_2)
                # N^6: O^3V^3 / N^4: O^1V^3
                - 0.25 * einsum("ikbd,jkad->ijab",
                                einsum("ic,kbdc->ikbd", ur1, t2eri_7), t2_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("ijad,bd->ijab",
                                einsum("ic,jcad->ijad", ur1, hf.ovvv), p0_2_vv)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijkb,ka->ijab",
                               einsum("jc,ickb->ijkb", t1_2, hf.ovov), ur1)
                # N^6: O^4V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijlb,la->ijab",
                               einsum("jkbc,klic->ijlb", t2_2, hf.ooov), ur1)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("jlbc,lkic->ijkb", t2_1, t2eri_2), ur1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkb,ka->ijab",
                                einsum("jlkb,il->ijkb", hf.ooov, p0_2_oo), ur1)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


#
# 4th order main
#
def block_ph_ph_4(hf, mp, intermediates):
    m11 = intermediates.adc4_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))
    return AdcBlock(apply, diagonal)


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
    p0 = mp.second_order_dm_correction(apply_cvs=hf.has_core_occupied_space)

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
    cvs_p0 = mp.second_order_dm_correction(apply_cvs=True)
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
        # + 0.5 * einsum("ikcd,jlcd,kalb->iajb", mp.t2oo, mp.t2oo, hf.ovov)
        + 0.5 * einsum("ikjl,kalb->iajb",
                       einsum("ikcd,jlcd->ikjl", mp.t2oo, mp.t2oo), hf.ovov)
        - einsum("iljk,kalb->iajb", hf.oooo, t2sq)
        - einsum("idjc,acbd->iajb", t2sq, hf.vvvv)
    )


@register_as_intermediate
def cvs_adc3_m11(hf, mp, intermediates):
    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = cvs_adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()
    cvs_p0 = mp.second_order_dm_correction(apply_cvs=True)

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
            + einsum("JaIc,bc->IaJb", hf.cvcv, cvs_p0.vv)
            - einsum("kIJa,kb->IaJb", hf.occv, 2.0 * cvs_p0.ov)
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


@register_as_intermediate
def adc4_m11(hf, mp, intermediates):
    p0_2 = mp.diffdm(level=2)
    p0_2_oo, p0_2_vv, t1_2 = p0_2.oo, p0_2.vv, p0_2.ov

    t2_1 = mp.t2(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t2_3 = mp.td3(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    p0_3 = mp.diffdm(level=3)
    p0_3_vv, p0_3_ov, p0_3_oo = p0_3.vv, p0_3.ov, p0_3.oo

    # third order ph-ph matrix
    m11 = intermediates.adc3_m11.evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()
    t2eri_1 = mp.t2eri(b.ooov, b.vv).evaluate()
    t2eri_2 = mp.t2eri(b.ooov, b.ov).evaluate()
    t2eri_3 = mp.t2eri(b.oovv, b.oo).evaluate()
    t2eri_4 = mp.t2eri(b.oovv, b.ov).evaluate()
    t2eri_5 = mp.t2eri(b.oovv, b.vv).evaluate()
    t2eri_6 = mp.t2eri(b.ovvv, b.oo).evaluate()
    t2eri_7 = mp.t2eri(b.ovvv, b.ov).evaluate()

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)
    return (
        + m11
        # fourth order terms
        + 2 * (
            ## ab factored
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ij,ab->iajb",
                           einsum("kc,ikjc->ij", t1_2, t2eri_2), d_vv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("ikjc,kc->ij", hf.ooov, p0_3_ov), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ikcd,jkcd->ij", t2_2, t2eri_4), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("ikcd,jkcd->ij", t2_2, t2eri_3), d_vv)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("kc,ikjc->ij", t1_2, t2eri_1), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("ikcd,jkcd->ij", t2_3, hf.oovv), d_vv)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.5 * einsum("ij,ab->iajb",
                           einsum("ikde,jkde->ij",
                                  einsum("ikcd,ce->ikde", t2_1, p0_2_vv),
                                  hf.oovv), d_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ilde,jdle->ij",
                                  einsum("ikce,klcd->ilde", t2_1, t2_2),
                                  hf.ovov), d_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ilde,jdle->ij",
                                  einsum("klce,ikcd->ilde", t2_1, hf.oovv),
                                  t2sq), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ij,ab->iajb",
                            einsum("jc,ic->ij",
                                   einsum("klcd,kljd->jc", t2_1, hf.ooov),
                                   t1_2), d_vv)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("ij,ab->iajb",
                            einsum("jlmd,ilmd->ij",
                                   einsum("kmce,jklcde->jlmd", t2_1, t3_2),
                                   hf.ooov), d_vv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("iklm,jmkl->ij",
                                    einsum("imcd,klcd->iklm", t2_1, t2_2),
                                    hf.oooo), d_vv)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("ilmd,lmjd->ij",
                                    einsum("ikce,klmcde->ilmd", t2_1, t3_2),
                                    hf.ooov), d_vv)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 1 / 8 * einsum("ij,ab->iajb",
                             einsum("idef,jfde->ij",
                                    einsum("klcf,iklcde->idef", t2_1, t3_2),
                                    hf.ovvv), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("ij,ab->iajb",
                              einsum("jk,ik->ij",
                                     einsum("jlcd,klcd->jk", t2_1, hf.oovv),
                                     p0_2_oo), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("ij,ab->iajb",
                              einsum("il,jl->ij",
                                     einsum("klcd,ikcd->il", t2_1, hf.oovv),
                                     p0_2_oo), d_vv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("ilcd,jlcd->ij",
                                   einsum("ikcd,kl->ilcd", t2_1, p0_2_oo),
                                   hf.oovv), d_vv)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("ij,ab->iajb",
                              einsum("jklm,iklm->ij",
                                     einsum("lmef,jkef->jklm", t2_1, hf.oovv),
                                     einsum("ikcd,lmcd->iklm", t2_1, t2_1)), d_vv)
            ## ij factored
            # N^4: O^1V^3 / N^4: O^1V^3
            - 1 * einsum("ab,ij->iajb",
                         einsum("kabc,kc->ab", hf.ovvv, p0_3_ov), d_oo)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("kc,kabc->ab", t1_2, t2eri_7), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("klac,lkcb->ab", t2_2, t2eri_4), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klac,klbc->ab", t2_2, t2eri_5), d_oo)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("kc,kabc->ab", t1_2, t2eri_6), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("klac,klbc->ab", t2_3, hf.oovv), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("lmac,lmbc->ab",
                                  einsum("klac,km->lmac", t2_1, p0_2_oo),
                                  hf.oovv), d_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("lmad,lmbd->ab",
                                  einsum("klac,kcmd->lmad", t2_1, t2sq),
                                  hf.oovv), d_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("lmad,lbmd->ab",
                                  einsum("kmac,klcd->lmad", t2_1, t2_2),
                                  hf.ovov), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ab,ij->iajb",
                            einsum("klad,klbd->ab",
                                   einsum("klac,cd->klad", t2_1, p0_2_vv),
                                   hf.oovv), d_oo)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ab,ij->iajb",
                            einsum("kb,ka->ab",
                                   einsum("klcd,lbcd->kb", t2_1, hf.ovvv),
                                   t1_2), d_oo)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klbd,klad->ab",
                                    einsum("klmcde,mbce->klbd", t3_2, hf.ovvv),
                                    t2_1), d_oo)
            # N^6: O^2V^4 / N^4: V^4
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klbe,klae->ab",
                                    einsum("klcd,becd->klbe", t2_2, hf.vvvv),
                                    t2_1), d_oo)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 1 / 8 * einsum("ab,ij->iajb",
                             einsum("klna,klnb->ab",
                                    einsum("mncd,klmacd->klna", t2_1, t3_2),
                                    hf.ooov), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("ab,ij->iajb",
                              einsum("ad,bd->ab",
                                     einsum("klac,klcd->ad", t2_1, hf.oovv),
                                     p0_2_vv), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("ab,ij->iajb",
                              einsum("bc,ac->ab",
                                     einsum("klcd,klbd->bc", t2_1, hf.oovv),
                                     p0_2_vv), d_oo)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("kade,kebd->ab",
                                   einsum("lmce,klmacd->kade", t2_1, t3_2),
                                   hf.ovvv), d_oo)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("ab,ij->iajb",
                              einsum("mnac,mnbc->ab",
                                     einsum("klmn,klac->mnac",
                                            einsum("klde,mnde->klmn",
                                                   t2_1, t2_1), t2_1),
                                     hf.oovv), d_oo)
            ## none factored
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ka,kjib->iajb", t1_2, t2eri_2)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("ibac,jc->iajb", hf.ovvv, p0_3_ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ikja,kb->iajb", hf.ooov, p0_3_ov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ic,jabc->iajb", t1_2, t2eri_7)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb", t2_2, t2eri_4)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,kjcb->iajb", t2_2, t2eri_4)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.5 * einsum("ibjc,ac->iajb", hf.ovov, p0_3_vv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ic,jacb->iajb", t1_2, t2eri_7)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ka,jkib->iajb", t1_2, t2eri_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ka,jkib->iajb", t1_2, t2eri_2)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ic,jabc->iajb", t1_2, t2eri_6)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ikac,jkbc->iajb", t2_3, hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ibka,jk->iajb", hf.ovov, p0_3_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb", t2_2, t2eri_3)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb", t2_2, t2eri_5)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 * einsum("kmab,ikjm->iajb",
                         einsum("lmac,klbc->kmab", t2_1, t2_2), hf.oooo)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 * einsum("jkad,ibkd->iajb",
                         einsum("jlcd,klac->jkad", t2_1, t2_2), hf.ovov)
            # N^6: O^2V^4 / N^4: V^4
            + 1 * einsum("ijce,aebc->iajb",
                         einsum("jkde,ikcd->ijce", t2_1, t2_2), hf.vvvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikad,jkbd->iajb",
                           einsum("ikac,cd->ikad", t2_1, p0_2_vv), hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("jlcd,kbld->jkbc", hf.oovv, t2sq), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("klbd,jcld->jkbc", hf.oovv, t2sq), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbd,ikad->iajb",
                           einsum("klbc,jcld->jkbd", t2_2, hf.ovov), t2_1)
            # N^6: O^2V^4 / N^4: V^4
            + 0.5 * einsum("abcd,idjc->iajb",
                           einsum("klad,klbc->abcd", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb",
                           einsum("klad,icld->ikac", t2_1, t2sq), hf.oovv)
            # N^8: O^5V^3 / N^6: O^3V^3
            + 0.5 * einsum("ijkmbd,kmad->iajb",
                           einsum("jklbcd,ilmc->ijkmbd", t3_2, hf.ooov), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikbd,jdka->iajb",
                           einsum("klbc,ilcd->ikbd", t2_1, hf.oovv), t2sq)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.5 * einsum("kabc,jkic->iajb",
                           einsum("lmbd,klmacd->kabc", t2_1, t3_2), hf.ooov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkl,kbla->iajb",
                           einsum("ilcd,jkcd->ijkl", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikbd,jdka->iajb",
                           einsum("ilcd,klbc->ikbd", t2_1, hf.oovv), t2sq)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.5 * einsum("ijld,lbad->iajb",
                           einsum("ikce,jklcde->ijld", t2_1, t3_2), hf.ovvv)
            # N^8: O^4V^4 / N^6: O^3V^3
            + 0.5 * einsum("ijlbce,leac->iajb",
                           einsum("ikde,jklbcd->ijlbce", t2_1, t3_2), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 * einsum("jlad,ibld->iajb",
                         einsum("klac,jkcd->jlad", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 3 / 8 * einsum("ijad,bd->iajb",
                             einsum("ikac,jkcd->ijad", t2_1, hf.oovv), p0_2_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ilac,jlbc->iajb",
                           einsum("ikac,kl->ilac", t2_1, p0_2_oo), hf.oovv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ikla,jlkb->iajb",
                           einsum("ilac,kc->ikla", t2_1, t1_2), hf.ooov)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("jkbd,ikad->iajb",
                           einsum("kc,jcbd->jkbd", t1_2, hf.ovvv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("jlbd,ilad->iajb",
                           einsum("jkcd,kblc->jlbd", t2_2, hf.ovov), t2_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("jkbc,ikac->iajb",
                            einsum("klmbcd,lmjd->jkbc", t3_2, hf.ooov), t2_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("iklb,klja->iajb",
                            einsum("imcd,klmbcd->iklb", t2_1, t3_2), hf.ooov)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("ilma,jlmb->iajb",
                            einsum("kmcd,iklacd->ilma", t2_1, t3_2), hf.ooov)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbd,iakd->iajb",
                             einsum("jlbc,klcd->jkbd", t2_1, hf.oovv), t2sq)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ijbd,ad->iajb",
                             einsum("ikcd,jkbc->ijbd", t2_1, hf.oovv), p0_2_vv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ilad,jbld->iajb",
                             einsum("klcd,ikac->ilad", t2_1, hf.oovv), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("il,jbla->iajb",
                              einsum("klcd,ikcd->il", t2_1, hf.oovv), t2sq)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("ad,idjb->iajb",
                              einsum("klcd,klac->ad", t2_1, hf.oovv), t2sq)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("jmbc,imac->iajb",
                            einsum("klbc,jmkl->jmbc", t2_2, hf.oooo), t2_1)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("jkbd,ikad->iajb",
                            einsum("jklcde,lbce->jkbd", t3_2, hf.ovvv), t2_1)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("jace,ibce->iajb",
                            einsum("klad,jklcde->jace", t2_1, t3_2), hf.ovvv)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("abcd,icjd->iajb",
                            einsum("klad,klbc->abcd", t2_1, hf.oovv), t2sq)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("jkbe,ikae->iajb",
                            einsum("jkcd,becd->jkbe", t2_2, hf.vvvv), t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkl,kalb->iajb",
                            einsum("ilcd,jkcd->ijkl", t2_1, hf.oovv), t2sq)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("iace,jebc->iajb",
                            einsum("klde,iklacd->iace", t2_1, t3_2), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 8 * einsum("jlab,il->iajb",
                             einsum("klac,jkbc->jlab", t2_1, hf.oovv), p0_2_oo)
            # N^8: O^4V^4 / N^6: O^3V^3
            + 1 / 8 * einsum("ijklbe,klae->iajb",
                             einsum("jklbcd,iecd->ijklbe", t3_2, hf.ovvv), t2_1)
            # N^8: O^5V^3 / N^6: O^3V^3
            + 1 / 8 * einsum("ijklmb,klma->iajb",
                             einsum("imcd,jklbcd->ijklmb", t2_1, t3_2), hf.ooov)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 1 / 16 * einsum("ad,idjb->iajb",
                              einsum("klac,klcd->ad", t2_1, hf.oovv), t2sq)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("il,jbla->iajb",
                              einsum("ikcd,klcd->il", t2_1, hf.oovv), t2sq)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 3 / 8 * einsum("ilab,jl->iajb",
                             einsum("ikac,klbc->ilab", t2_1, hf.oovv), p0_2_oo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbc,ikac->iajb",
                             einsum("jklm,lmbc->jkbc",
                                    einsum("lmde,jkde->jklm", t2_1, hf.oovv),
                                    t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbc,ikac->iajb",
                             einsum("jklm,lmbc->jkbc",
                                    einsum("jkde,lmde->jklm", t2_1, t2_1),
                                    hf.oovv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("iklm,lmac->ikac",
                                    einsum("ikde,lmde->iklm", t2_1, t2_1),
                                    t2_1), hf.oovv)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("ijde,abde->iajb",
                            einsum("imcd,jmce->ijde", t2_1, hf.oovv),
                            einsum("klae,klbd->abde", t2_1, t2_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ijlm,lmab->iajb",
                            einsum("imcd,jlcd->ijlm", t2_1, t2_1),
                            einsum("klae,kmbe->lmab", t2_1, hf.oovv))
        ).antisymmetrise([(0, 2), (1, 3)])
        + (
            ## ab factored
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("idjc,cd->ij", hf.ovov, p0_3_vv), d_vv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("iljk,kl->ij", hf.oooo, p0_3_oo), d_vv)
            ## ij factored
            # N^4: V^4 / N^4: V^4
            + 1 * einsum("ab,ij->iajb",
                         einsum("adbc,cd->ab", hf.vvvv, p0_3_vv), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ab,ij->iajb",
                         einsum("kbla,kl->ab", hf.ovov, p0_3_oo), d_oo)
        )
    )


@register_as_intermediate
def t2sq(hf, mp, intermediates):
    return einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()
