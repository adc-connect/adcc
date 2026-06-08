#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2022 by the adcc authors
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
from collections import namedtuple

from adcc import block as b
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import einsum

__all__ = ["block"]


#
# Dispatch routine
#

"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ISR matrix.
"""
IsrBlock = namedtuple("IsrBlock", ["apply"])


def block(ground_state, operator, spaces, order, variant=None):
    """
    Gets ground state, one-particle matrix elements associated
    with a one-particle operator, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    The matrix-vector product was derived up to second order
    using the original equations from
    J. Schirmer and A. B. Trofimov, J. Chem. Phys. 120, 11449–11464 (2004).
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []

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
                         "Probably the B-matrix is not implemented for the "
                         "requested method.")
    return globals()[fn](ground_state, operator)


#
# 0th order main
#
def block_ph_ph_0(ground_state, op):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + 1.0 * einsum('ic,ac->ia', ampl.ph, op.vv)
            - 1.0 * einsum('ka,ki->ia', ampl.ph, op.oo)
        ))
    return IsrBlock(apply)


def block_pphh_pphh_0(ground_state, op):
    def apply(ampl):
        return AmplitudeVector(pphh=0.5 * (
            (
                + 2.0 * einsum('ijcb,ac->ijab', ampl.pphh, op.vv)
                - 2.0 * einsum('ijca,bc->ijab', ampl.pphh, op.vv)
            ).antisymmetrise(2, 3)
            + (
                - 2.0 * einsum('kjab,ki->ijab', ampl.pphh, op.oo)
                + 2.0 * einsum('kiab,kj->ijab', ampl.pphh, op.oo)
            ).antisymmetrise(0, 1)
        ))
    return IsrBlock(apply)


#
# 0th order coupling
#
def block_ph_pphh_0(ground_state, op):
    def apply(ampl):
        return AmplitudeVector(ph=0.5 * (
            - 2.0 * einsum('ilad,ld->ia', ampl.pphh, op.ov)
            + 2.0 * einsum('ilca,lc->ia', ampl.pphh, op.ov)
        ))
    return IsrBlock(apply)


def block_pphh_ph_0(ground_state, op):
    def apply(ampl):
        return AmplitudeVector(pphh=0.5 * (
            (
                - 1.0 * einsum('ia,bj->ijab', ampl.ph, op.vo)
                + 1.0 * einsum('ja,bi->ijab', ampl.ph, op.vo)
                + 1.0 * einsum('ib,aj->ijab', ampl.ph, op.vo)
                - 1.0 * einsum('jb,ai->ijab', ampl.ph, op.vo)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))
    return IsrBlock(apply)


#
# 1st order main
#
block_ph_ph_1 = block_ph_ph_0


block_pphh_pphh_1 = block_pphh_pphh_0


#
# 1st order coupling
#
def block_ph_pphh_1(ground_state, op):
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        return AmplitudeVector(ph=0.5 * (
            # zeroth order
            - 2.0 * einsum('ilad,ld->ia', ampl.pphh, op.ov)
            + 2.0 * einsum('ilca,lc->ia', ampl.pphh, op.ov)
            # first order
            + 2.0 * einsum('ilad,lndf,fn->ia', ampl.pphh, t2, op.vo)
            - 2.0 * einsum('ilca,lncf,fn->ia', ampl.pphh, t2, op.vo)
            - 2.0 * einsum('klad,kled,ei->ia', ampl.pphh, t2, op.vo)
            - 2.0 * einsum('ilcd,nlcd,an->ia', ampl.pphh, t2, op.vo)
        ))
    return IsrBlock(apply)


def block_pphh_ph_1(ground_state, op):
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        return AmplitudeVector(pphh=0.5 * (
            (
                # zeroth order
                - 1.0 * einsum('ia,bj->ijab', ampl.ph, op.vo)
                + 1.0 * einsum('ja,bi->ijab', ampl.ph, op.vo)
                + 1.0 * einsum('ib,aj->ijab', ampl.ph, op.vo)
                - 1.0 * einsum('jb,ai->ijab', ampl.ph, op.vo)
                # first order
                + 1.0 * einsum('ia,jnbf,nf->ijab', ampl.ph, t2, op.ov)
                - 1.0 * einsum('ja,inbf,nf->ijab', ampl.ph, t2, op.ov)
                - 1.0 * einsum('ib,jnaf,nf->ijab', ampl.ph, t2, op.ov)
                + 1.0 * einsum('jb,inaf,nf->ijab', ampl.ph, t2, op.ov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + (
                - 1.0 * einsum('ka,ijeb,ke->ijab', ampl.ph, t2, op.ov)
                + 1.0 * einsum('kb,ijea,ke->ijab', ampl.ph, t2, op.ov)
            ).antisymmetrise(2, 3)
            + (
                - 1.0 * einsum('ic,njab,nc->ijab', ampl.ph, t2, op.ov)
                + 1.0 * einsum('jc,niab,nc->ijab', ampl.ph, t2, op.ov)
            ).antisymmetrise(0, 1)
        ))
    return IsrBlock(apply)


#
# 2nd order main
#
def block_ph_ph_2(ground_state, op):
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        return AmplitudeVector(ph=(
            # 0th order
            + 1.0 * einsum('ic,ac->ia', ampl.ph, op.vv)
            - 1.0 * einsum('ka,ki->ia', ampl.ph, op.oo)
            # 2nd order
            # (2,1)
            - 1.0 * einsum('ic,jc,aj->ia', ampl.ph, p0.ov, op.vo)
            - 1.0 * einsum('ka,kb,bi->ia', ampl.ph, p0.ov, op.vo)
            - 1.0 * einsum('ic,ja,jc->ia', ampl.ph, p0.ov, op.ov)  # h.c.
            - 1.0 * einsum('ka,ib,kb->ia', ampl.ph, p0.ov, op.ov)  # h.c.
            # (2,2)
            - 0.25 * einsum('ic,mnef,mnaf,ec->ia', ampl.ph, t2, t2, op.vv)
            - 0.25 * einsum('ic,mnef,mncf,ae->ia', ampl.ph, t2, t2, op.vv)  # h.c.
            # (2,3)
            - 0.5 * einsum('ic,mnce,mnaf,ef->ia', ampl.ph, t2, t2, op.vv)
            + 1.0 * einsum('ic,mncf,jnaf,jm->ia', ampl.ph, t2, t2, op.oo)
            # (2,4)
            + 0.25 * einsum('ka,mnef,inef,km->ia', ampl.ph, t2, t2, op.oo)
            + 0.25 * einsum('ka,mnef,knef,mi->ia', ampl.ph, t2, t2, op.oo)  # h.c.
            # (2,5)
            - 1.0 * einsum('ka,knef,indf,ed->ia', ampl.ph, t2, t2, op.vv)
            + 0.5 * einsum('ka,knef,imef,mn->ia', ampl.ph, t2, t2, op.oo)
            # (2,6)
            + 0.5 * einsum('kc,knef,inaf,ec->ia', ampl.ph, t2, t2, op.vv)
            - 0.5 * einsum('kc,mncf,inaf,km->ia', ampl.ph, t2, t2, op.oo)
            + 0.5 * einsum('kc,inef,kncf,ae->ia', ampl.ph, t2, t2, op.vv)  # h.c.
            - 0.5 * einsum('kc,mnaf,kncf,mi->ia', ampl.ph, t2, t2, op.oo)  # h.c.
            # (2,7)
            - 1.0 * einsum('kc,kncf,imaf,mn->ia', ampl.ph, t2, t2, op.oo)
            + 1.0 * einsum('kc,knce,inaf,ef->ia', ampl.ph, t2, t2, op.vv)
        ))
    return IsrBlock(apply)


#
# 2nd order coupling
#
def block_ph_pphh_2(ground_state, op):

    d_vo, d_ov, d_oo, d_vv = op.vo, op.ov, op.oo, op.vv

    t1_2 = ground_state.diffdm(level=2).ov
    t2_1 = ground_state.t2(b.oovv)
    t2_2 = ground_state.td2(b.oovv)
    t2sq = einsum(
        "ikac,jkbc->iajb",
        ground_state.t2oo,
        ground_state.t2oo).evaluate()

    p0_2 = ground_state.mp2_dm_correction
    p0_2_vv, p0_2_oo = p0_2.vv, p0_2.oo

    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(
            ph=(
                # zeroth order
                -2 * einsum("ijab,jb->ia", ur2, d_ov)  # N^4: O^2V^2 / N^4: O^2V^2
                # first order
                + 1 * einsum(
                    "ijkb,jkab->ia", einsum("jkbc,ci->ijkb", t2_1, d_vo), ur2
                )  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "ik,ak->ia", einsum("ijbc,jkbc->ik", ur2, t2_1), d_vo
                )  # N^5: O^3V^2 / N^4: O^2V^2
                + 2 * einsum(
                    "jb,ijab->ia", einsum("jkbc,ck->jb", t2_1, d_vo), ur2
                )  # N^4: O^2V^2 / N^4: O^2V^2
                # second order
                + 1 * einsum(
                    "jb,ijab->ia", einsum("jc,bc->jb", d_ov, p0_2_vv), ur2
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "ijkb,jkab->ia", einsum("jkbc,ci->ijkb", t2_2, d_vo), ur2
                )  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "ijkb,jkab->ia", einsum("jc,ickb->ijkb", d_ov, t2sq), ur2
                )  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "ik,ak->ia", einsum("ijbc,jkbc->ik", ur2, t2_2), d_vo
                )  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "ijkc,jcka->ia", einsum("ijbc,kb->ijkc", ur2, d_ov), t2sq
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "jb,ijab->ia", einsum("kb,jk->jb", d_ov, p0_2_oo), ur2
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "jb,ijab->ia", einsum("kc,jbkc->jb", d_ov, t2sq), ur2
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 2 * einsum(
                    "jb,ijab->ia", einsum("jc,cb->jb", t1_2, d_vv), ur2
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 2 * einsum(
                    "jb,ijab->ia", einsum("kb,jk->jb", t1_2, d_oo), ur2
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 2 * einsum(
                    "jb,ijab->ia", einsum("jkbc,ck->jb", t2_2, d_vo), ur2
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ld,ilad->ia",
                    einsum(
                        "jklb,jkbd->ld",
                        einsum(
                            "jkbc,lc->jklb", ur2, d_ov), t2_1), t2_1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "ijkb,jkab->ia",
                    einsum(
                        "ic,jkbc->ijkb",
                        einsum(
                            "ilcd,ld->ic", t2_1, d_ov), t2_1), ur2,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "ka,ik->ia",
                    einsum("klad,ld->ka", t2_1, d_ov),
                    einsum("ijbc,jkbc->ik", ur2, t2_1),
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "ld,ilad->ia",
                    einsum("jl,jd->ld", einsum("jkbc,klbc->jl", ur2, t2_1), d_ov),
                    t2_1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum(
                    "ijkl,jkla->ia",
                    einsum("ilcd,jkcd->ijkl", t2_1, t2_1),
                    einsum("jkab,lb->jkla", ur2, d_ov),
                )  # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum(
                    "ikld,klad->ia",
                    einsum(
                        "ijkl,jd->ikld",
                        einsum(
                            "ijbc,klbc->ijkl", ur2, t2_1), d_ov), t2_1,
                )  # N^6: O^4V^2 / N^4: O^2V^2
            )
        )

    return IsrBlock(apply)


def block_pphh_ph_2(ground_state, op):
    d_vo, d_ov, d_oo, d_vv = op.vo, op.ov, op.oo, op.vv

    t1_2 = ground_state.diffdm(level=2).ov
    t2_1 = ground_state.t2(b.oovv)
    t2_2 = ground_state.td2(b.oovv)

    t2sq = einsum(
        "ikac,jkbc->iajb",
        ground_state.t2oo,
        ground_state.t2oo).evaluate()

    p0_2 = ground_state.mp2_dm_correction
    p0_2_vv, p0_2_oo = p0_2.vv, p0_2.oo

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(
            pphh=0.5 * (
                # zeroth order
                4 * einsum(
                    "ja,bi->ijab", ur1, d_vo
                ).antisymmetrise(0, 1).antisymmetrise(2, 3)
                # N^4: O^2V^2 / N^4: O^2V^2
                + 4 * einsum(
                    "jb,ia->ijab", einsum("jkbc,kc->jb", t2_1, d_ov), ur1
                ).antisymmetrise(0, 1).antisymmetrise(2, 3)
                # N^4: O^2V^2 / N^4: O^2V^2
                + 2 * einsum(
                    "ijkb,ka->ijab", einsum("ijbc,kc->ijkb", t2_1, d_ov), ur1
                ).antisymmetrise(2, 3)  # N^5: O^3V^2 / N^4: O^2V^2
                + 2 * einsum(
                    "ik,jkab->ijab", einsum("ic,kc->ik", ur1, d_ov), t2_1
                ).antisymmetrise(0, 1)  # N^5: O^3V^2 / N^4: O^2V^2
                # second order
                + 4 * (
                    +1 * einsum(
                        "jb,ia->ijab", einsum("kb,kj->jb", t1_2, d_oo), ur1
                    )  # N^4: O^2V^2 / N^4: O^2V^2
                    + 1 * einsum(
                        "jb,ia->ijab", einsum("jkbc,kc->jb", t2_2, d_ov), ur1
                    )  # N^4: O^2V^2 / N^4: O^2V^2
                    + 1 * einsum(
                        "ib,ja->ijab", einsum("ic,bc->ib", t1_2, d_vv), ur1
                    )  # N^4: O^2V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "jb,ia->ijab", einsum("cj,bc->jb", d_vo, p0_2_vv), ur1
                    )  # N^4: O^2V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "ib,ja->ijab", einsum("bk,ik->ib", d_vo, p0_2_oo), ur1
                    )  # N^4: O^2V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "ib,ja->ijab", einsum("ck,ibkc->ib", d_vo, t2sq), ur1
                    )  # N^4: O^2V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "ijkb,ka->ijab", einsum("ci,jbkc->ijkb", d_vo, t2sq), ur1
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "ijkb,ak->ijab", einsum("ic,jbkc->ijkb", ur1, t2sq), d_vo
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                ).antisymmetrise(0, 1).antisymmetrise(2, 3)
                + 2 * (
                    +1 * einsum(
                        "ijkb,ka->ijab", einsum("ijbc,kc->ijkb", t2_2, d_ov), ur1
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "ijka,kb->ijab",
                        einsum(
                            "kc,ijac->ijka", einsum("klcd,dl->kc", t2_1, d_vo), t2_1
                        ), ur1,
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "ijla,bl->ijab",
                        einsum(
                            "ld,ijad->ijla",
                            einsum(
                                "kc,klcd->ld", ur1, t2_1), t2_1), d_vo,
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                    - 0.25
                    * einsum(
                        "ijla,bl->ijab",
                        einsum(
                            "ijkl,ka->ijla", einsum("ijcd,klcd->ijkl",
                                                    t2_1, t2_1), ur1
                        ), d_vo,
                    )  # N^6: O^4V^2 / N^4: O^2V^2
                ).antisymmetrise(2, 3)
                + 2 * (
                    +1 * einsum(
                        "ik,jkab->ijab", einsum("ic,kc->ik", ur1, d_ov), t2_2
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "jk,ikab->ijab",
                        einsum("kc,jc->jk", einsum("klcd,dl->kc", t2_1, d_vo), ur1),
                        t2_1,
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                    + 0.5 * einsum(
                        "jl,ilab->ijab",
                        einsum("ld,dj->jl", einsum("kc,klcd->ld", ur1, t2_1), d_vo),
                        t2_1,
                    )  # N^5: O^3V^2 / N^4: O^2V^2
                    - 0.25 * einsum(
                        "ijkl,klab->ijab",
                        einsum(
                            "ikld,dj->ijkl", einsum("ic,klcd->ikld",
                                                    ur1, t2_1), d_vo
                        ), t2_1,
                    )  # N^6: O^4V^2 / N^4: O^2V^2
                ).antisymmetrise(0, 1)
            )
        )

    return IsrBlock(apply)


#
# 3rd order main
#
def block_ph_ph_3(ground_state, op):

    d_vo, d_ov, d_oo, d_vv = op.vo, op.ov, op.oo, op.vv

    t1_2 = ground_state.diffdm(level=2).ov
    t2_1 = ground_state.t2(b.oovv)
    t2_2 = ground_state.td2(b.oovv)
    t3_2 = ground_state.tt2(b.ooovvv)
    t2sq = einsum(
        "ikac,jkbc->iajb",
        ground_state.t2oo,
        ground_state.t2oo).evaluate()

    p0_2 = ground_state.mp2_dm_correction
    p0_2_vv, p0_2_oo = p0_2.vv, p0_2.oo

    p0_3 = ground_state.mp3_dm_correction
    p0_3_vv, p0_3_ov, p0_3_oo = p0_3.vv, p0_3.ov, p0_3.oo

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(
            ph=(
                # zeroth order
                +1 * einsum("ib,ab->ia", ur1, d_vv)  # N^3: O^1V^2 / N^2: V^2
                - 1 * einsum("ja,ji->ia", ur1, d_oo)  # N^3: O^2V^1 / N^2: O^1V^1
                # second order
                + 1 * einsum(
                    "ab,ib->ia", einsum("kj,jbka->ab", d_oo, t2sq), ur1
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ic,ac->ia", einsum("jb,icjb->ic", ur1, t2sq), d_vv
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "jc,iajc->ia", einsum("jb,cb->jc", ur1, d_vv), t2sq
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ij,ja->ia", einsum("ib,jb->ij", t1_2, d_ov), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 1 * einsum(
                    "ij,ja->ia", einsum("jb,bi->ij", t1_2, d_vo), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 1 * einsum(
                    "ij,ja->ia", einsum("bc,icjb->ij", d_vv, t2sq), ur1
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ij,ja->ia", einsum("ib,jb->ij", ur1, d_ov), t1_2
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 1 * einsum(
                    "ij,aj->ia", einsum("ib,jb->ij", ur1, t1_2), d_vo
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 0.5 * einsum(
                    "ij,ja->ia", einsum("jk,ik->ij", d_oo, p0_2_oo), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 0.5 * einsum(
                    "ij,ja->ia", einsum("ki,jk->ij", d_oo, p0_2_oo), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 0.5 * einsum(
                    "ic,ac->ia", einsum("ib,bc->ic", ur1, p0_2_vv), d_vv
                )  # N^3: O^1V^2 / N^2: V^2
                - 0.5 * einsum(
                    "ic,ac->ia", einsum("ib,cb->ic", ur1, d_vv), p0_2_vv
                )  # N^3: O^1V^2 / N^2: V^2
                - 0.5 * einsum(
                    "kb,iakb->ia", einsum("jb,jk->kb", ur1, d_oo), t2sq
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "ka,ki->ia", einsum("jb,jbka->ka", ur1, t2sq), d_oo
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "kd,ikad->ia",
                    einsum("kc,cd->kd", einsum("jb,jkbc->kc", ur1, t2_1), d_vv),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ij,ja->ia",
                    einsum(
                        "ikbc,jkbc->ij",
                        einsum(
                            "ilbc,lk->ikbc", t2_1, d_oo), t2_1), ur1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "lc,ilac->ia",
                    einsum("kc,lk->lc", einsum("jb,jkbc->kc", ur1, t2_1), d_oo),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "ijkd,jkad->ia",
                    einsum(
                        "ijkc,cd->ijkd",
                        einsum(
                            "ib,jkbc->ijkc", ur1, t2_1), d_vv), t2_1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                # third order
                - 1 * einsum(
                    "ij,ja->ia", einsum("bi,jb->ij", d_vo, p0_3_ov), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 1 * einsum(
                    "ij,ja->ia", einsum("jb,ib->ij", d_ov, p0_3_ov), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 1 * einsum(
                    "ij,aj->ia", einsum("ib,jb->ij", ur1, p0_3_ov), d_vo
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 1 * einsum(
                    "ij,ja->ia", einsum("ib,jb->ij", ur1, d_ov), p0_3_ov
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 0.5 * einsum(
                    "ik,ka->ia", einsum("ji,jk->ik", d_oo, p0_3_oo), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 0.5 * einsum(
                    "ik,ka->ia", einsum("kj,ij->ik", d_oo, p0_3_oo), ur1
                )  # N^3: O^2V^1 / N^2: O^1V^1
                - 0.5 * einsum(
                    "ib,ab->ia", einsum("ic,bc->ib", ur1, p0_3_vv), d_vv
                )  # N^3: O^1V^2 / N^2: V^2
                - 0.5 * einsum(
                    "ib,ab->ia", einsum("ic,bc->ib", ur1, d_vv), p0_3_vv
                )  # N^3: O^1V^2 / N^2: V^2
                + 1 * einsum(
                    "kc,ikac->ia",
                    einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, t1_2), d_ov),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "kc,ikac->ia",
                    einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, d_ov), t1_2),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "kc,ikac->ia",
                    einsum("kd,dc->kc", einsum("jb,jkbd->kd", ur1, t2_2), d_vv),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "ik,ka->ia",
                    einsum("kc,ci->ik", einsum("jb,jkbc->kc", ur1, t2_1), d_vo),
                    t1_2,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "ik,ak->ia",
                    einsum("kc,ic->ik", einsum("jb,jkbc->kc", ur1, t2_1), t1_2),
                    d_vo,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum(
                    "kd,ikad->ia",
                    einsum("kc,cd->kd", einsum("jb,jkbc->kc", ur1, t2_1), d_vv),
                    t2_2,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ij,ja->ia",
                    einsum(
                        "ilbc,jlbc->ij",
                        einsum(
                            "ikbc,kl->ilbc", t2_1, d_oo), t2_2), ur1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ij,ja->ia",
                    einsum(
                        "jkbc,ikbc->ij", einsum("jklbcd,dl->jkbc", t3_2, d_vo), t2_1
                    ), ur1,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                + 0.5 * einsum(
                    "ij,ja->ia",
                    einsum(
                        "jlbc,ilbc->ij",
                        einsum(
                            "jkbc,lk->jlbc", t2_1, d_oo), t2_2), ur1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ij,ja->ia",
                    einsum(
                        "ikbc,jkbc->ij", einsum("iklbcd,ld->ikbc", t3_2, d_ov), t2_1
                    ), ur1,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                + 0.5 * einsum(
                    "ijkc,jkac->ia",
                    einsum(
                        "jkbc,ib->ijkc",
                        einsum(
                            "jklbcd,dl->jkbc", t3_2, d_vo), ur1), t2_1,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                + 0.5 * einsum(
                    "jkac,ijkc->ia",
                    einsum(
                        "jklacd,ld->jkac", t3_2, d_ov),
                    einsum(
                        "ib,jkbc->ijkc", ur1, t2_1),
                )  # N^6: O^3V^3 / N^6: O^3V^3
                + 0.5 * einsum(
                    "kc,ikac->ia",
                    einsum("lb,klbc->kc", einsum("jb,jl->lb", ur1, d_oo), t2_2),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "kc,ikac->ia",
                    einsum("lb,klbc->kc", einsum("jb,jl->lb", ur1, d_oo), t2_1),
                    t2_2,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ic,ac->ia",
                    einsum("kd,ikcd->ic", einsum("jb,jkbd->kd", ur1, t2_1), t2_2),
                    d_vv,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum(
                    "ic,ac->ia",
                    einsum("kd,ikcd->ic", einsum("jb,jkbd->kd", ur1, t2_2), t2_1),
                    d_vv,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ij,ja->ia",
                    einsum("ic,jc->ij", einsum("ikbc,kb->ic", t2_1, d_ov), t1_2),
                    ur1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ij,ja->ia",
                    einsum(
                        "ikbd,jkbd->ij",
                        einsum(
                            "ikbc,dc->ikbd", t2_1, d_vv), t2_2), ur1,
                )  # N^5: O^2V^3 / N^4: O^2V^2
                - 1 * einsum(
                    "ij,ja->ia",
                    einsum("jc,ic->ij", einsum("jkbc,bk->jc", t2_1, d_vo), t1_2),
                    ur1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ij,ja->ia",
                    einsum(
                        "jkbd,ikbd->ij",
                        einsum(
                            "jkbc,cd->jkbd", t2_1, d_vv), t2_2), ur1,
                )  # N^5: O^2V^3 / N^4: O^2V^2
                - 1 * einsum(
                    "ka,ik->ia",
                    einsum("jkac,jc->ka", t2_1, d_ov),
                    einsum("ib,kb->ik", ur1, t1_2),
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ijkc,jkac->ia",
                    einsum(
                        "iklc,jl->ijkc",
                        einsum(
                            "ib,klbc->iklc", ur1, t2_2), d_oo), t2_1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ik,ka->ia",
                    einsum("kb,ib->ik", einsum("jkbc,cj->kb", t2_1, d_vo), ur1),
                    t1_2,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "iklc,klac->ia",
                    einsum(
                        "ijkc,lj->iklc",
                        einsum(
                            "ib,jkbc->ijkc", ur1, t2_1), d_oo), t2_2,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "kc,ikac->ia",
                    einsum("lc,kl->kc", einsum("jb,jlbc->lc", ur1, t2_2), d_oo),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "kc,ikac->ia",
                    einsum(
                        "klcd,dl->kc",
                        einsum(
                            "jb,jklbcd->klcd", ur1, t3_2), d_vo), t2_1,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                - 1
                * einsum(
                    "lc,ilac->ia",
                    einsum("kc,lk->lc", einsum("jb,jkbc->kc", ur1, t2_1), d_oo),
                    t2_2,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 1 * einsum(
                    "ilad,ld->ia",
                    einsum(
                        "kc,iklacd->ilad",
                        einsum(
                            "jb,jkbc->kc", ur1, t2_1), t3_2), d_ov,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                - 0.5 * einsum(
                    "ijkc,jkac->ia",
                    einsum(
                        "ijkd,dc->ijkc",
                        einsum(
                            "ib,jkbd->ijkd", ur1, t2_2), d_vv), t2_1,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "ijkd,jkad->ia",
                    einsum(
                        "ijkc,cd->ijkd",
                        einsum(
                            "ib,jkbc->ijkc", ur1, t2_1), d_vv), t2_2,
                )  # N^5: O^3V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "kc,ikac->ia",
                    einsum("jd,jkcd->kc", einsum("jb,db->jd", ur1, d_vv), t2_2),
                    t2_1,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "ka,ki->ia",
                    einsum("lc,klac->ka", einsum("jb,jlbc->lc", ur1, t2_2), t2_1),
                    d_oo,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "iklc,klac->ia",
                    einsum(
                        "klcd,di->iklc",
                        einsum(
                            "jb,jklbcd->klcd", ur1, t3_2), d_vo), t2_1,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                - 0.5
                * einsum(
                    "ka,ki->ia",
                    einsum("lc,klac->ka", einsum("jb,jlbc->lc", ur1, t2_1), t2_2),
                    d_oo,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "klcd,iklacd->ia",
                    einsum(
                        "jklc,jd->klcd",
                        einsum(
                            "jb,klbc->jklc", ur1, t2_1), d_ov), t3_2,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                - 0.5 * einsum(
                    "il,al->ia",
                    einsum(
                        "klcd,ikcd->il",
                        einsum(
                            "jb,jklbcd->klcd", ur1, t3_2), t2_1), d_vo,
                )  # N^6: O^3V^3 / N^6: O^3V^3
                - 0.5 * einsum(
                    "kc,ikac->ia",
                    einsum("jd,jkcd->kc", einsum("jb,db->jd", ur1, d_vv), t2_1),
                    t2_2,
                )  # N^4: O^2V^2 / N^4: O^2V^2
                - 0.5 * einsum(
                    "klcd,iklacd->ia",
                    einsum("jl,jkcd->klcd", einsum("jb,lb->jl", ur1, d_ov), t2_1),
                    t3_2,
                )  # N^6: O^3V^3 / N^6: O^3V^3
            )
        )

    return IsrBlock(apply)
