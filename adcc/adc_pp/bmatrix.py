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
from adcc.functions import einsum
from adcc.AmplitudeVector import AmplitudeVector


__all__ = ["block"]


#
# Dispatch routine
#

"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix.
"""
AdcBlock = namedtuple("AdcBlock", ["apply"])


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
    return AdcBlock(apply)


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
    return AdcBlock(apply)


#
# 0th order coupling
#
def block_ph_pphh_0(ground_state, op):
    def apply(ampl):
        return AmplitudeVector(ph=0.5 * (
            - 2.0 * einsum('ilad,ld->ia', ampl.pphh, op.ov)
            + 2.0 * einsum('ilca,lc->ia', ampl.pphh, op.ov)
        ))
    return AdcBlock(apply)


def block_pphh_ph_0(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose()
    else:
        op_vo = op.vo

    def apply(ampl):
        return AmplitudeVector(pphh=0.5 * (
            (
                - 1.0 * einsum('ia,bj->ijab', ampl.ph, op_vo)
                + 1.0 * einsum('ja,bi->ijab', ampl.ph, op_vo)
                + 1.0 * einsum('ib,aj->ijab', ampl.ph, op_vo)
                - 1.0 * einsum('jb,ai->ijab', ampl.ph, op_vo)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply)


#
# 1st order main
#
block_ph_ph_1 = block_ph_ph_0


#
# 1st order coupling
#
def block_ph_pphh_1(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose()
    else:
        op_vo = op.vo
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        return AmplitudeVector(ph=0.5 * (
            # zeroth order
            - 2.0 * einsum('ilad,ld->ia', ampl.pphh, op.ov)
            + 2.0 * einsum('ilca,lc->ia', ampl.pphh, op.ov)
            # first order
            + 2.0 * einsum('ilad,lndf,fn->ia', ampl.pphh, t2, op_vo)
            - 2.0 * einsum('ilca,lncf,fn->ia', ampl.pphh, t2, op_vo)
            - 2.0 * einsum('klad,kled,ei->ia', ampl.pphh, t2, op_vo)
            - 2.0 * einsum('ilcd,nlcd,an->ia', ampl.pphh, t2, op_vo)
        ))
    return AdcBlock(apply)


def block_pphh_ph_1(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose()
    else:
        op_vo = op.vo
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        return AmplitudeVector(pphh=0.5 * (
            (
                # zeroth order
                - 1.0 * einsum('ia,bj->ijab', ampl.ph, op_vo)
                + 1.0 * einsum('ja,bi->ijab', ampl.ph, op_vo)
                + 1.0 * einsum('ib,aj->ijab', ampl.ph, op_vo)
                - 1.0 * einsum('jb,ai->ijab', ampl.ph, op_vo)
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
    return AdcBlock(apply)


#
# 2nd order main
#
def block_ph_ph_2(ground_state, op):
    if op.is_symmetric:
        op_vo = op.ov.transpose()
    else:
        op_vo = op.vo
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)

    def apply(ampl):
        return AmplitudeVector(ph=(
            # 0th order
            + 1.0 * einsum('ic,ac->ia', ampl.ph, op.vv)
            - 1.0 * einsum('ka,ki->ia', ampl.ph, op.oo)
            # 2nd order
            # (2,1)
            - 1.0 * einsum('ic,jc,aj->ia', ampl.ph, p0.ov, op_vo)
            - 1.0 * einsum('ka,kb,bi->ia', ampl.ph, p0.ov, op_vo)
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
    return AdcBlock(apply)
