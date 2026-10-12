## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
from math import sqrt

from adcc import block as b
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate

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
        raise ValueError(
            "Cannot run a general (non-core-valence approximated) "
            "ADC method on top of a ground state with a "
            "core-valence separation."
        )
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError(
            "Cannot run a core-valence approximated ADC method on "
            "top of a ground state without a "
            "core-valence separation."
        )

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError(
            "Could not dispatch: "
            f"spaces={spaces} order={order} variant={variant}. "
            "Probably the secular matrix is not implemented for "
            "the requested method."
        )
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def block_ph_ph_0(hf, mp, intermediates):
    diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal()))

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(ph=(
            + 1.0 * einsum("ib,ab->ia", ur1, hf.fvv)  # N^3: O^1V^2 / N^2: V^2
            - 1.0 * einsum("ja,ij->ia", ur1, hf.foo)  # N^3: O^2V^1 / N^2: O^1V^1
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


def diagonal_pphh_pphh_0(hf):
    res = direct_sum(
        "-i-J+a+b->iJab",
        hf.foo.diagonal(), hf.foo.diagonal(), hf.fvv.diagonal(), hf.fvv.diagonal()
    ).symmetrise(2, 3)  # fmt: skip
    return AmplitudeVector(pphh=res)


def block_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(pphh=(
            2.0 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum("jkab,ik->ijab", ur2, hf.foo)
            ).antisymmetrise(0, 1)
            + 2.0 * (
                # N^5: O^2V^3 / N^4: O^2V^2
                + 1 * einsum("ijac,bc->ijab", ur2, hf.fvv)
            ).antisymmetrise(2, 3)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_ppphhh_ppphhh_0(hf, mp, intermediates):
    pass


#
# 0th order coupling
#
def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ph_ppphhh_0(hf, mp, intermediates):
    pass


def block_ppphhh_ph_0(hf, mp, intermediates):
    pass


def block_pphh_ppphhh_0(hf, mp, intermediates):
    pass


def block_ppphhh_pphh_0(hf, mp, intermediates):
    pass


#
# 1st order main
#
def block_ph_ph_1(hf, mp, intermediates):
    F = hf.operators.fock
    # TODO first-order diagonal, for now zeroth order only
    diagonal = block_ph_ph_0(hf, mp, intermediates).diagonal

    def apply(ampl):
        ur1 = ampl.ph
        # the zeroth-order terms (f_vv, f_oo) cancel against the
        # first-order (F - f) terms, only F remains
        return AmplitudeVector(ph=(
            + 1.0 * einsum("ib,ab->ia", ur1, F.vv)  # N^3: O^1V^2 / N^2: V^2
            - 1.0 * einsum("ja,ji->ia", ur1, F.oo)  # N^3: O^2V^1 / N^2: O^1V^1
            - 1.0 * einsum("jb,ibja->ia", ur1, hf.ovov)  # N^4: O^2V^2 / N^4: O^2V^2
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


def block_pphh_pphh_1(hf, mp, intermediates):
    pass


#
# 1st order coupling
#
def block_ph_pphh_1(hf, mp, intermediates):
    # zeroth order vanishes (f_ov = 0)
    def apply(ampl):
        ur2 = ampl.pphh
        return AmplitudeVector(ph=(
            + 1.0 * einsum("jkab,jkib->ia", ur2, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
            + 1.0 * einsum("ijbc,jabc->ia", ur2, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_1(hf, mp, intermediates):
    pass


def block_pphh_ph_1(hf, mp, intermediates):
    # zeroth order vanishes (f_ov = 0)
    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(pphh=(
            2.0 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ka,ijkb->ijab", ur1, hf.ooov)
            ).antisymmetrise(2, 3)
            + 2.0 * (
                # N^5: O^2V^3 / N^4: O^1V^3
                + 0.5 * einsum("ic,jcab->ijab", ur1, hf.ovvv)
            ).antisymmetrise(0, 1)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ph_ppphhh_1(hf, mp, intermediates):
    pass


def block_ppphhh_ph_1(hf, mp, intermediates):
    pass


def block_pphh_ppphhh_1(hf, mp, intermediates):
    pass


def block_ppphhh_pphh_1(hf, mp, intermediates):
    pass


#
# 2nd order main
#
def block_ph_ph_2(hf, mp, intermediates):
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    # TODO second-order diagonal, for now zeroth order only
    diagonal = block_ph_ph_0(hf, mp, intermediates).diagonal

    # Contractions independent of the trial vector, together with the
    # Fock terms of block_ph_ph_1
    i_vv = (
        + 1.0 * F.vv  # 0 + 1
        - 1.0 * einsum("jc,jabc->ab", t1_1, hf.ovvv)  # N^4: O^1V^3 / N^4: O^1V^3
        - 1.0 * einsum("jc,jbac->ab", t1_1, hf.ovvv)  # N^4: O^1V^3 / N^4: O^1V^3
    ).evaluate()  # fmt: skip
    i_oo = (
        - 1.0 * F.oo  # 0 + 1
        - 1.0 * einsum("kb,ikjb->ij", t1_1, hf.ooov)  # N^4: O^3V^1 / N^4: O^3V^1
        - 1.0 * einsum("kb,jkib->ij", t1_1, hf.ooov)  # N^4: O^3V^1 / N^4: O^3V^1
        - 0.5 * einsum("ib,jb->ij", t1_1, F.ov)  # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("jb,ib->ij", t1_1, F.ov)  # N^3: O^2V^1 / N^2: O^1V^1
        + 0.25 * einsum("ikbc,jkbc->ij", t2_1, hf.oovv)  # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("jkbc,ikbc->ij", t2_1, hf.oovv)  # N^5: O^3V^2 / N^4: O^2V^2
    ).evaluate()  # fmt: skip

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(ph=(
            # 0 + 1 + 2 (via i_vv and i_oo)
            + 1.0 * einsum("ib,ab->ia", ur1, i_vv)  # N^3: O^1V^2 / N^2: V^2
            + 1.0 * einsum("ij,ja->ia", i_oo, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
            # 1
            - 1.0 * einsum("jb,ibja->ia", ur1, hf.ovov)  # N^4: O^2V^2 / N^4: O^2V^2
            # 2
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1.0 * einsum("ik,ka->ia", einsum("jb,jkib->ik", ur1, hf.ooov), t1_1)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1.0 * einsum("jk,ikja->ia", einsum("jb,kb->jk", ur1, t1_1), hf.ooov)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1.0 * einsum("ac,ic->ia", einsum("jb,jabc->ac", ur1, hf.ovvv), t1_1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1.0 * einsum("bc,ibac->ia", einsum("jb,jc->bc", ur1, t1_1), hf.ovvv)
            # N^3: O^2V^1 / N^2: O^1V^1
            - 0.5 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, F.ov), t1_1)
            # N^3: O^2V^1 / N^2: O^1V^1
            - 0.5 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, t1_1), F.ov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,ikac->ia", einsum("jb,jkbc->kc", ur1, hf.oovv), t2_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,ikac->ia", einsum("jb,jkbc->kc", ur1, t2_1), hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkc,jkac->ia", einsum("ib,jkbc->ijkc", ur1, hf.oovv), t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkc,jkac->ia", einsum("ib,jkbc->ijkc", ur1, t2_1), hf.oovv)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


def block_pphh_pphh_2(hf, mp, intermediates):
    pass


#
# 2nd order coupling
#
def block_ph_pphh_2(hf, mp, intermediates):
    pass


def block_pphh_ph_2(hf, mp, intermediates):
    pass


def block_ph_ppphhh_2(hf, mp, intermediates):
    pass


def block_ppphhh_ph_2(hf, mp, intermediates):
    pass


#
# 3rd order main
#
def block_ph_ph_3(hf, mp, intermediates):
    pass


#
# 3rd order coupling
#
def block_ph_pphh_3(hf, mp, intermediates):
    pass


def block_pphh_ph_3(hf, mp, intermediates):
    pass


#
# 4th order main
#
def block_ph_ph_4(hf, mp, intermediates):
    pass
