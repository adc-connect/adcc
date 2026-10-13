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

from adcc import block as b
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import direct_sum, einsum
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
        "-i-j+a+b->ijab",
        hf.foo.diagonal(), hf.foo.diagonal(), hf.fvv.diagonal(), hf.fvv.diagonal()
    ).symmetrise(2, 3)  # fmt: skip
    if not hf.has_core_occupied_space:
        res = res.symmetrise(0, 1)
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
    assert not hf.has_core_occupied_space
    F = hf.operators.fock
    # Zeroth order:
    # {f^{a}_{a}} - {f^{i}_{i}}
    # First order:
    # - {F^{i}_{i}} - {V^{ia}_{ia}} + {F^{a}_{a}} - {f^{a}_{a}} + {f^{i}_{i}}
    # -> f_aa and f_ii terms cancel
    # -> identical to the normal diagonal (exactly like the apply)
    diagonal = AmplitudeVector(ph=(
        direct_sum("a-i->ia", F.vv.diagonal(), F.oo.diagonal())
        - einsum("iaia->ia", hf.ovov)
    ))  # fmt: skip

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


def diagonal_pphh_pphh_1(hf):
    assert not hf.has_core_occupied_space
    F = hf.operators.fock
    # the zeroth order contributions cancel exactly against the first order contributions
    # -> only F orbital energies remain
    # -> basically identical to the diagonal of standard ADC
    ov_itmd = (
        direct_sum("a-i->ia", F.vv.diagonal(), F.oo.diagonal())
        - 4 * 0.5 * einsum("iaia->ia", hf.ovov)
    ).evaluate()  # fmt: skip
    return AmplitudeVector(pphh=(
        direct_sum("ia+jb->ijab", ov_itmd, ov_itmd)
        + direct_sum(
            "ij+ab->ijab",
            einsum("ijij->ij", hf.oooo),
            einsum("abab->ab", hf.vvvv),
        )
    ).symmetrise(0, 1).symmetrise(2, 3))  # fmt: skip


def block_pphh_pphh_1(hf, mp, intermediates):
    F = hf.operators.fock
    diagonal = diagonal_pphh_pphh_1(hf)

    def apply(ampl):
        ur2 = ampl.pphh
        # the zeroth-order terms (f_oo, f_vv) cancel against the
        # first-order (F - f) terms, only F remains
        return AmplitudeVector(pphh=(
            2.0 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum("jkab,ik->ijab", ur2, F.oo)
            ).antisymmetrise(0, 1)
            + 2.0 * (
                # N^5: O^2V^3 / N^4: O^2V^2
                + 1 * einsum("ijac,bc->ijab", ur2, F.vv)
            ).antisymmetrise(2, 3)
            + 4.0 * (
                # N^6: O^3V^3 / N^4: O^2V^2
                + 1 * einsum("jkac,ickb->ijab", ur2, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("klab,ijkl->ijab", ur2, hf.oooo)
            # N^6: O^2V^4 / N^4: V^4
            + 0.5 * einsum("ijcd,abcd->ijab", ur2, hf.vvvv)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


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
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    i1 = intermediates.rohf_dods_adc2_i1
    i2 = intermediates.rohf_dods_adc2_i2

    diagonal = AmplitudeVector(ph=(
        direct_sum("a+i->ia", i1.diagonal(), i2.diagonal())
        # first order contribution missing in i1 and i2
        - einsum("iaia->ia", hf.ovov)
        - 1 * einsum("ijab,ijab->ia", t2_1, hf.oovv)  # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("ja,ijia->ia", t1_1, hf.ooov)  # N^3: O^2V^1 / N^4: O^3V^1
        + 2 * einsum("ib,iaab->ia", t1_1, hf.ovvv)  # N^3: O^1V^2 / N^4: O^1V^3
    ))  # fmt: skip

    def apply(ampl):
        ur1 = ampl.ph
        return AmplitudeVector(ph=(
            # 0 + 1 + 2 (via i1 and i2)
            + 1.0 * einsum("ib,ab->ia", ur1, i1)  # N^3: O^1V^2 / N^2: V^2
            + 1.0 * einsum("ij,ja->ia", i2, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
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
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,ikac->ia", einsum("jb,jkbc->kc", ur1, hf.oovv), t2_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,ikac->ia", einsum("jb,jkbc->kc", ur1, t2_1), hf.oovv)
        ))  # fmt: skip

    return AdcBlock(apply, diagonal)


def block_pphh_pphh_2(hf, mp, intermediates):
    pass


#
# 2nd order coupling
#
def block_ph_pphh_2(hf, mp, intermediates):
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    m12 = intermediates.rohf_dods_adc3_m12
    lower = block_ph_pphh_1(hf, mp, intermediates)

    def apply(ampl):
        ur2 = ampl.pphh
        return lower.apply(ampl) + AmplitudeVector(ph=(
            # 2 (via m12)
            + 1 * einsum("ijkb,jkab->ia", m12, ur2)
            # 2
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ik,ka->ia", einsum("ijbc,jkbc->ik", ur2, t2_1), F.ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("kl,ilka->ia", einsum("jkbc,jlbc->kl", ur2, t2_1), hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("cd,icad->ia", einsum("jkbc,jkbd->cd", ur2, t2_1), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("jkla,iljk->ia", einsum("jkab,lb->jkla", ur2, t1_1), hf.oooo)
            # N^5: O^1V^4 / N^4: V^4
            - 1 * einsum("ibcd,adbc->ia", einsum("ijbc,jd->ibcd", ur2, t1_1), hf.vvvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 2 * einsum("ijkb,jakb->ia", einsum("ijbc,kc->ijkb", ur2, t1_1), hf.ovov)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 2 * einsum("ikcd,kcad->ia", einsum("ijbc,jkbd->ikcd", ur2, t2_1), hf.ovvv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkl,klja->ia", einsum("ijbc,klbc->ijkl", ur2, t2_1), hf.ooov)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_pphh_ph_2(hf, mp, intermediates):
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    m21 = intermediates.rohf_dods_adc3_m21
    lower = block_pphh_ph_1(hf, mp, intermediates)

    def apply(ampl):
        ur1 = ampl.ph
        return lower.apply(ampl) + AmplitudeVector(pphh=(
            # 2 (via m21)
            2.0 * (
                + 1 * einsum("ijkb,ka->ijab", m21, ur1)
            ).antisymmetrise(2, 3)
            # 2
            + 4.0 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijka,kb->ijab", einsum("ic,jcka->ijka", ur1, hf.ovov), t1_1)
                # N^6: O^3V^3 / N^4: O^1V^3
                + 0.5 * einsum("ikbd,jkad->ijab", einsum("ic,kbcd->ikbd", ur1, hf.ovvv), t2_1)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 2.0 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ijlb,la->ijab", einsum("kb,ijkl->ijlb", ur1, hf.oooo), t1_1)
                # N^5: O^2V^3 / N^4: O^1V^3
                + 0.5 * einsum("bd,ijad->ijab", einsum("kc,kbcd->bd", ur1, hf.ovvv), t2_1)
            ).antisymmetrise(2, 3)
            + 2.0 * (
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ik,jkab->ijab", einsum("ic,kc->ik", ur1, F.ov), t2_1)
                # N^5: O^1V^4 / N^4: V^4
                + 0.5 * einsum("jabd,id->ijab", einsum("jc,abcd->jabd", ur1, hf.vvvv), t1_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("jl,ilab->ijab", einsum("kc,kljc->jl", ur1, hf.ooov), t2_1)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 0.25 * einsum("ijkl,klab->ijab", einsum("ic,kljc->ijkl", ur1, hf.ooov), t2_1)
            ).antisymmetrise(0, 1)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


def block_ph_ppphhh_2(hf, mp, intermediates):
    pass


def block_ppphhh_ph_2(hf, mp, intermediates):
    pass


#
# 3rd order main
#
def block_ph_ph_3(hf, mp, intermediates):
    m11 = intermediates.rohf_dods_adc3_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))

    return AdcBlock(apply, diagonal)


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


#
# Intermediates
#
@register_as_intermediate
def rohf_dods_adc2_i1(hf, mp, intermediates):
    # terms with delta_ij from the adc2 m11 block including contributions through second-order
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.td1(b.oovv)
    return (
        + 1.0 * F.vv
        # terms with (1 + P_ab)
        + 2 * (
            - 1 * einsum("ic,iabc->ab", t1_1, hf.ovvv)  # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("ia,ib->ab", t1_1, F.ov)  # N^3: O^1V^2 / N^2: V^2
            + 0.25 * einsum("ijac,ijbc->ab", t2_1, hf.oovv)  # N^5: O^2V^3 / N^4: O^2V^2
        )
    ).symmetrise()  # fmt: skip


@register_as_intermediate
def rohf_dods_adc2_i2(hf, mp, intermediates):
    # terms with delta_ab from the adc2 m11 block including contributions through second-order
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    return (
        - 1.0 * F.oo
        # terms with (1 + P_ij)
        + 2 * (
            - 1 * einsum("ka,ikja->ij", t1_1, hf.ooov)  # N^4: O^3V^1 / N^4: O^3V^1
            - 0.5 * einsum("ia,ja->ij", t1_1, F.ov)  # N^3: O^2V^1 / N^2: O^1V^1
            + 0.25 * einsum("ikab,jkab->ij", t2_1, hf.oovv)  # N^5: O^3V^2 / N^4: O^2V^2
        )
    ).symmetrise()  # fmt: skip


@register_as_intermediate
def rohf_dods_adc3_m11(hf, mp, intermediates):
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.td1(b.oovv)
    t1_2 = mp.ts2(b.ov)
    t2_2 = mp.td2(b.oovv)
    # Build two Kronecker deltas
    d_oo = hf.foo.zeros_like()
    d_vv = hf.fvv.zeros_like()
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    return (
        # terms with delta_ij: contributions through second order in adc2_i1
        einsum("ij,ab->iajb", d_oo, rohf_dods_adc2_i1(hf, mp, intermediates) + (
            # Apply (1 + P_ab) to:
            2 * (
                - 1 * einsum("ic,iabc->ab", t1_2, hf.ovvv)  # N^4: O^1V^3 / N^4: O^1V^3
                - 0.5 * einsum("ia,ib->ab", t1_2, F.ov)  # N^3: O^1V^2 / N^2: V^2
                + 0.25 * einsum("ijac,ijbc->ab", t2_2, hf.oovv)  # N^5: O^2V^3 / N^4: O^2V^2
                # N^4: O^1V^3 / N^4: O^1V^3
                + 1 * einsum("ic,iabc->ab", einsum("jd,ijcd->ic", t1_1, t2_1), hf.ovvv)
                # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum("ib,ia->ab", einsum("jc,icjb->ib", t1_1, hf.ovov), t1_1)
                # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum("ia,ib->ab", einsum("jc,ijac->ia", t1_1, t2_1), F.ov)
                # N^5: O^2V^3 / N^4: O^1V^3
                + 0.5 * einsum("ijbc,ijac->ab", einsum("jd,idbc->ijbc", t1_1, hf.ovvv), t2_1)
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("ikad,idkb->ab", einsum("ijac,jkcd->ikad", t2_1, t2_1), hf.ovov)
                # N^5: O^2V^3 / N^4: O^1V^3
                - 0.25 * einsum("ib,ia->ab", einsum("ijcd,jbcd->ib", t2_1, hf.ovvv), t1_1)
                # N^6: O^2V^4 / N^4: V^4
                - 1 / 8 * einsum("ijbc,ijac->ab", einsum("ijde,bcde->ijbc", t2_1, hf.vvvv), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.25 * einsum("ijka,ijkb->ab", einsum("kc,ijac->ijka", t1_1, t2_1), hf.ooov)
            )
            # Apply 1 to:
            # N^4: V^4 / N^4: V^4
            + 1 * einsum("cd,adbc->ab", einsum("ic,id->cd", t1_1, t1_1), hf.vvvv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,iajb->ab", einsum("ic,jc->ij", t1_1, t1_1), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("jk,jakb->ab", einsum("ijcd,ikcd->jk", t2_1, t2_1), hf.ovov)
            # N^5: O^2V^3 / N^4: V^4
            - 0.5 * einsum("ce,aebc->ab", einsum("ijcd,ijde->ce", t2_1, t2_1), hf.vvvv)
        ).symmetrise())
        # terms with delta_ab: contributions through second order in adc2_i2
        + einsum("ab,ij->iajb", d_vv, rohf_dods_adc2_i2(hf, mp, intermediates) + (
            # Apply (1 + P_ij) to:
            2 * (
                - 1 * einsum("ka,ikja->ij", t1_2, hf.ooov)  # N^4: O^3V^1 / N^4: O^3V^1
                - 0.5 * einsum("ia,ja->ij", t1_2, F.ov)  # N^3: O^2V^1 / N^2: O^1V^1
                + 0.25 * einsum("ikab,jkab->ij", t2_2, hf.oovv)  # N^5: O^3V^2 / N^4: O^2V^2
                # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum("ka,ikja->ij", einsum("lb,klab->ka", t1_1, t2_1), hf.ooov)
                # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum("ja,ia->ij", einsum("kb,jbka->ja", t1_1, hf.ovov), t1_1)
                # N^4: O^2V^2 / N^4: O^2V^2
                + 0.5 * einsum("ia,ja->ij", einsum("kb,ikab->ia", t1_1, t2_1), F.ov)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("ikla,jkla->ij", einsum("lb,ikab->ikla", t1_1, t2_1), hf.ooov)
                # N^6: O^3V^3 / N^4: O^2V^2
                + 0.5 * einsum("ilac,jcla->ij", einsum("ikab,klbc->ilac", t2_1, t2_1), hf.ovov)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.25 * einsum("ja,ia->ij", einsum("klab,kljb->ja", t2_1, hf.ooov), t1_1)
                # N^6: O^4V^2 / N^4: O^2V^2
                - 1 / 8 * einsum("iklm,jklm->ij", einsum("ikab,lmab->iklm", t2_1, t2_1), hf.oooo)
                # N^5: O^2V^3 / N^4: O^1V^3
                + 0.25 * einsum("jkab,ikab->ij", einsum("kc,jcab->jkab", t1_1, hf.ovvv), t2_1)
            )
            # Apply 1 to:
            # N^4: O^4 / N^4: O^4
            + 1 * einsum("kl,iljk->ij", einsum("ka,la->kl", t1_1, t1_1), hf.oooo)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ab,iajb->ij", einsum("ka,kb->ab", t1_1, t1_1), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("km,imjk->ij", einsum("klab,lmab->km", t2_1, t2_1), hf.oooo)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.5 * einsum("bc,ibjc->ij", einsum("klab,klac->bc", t2_1, t2_1), hf.ovov)
        ).symmetrise())
        # terms without deltas
        # Apply (1 + P_ijP_ab) to:
        + 2 * (
            # second order contributions:
            + 1 * einsum("ka,jkib->iajb", t1_1, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ic,jabc->iajb", t1_1, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ikac,jkbc->iajb", t2_1, hf.oovv)  # N^6: O^3V^3 / N^4: O^2V^2
            # third order contributions
            + 1 * einsum("ka,jkib->iajb", t1_2, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ic,jabc->iajb", t1_2, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ikac,jkbc->iajb", t2_2, hf.oovv)  # N^6: O^3V^3 / N^4: O^2V^2
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb", einsum("kc,ibjc->ijkb", t1_1, hf.ovov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb", einsum("jkbc,ic->ijkb", t2_1, F.ov), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijlb,la->iajb", einsum("jkbc,iklc->ijlb", t2_1, hf.ooov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jakb->iajb", einsum("ic,kc->ik", t1_1, t1_1), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb", einsum("ic,jkbc->ijkb", t1_1, t2_1), F.ov)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 0.5 * einsum("ikac,jkbc->iajb", einsum("id,kdac->ikac", t1_1, hf.ovvv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb", einsum("klbd,jdlc->jkbc", t2_1, hf.ovov), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb", einsum("jlcd,kdlb->jkbc", t2_1, hf.ovov), t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,ka->iajb", einsum("jc,ibkc->ijkb", t1_1, hf.ovov), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,ka->iajb", einsum("klbc,iljc->ijkb", t2_1, hf.ooov), t1_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 1 * einsum("ijkd,kbad->iajb", einsum("ic,jkcd->ijkd", t1_1, t2_1), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ka,jkib->iajb", einsum("lc,klac->ka", t1_1, t2_1), hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ic,jabc->iajb", einsum("kd,ikcd->ic", t1_1, t2_1), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 * einsum("jlad,ibld->iajb", einsum("klac,jkcd->jlad", t2_1, t2_1), hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("jb,ia->iajb", einsum("kc,jkbc->jb", t1_1, hf.oovv), t1_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijkb,ka->iajb", einsum("jkcd,ibcd->ijkb", t2_1, hf.ovvv), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("iklb,klja->iajb", einsum("ic,klbc->iklb", t1_1, t2_1), hf.ooov)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ikla,jklb->iajb", einsum("lc,ikac->ikla", t1_1, t2_1), hf.ooov)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("jkbc,ikac->iajb", einsum("kd,jdbc->jkbc", t1_1, hf.ovvv), t2_1)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ad,ibjd->iajb", einsum("klac,klcd->ad", t2_1, t2_1), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("il,jalb->iajb", einsum("ikcd,klcd->il", t2_1, t2_1), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("jkbc,ikac->iajb", einsum("lmbc,jklm->jkbc", t2_1, hf.oooo), t2_1)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("jkbc,ikac->iajb", einsum("jkde,bcde->jkbc", t2_1, hf.vvvv), t2_1)
        )
        # Apply 1 to:
        # first order contributions
        - 1 * einsum("ibja->iajb", hf.ovov)
        # (there are not second order contributions with "Apply 1 to")
        # third order contributions:
        # N^6: O^2V^4 / N^4: V^4
        + 0.5 * einsum("abcd,icjd->iajb", einsum("klac,klbd->abcd", t2_1, t2_1), hf.ovov)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijkl,kalb->iajb", einsum("ikcd,jlcd->ijkl", t2_1, t2_1), hf.ovov)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijla,lb->iajb", einsum("ka,iljk->ijla", t1_1, hf.oooo), t1_1)
        # N^5: O^1V^4 / N^4: V^4
        - 1 * einsum("iabd,jd->iajb", einsum("ic,adbc->iabd", t1_1, hf.vvvv), t1_1)
        # N^6: O^3V^3 / N^4: O^2V^2
        - 1 * einsum("lmab,imjl->iajb", einsum("klac,kmbc->lmab", t2_1, t2_1), hf.oooo)
        # N^6: O^2V^4 / N^4: V^4
        - 1 * einsum("ijde,aebd->iajb", einsum("ikcd,jkce->ijde", t2_1, t2_1), hf.vvvv)
    ).symmetrise((0, 2), (1, 3))  # fmt: skip


@register_as_intermediate
def rohf_dods_adc3_m12(hf, mp, intermediates):
    # Second-order part of block_ph_pphh_2 that is independent of the
    # trial vector, contracted as einsum("ijkb,jkab->ia", m12, ur2)
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    return (
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jkbc,ic->ijkb", t2_1, hf.operators.fock.ov)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("kc,ibjc->ijkb", t1_1, hf.ovov)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 2 * einsum("jlbc,ilkc->ijkb", t2_1, hf.ooov)
        # N^6: O^3V^3 / N^4: O^1V^3
        - 0.5 * einsum("jkcd,ibcd->ijkb", t2_1, hf.ovvv)
    )  # fmt: skip


@register_as_intermediate
def rohf_dods_adc3_m21(hf, mp, intermediates):
    # Second-order part of block_pphh_ph_2 that is independent of the
    # trial vector, contracted as 2 * einsum("ijkb,ka->ijab", m21, ur1)
    # antisymmetrised in (a, b). The (1 - P_ij) of the first two terms
    # is applied here.
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    return (
        2.0 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("jc,ickb->ijkb", t1_1, hf.ovov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ilbc,kljc->ijkb", t2_1, hf.ooov)
        ).antisymmetrise(0, 1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijbc,kc->ijkb", t2_1, hf.operators.fock.ov)
        # N^6: O^3V^3 / N^4: O^1V^3
        - 0.25 * einsum("ijcd,kbcd->ijkb", t2_1, hf.ovvv)
    )  # fmt: skip
