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
    return AdcBlock(lambda ampl: 0, 0)


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
    return AdcBlock(lambda ampl: 0, 0)


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
    t2_1 = mp.td1(b.oovv)

    def apply(ampl):
        ur3 = ampl.ppphhh
        return AmplitudeVector(ph=(
            # Apply 1 to:
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 * einsum("kacd,ickd->ia", einsum("jklabc,jlbd->kacd", ur3, t2_1), hf.ovov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 * einsum("ijld,jald->ia", einsum("ijkbcd,klbc->ijld", ur3, t2_1), hf.ovov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 3 / 2 * einsum("jlma,imjl->ia", einsum("jklabc,kmbc->jlma", ur3, t2_1), hf.oooo)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 3 / 2 * einsum("ibde,aebd->ia", einsum("ijkbcd,jkce->ibde", ur3, t2_1), hf.vvvv)
        ))  # fmt: skip

    return AdcBlock(apply, 0)


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
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.td1(b.oovv)
    t1_2 = mp.ts2(b.ov)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    ph_pphh_2 = block_ph_pphh_2(hf, mp, intermediates).apply

    def apply(ampl):
        ur2 = ampl.pphh
        # The scaling comment is given as: [comp_scaling] / [mem_scaling]
        res = ph_pphh_2(ampl)
        res.ph += (
            # third-order contributions: part with no target indices on ur2
            # Apply 1 to:
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("kl,ikla->ia", einsum("jlbc,jkbc->kl", ur2, t2_2), hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("cd,idac->ia", einsum("jkbd,jkbc->cd", ur2, t2_2), hf.ovvv)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.5 * einsum("kd,ikad->ia", einsum("jlbc,jklbcd->kd", ur2, t3_2), hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("il,la->ia",
                         einsum("jm,imjl->il",
                                einsum("jkbc,kmbc->jm", ur2, t2_1), hf.oooo), t1_1)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 1 * einsum("il,la->ia",
                         einsum("bd,ibld->il",
                                einsum("jkbc,jkcd->bd", ur2, t2_1), hf.ovov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ad,id->ia",
                         einsum("jl,jald->ad",
                                einsum("jkbc,klbc->jl", ur2, t2_1), hf.ovov), t1_1)
            # N^5: O^2V^3 / N^4: V^4
            + 1 * einsum("ad,id->ia",
                         einsum("be,aebd->ad",
                                einsum("jkbc,jkce->be", ur2, t2_1), hf.vvvv), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ld,ilad->ia",
                           einsum("mb,lmbd->ld",
                                  einsum("jkbc,jkmc->mb", ur2, hf.ooov), t2_1), t2_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ld,ilad->ia",
                           einsum("je,jlde->ld",
                                  einsum("jkbc,kebc->je", ur2, hf.ovvv), t2_1), t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ilmd,lmad->ia",
                         einsum("ijlb,jmbd->ilmd",
                                einsum("jkbc,klic->ijlb", ur2, hf.ooov), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 1 * einsum("ijce,jace->ia",
                         einsum("jlcd,ilde->ijce",
                                einsum("jkbc,klbd->jlcd", ur2, t2_1), t2_1), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 2 * einsum("il,la->ia",
                         einsum("klcd,ickd->il",
                                einsum("jkbc,jlbd->klcd", ur2, t2_1), hf.ovov), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * einsum("kl,ilka->ia",
                         einsum("kc,lc->kl",
                                einsum("jkbc,jb->kc", ur2, t1_1), t1_1), hf.ooov)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 2 * einsum("cd,icad->ia",
                         einsum("kc,kd->cd",
                                einsum("jkbc,jb->kc", ur2, t1_1), t1_1), hf.ovvv)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * einsum("ld,ilad->ia",
                         einsum("jc,jlcd->ld",
                                einsum("jkbc,kb->jc", ur2, t1_1), t2_1), hf.oovv)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 2 * einsum("iklc,kalc->ia",
                         einsum("ijlb,jkbc->iklc",
                                einsum("id,jlbd->ijlb", t1_1, t2_1), ur2), hf.ovov)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("il,la->ia",
                           einsum("jklm,imjk->il",
                                  einsum("jkbc,lmbc->jklm", ur2, t2_1), hf.oooo), t1_1)
            # N^6: O^3V^3 / N^4: V^4
            - 0.5 * einsum("ibce,aebc->ia",
                           einsum("ijke,jkbc->ibce",
                                  einsum("id,jkde->ijke", t1_1, t2_1), ur2), hf.vvvv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ld,ilad->ia", einsum("jkbc,jkbc->", ur2, t2_1) * t1_1, hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ilmd,lmad->ia",
                           einsum("kl,kmid->ilmd",
                                  einsum("jkbc,jlbc->kl", ur2, t2_1), hf.ooov), t2_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ilce,lace->ia",
                           einsum("cd,ilde->ilce",
                                  einsum("jkbc,jkbd->cd", ur2, t2_1), t2_1), hf.ovvv)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ilmd,lmad->ia",
                            einsum("cd,lmic->ilmd",
                                   einsum("jkbc,jkbd->cd", ur2, t2_1), hf.ooov), t2_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ikde,kade->ia",
                            einsum("kl,ilde->ikde",
                                   einsum("jkbc,jlbc->kl", ur2, t2_1), t2_1), hf.ovvv)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ilmd,lmad->ia",
                             einsum("jklm,jkid->ilmd",
                                    einsum("jkbc,lmbc->jklm", ur2, t2_1), hf.ooov), t2_1)
            # N^6: O^4V^2 / N^4: O^1V^3
            + 1 / 8 * einsum("ilbc,labc->ia",
                             einsum("ijkl,jkbc->ilbc",
                                    einsum("ilde,jkde->ijkl", t2_1, t2_1), ur2), hf.ovvv)
            # third-order contributions: part with 1 occ target index on ur2
            # Apply 1 to:
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ik,ka->ia", einsum("ijbc,jkbc->ik", ur2, t2_2), F.ov)
            # N^5: O^1V^4 / N^4: V^4
            - 1 * einsum("ibcd,abcd->ia", einsum("ijcd,jb->ibcd", ur2, t1_2), hf.vvvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkc,jcka->ia", einsum("ikbc,jb->ijkc", ur2, t1_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 2 * einsum("ikcd,kdac->ia", einsum("ijbd,jkbc->ikcd", ur2, t2_2), hf.ovvv)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkl,jkla->ia", einsum("ilbc,jkbc->ijkl", ur2, t2_2), hf.ooov)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.5 * einsum("iklb,klab->ia", einsum("ijcd,jklbcd->iklb", ur2, t3_2), hf.oovv)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("ijcd,jacd->ia",
                         einsum("ijkc,kd->ijcd", einsum("ijbc,kb->ijkc", ur2, t1_1), t1_1), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ikld,klad->ia",
                         einsum("ib,klbd->ikld",
                                einsum("ijbc,jc->ib", ur2, t1_1), t2_1), hf.oovv)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 * einsum("ijld,jald->ia",
                         einsum("ijkl,kd->ijld",
                                einsum("ijbc,klbc->ijkl", ur2, t2_1), t1_1), hf.ovov)
            # N^5: O^1V^4 / N^4: V^4
            + 1 * einsum("ibcd,adbc->ia",
                         einsum("jd,ijbc->ibcd",
                                einsum("ke,jkde->jd", t1_1, t2_1), ur2), hf.vvvv)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ik,ka->ia",
                           einsum("id,kd->ik",
                                  einsum("ijbc,jdbc->id", ur2, hf.ovvv), t1_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ka,ik->ia",
                         einsum("ld,kdla->ka", t1_1, hf.ovov),
                         einsum("ijbc,jkbc->ik", ur2, t2_1))
            # N^6: O^3V^3 / N^4: O^1V^3
            - 1 * einsum("ilce,lace->ia",
                         einsum("ikcd,klde->ilce",
                                einsum("ijbc,jkbd->ikcd", ur2, t2_1), t2_1), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkb,jakb->ia",
                         einsum("kc,ijbc->ijkb",
                                einsum("ld,klcd->kc", t1_1, t2_1), ur2), hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * einsum("ik,ka->ia",
                         einsum("ic,kc->ik", einsum("ijbc,jb->ic", ur2, t1_1), t1_1), F.ov)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 2 * einsum("ac,ic->ia",
                         einsum("kd,kcad->ac", t1_1, hf.ovvv), einsum("ijbc,jb->ic", ur2, t1_1))
            # N^6: O^4V^2 / N^4: O^2V^2
            + 2 * einsum("iklc,kalc->ia",
                         einsum("jklb,ijbc->iklc",
                                einsum("kd,jlbd->jklb", t1_1, t2_1), ur2), hf.ovov)
            # N^6: O^3V^3 / N^4: V^4
            + 2 * einsum("ibde,aebd->ia",
                         einsum("ikbe,kd->ibde",
                                einsum("ijbc,jkce->ikbe", ur2, t2_1), t1_1), hf.vvvv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ikbc,kabc->ia",
                           einsum("jk,ijbc->ikbc", einsum("jd,kd->jk", t1_1, t1_1), ur2), hf.ovvv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijce,jace->ia",
                           einsum("be,ijbc->ijce",
                                  einsum("klbd,klde->be", t2_1, t2_1), ur2), hf.ovvv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ikld,klad->ia",
                            einsum("ie,klde->ikld",
                                   einsum("ijbc,jebc->ie", ur2, hf.ovvv), t2_1), t2_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ikbc,kabc->ia",
                            einsum("jk,ijbc->ikbc",
                                   einsum("jlde,klde->jk", t2_1, t2_1), ur2), hf.ovvv)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("ka,ik->ia",
                            einsum("klde,lade->ka", t2_1, hf.ovvv),
                            einsum("ijbc,jkbc->ik", ur2, t2_1))
            # N^6: O^4V^2 / N^4: O^1V^3
            + 1 / 8 * einsum("ijde,jade->ia",
                             einsum("ijkl,klde->ijde",
                                    einsum("ijbc,klbc->ijkl", ur2, t2_1), t2_1), hf.ovvv)
            # third-order contributions: part with 1 virt target index on ur2
            # Apply 1 to:
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijkb,jkab->ia", einsum("jkbc,ic->ijkb", t2_2, F.ov), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("jkla,ijkl->ia", einsum("klab,jb->jkla", ur2, t1_2), hf.oooo)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkc,jkac->ia", einsum("jb,ickb->ijkc", t1_2, hf.ovov), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 2 * einsum("ijlb,jlab->ia", einsum("jkbc,iklc->ijlb", t2_2, hf.ooov), ur2)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.5 * einsum("ijlc,jlac->ia", einsum("jklbcd,ikbd->ijlc", t3_2, hf.oovv), ur2)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijkd,jkad->ia", einsum("jkbc,idbc->ijkd", t2_2, hf.ovvv), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ij,ja->ia",
                         einsum("jlcd,ilcd->ij", t2_1, hf.oovv),
                         einsum("jkab,kb->ja", ur2, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijkb,jkab->ia",
                         einsum("jl,klib->ijkb", einsum("jc,lc->jl", t1_1, t1_1), hf.ooov), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 * einsum("ijkb,jkab->ia",
                         einsum("jkld,ibld->ijkb",
                                einsum("lc,jkcd->jkld", t1_1, t2_1), hf.ovov), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jkla,iljk->ia",
                         einsum("lb,jkab->jkla", einsum("mc,lmbc->lb", t1_1, t2_1), ur2), hf.oooo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("il,la->ia",
                           einsum("ic,lc->il", t1_1, t1_1), einsum("jkab,jklb->la", ur2, hf.ooov))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,jkab->ia",
                         einsum("ic,jkbc->ijkb", einsum("ld,idlc->ic", t1_1, hf.ovov), t2_1), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,jkab->ia",
                         einsum("iklc,jlbc->ijkb",
                                einsum("lmcd,kmid->iklc", t2_1, hf.ooov), t2_1), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkb,jkab->ia",
                         einsum("kc,ibjc->ijkb", einsum("ld,klcd->kc", t1_1, t2_1), hf.ovov), ur2)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * einsum("ik,ka->ia",
                         einsum("kc,ic->ik", t1_1, F.ov), einsum("jkab,jb->ka", ur2, t1_1))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * einsum("ik,ka->ia",
                         einsum("lc,ilkc->ik", t1_1, hf.ooov), einsum("jkab,jb->ka", ur2, t1_1))
            # N^6: O^5V^1 / N^4: O^2V^2
            + 2 * einsum("ijkb,jkab->ia",
                         einsum("klmb,imjl->ijkb",
                                einsum("lc,kmbc->klmb", t1_1, t2_1), hf.oooo), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 2 * einsum("ijkb,jkab->ia",
                         einsum("ikld,jlbd->ijkb",
                                einsum("lc,ickd->ikld", t1_1, hf.ovov), t2_1), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkl,jkla->ia",
                           einsum("lc,jkic->ijkl", t1_1, hf.ooov),
                           einsum("jkab,lb->jkla", ur2, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkb,jkab->ia",
                           einsum("jm,kmib->ijkb",
                                  einsum("jlcd,lmcd->jm", t2_1, t2_1), hf.ooov), ur2)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ijkb,jkab->ia",
                            einsum("bc,jkic->ijkb",
                                   einsum("lmbd,lmcd->bc", t2_1, t2_1), hf.ooov), ur2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("im,ma->ia",
                            einsum("ilcd,lmcd->im", t2_1, t2_1),
                            einsum("jkab,jkmb->ma", ur2, hf.ooov))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkb,jkab->ia",
                            einsum("ic,jkbc->ijkb",
                                   einsum("lmcd,lmid->ic", t2_1, hf.ooov), t2_1), ur2)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ijkb,jkab->ia",
                             einsum("jklm,lmib->ijkb",
                                    einsum("jkcd,lmcd->jklm", t2_1, t2_1), hf.ooov), ur2)
        )  # fmt: skip
        return res

    return AdcBlock(apply, 0)


def block_pphh_ph_3(hf, mp, intermediates):
    pass


#
# 4th order main
#
def block_ph_ph_4(hf, mp, intermediates):
    m11 = intermediates.rohf_dods_adc4_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))

    return AdcBlock(apply, diagonal)


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


@register_as_intermediate
def rohf_dods_adc4_m11(hf, mp, intermediates):
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.td1(b.oovv)
    t1_2 = mp.ts2(b.ov)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)
    t1_3 = mp.ts3(b.ov)
    t2_3 = mp.td3(b.oovv)
    # Build two Kronecker deltas
    d_oo = hf.foo.zeros_like()
    d_vv = hf.fvv.zeros_like()
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)
    # The scaling comment is given as: [comp_scaling] / [mem_scaling]
    return rohf_dods_adc3_m11(hf, mp, intermediates) + (
        # fourth order contributions with delta_ij
        einsum("ij,ab->iajb", d_oo, 2 * (
            # Apply (1 + P_ab) to:
            - 1 * einsum("ic,iabc->ab", t1_3, hf.ovvv)  # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("ia,ib->ab", t1_3, F.ov)  # N^3: O^1V^2 / N^2: V^2
            + 0.25 * einsum("ijac,ijbc->ab", t2_3, hf.oovv)  # N^5: O^2V^3 / N^4: O^2V^2
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ic,iabc->ab", einsum("jd,jd->", t1_1, t1_1) * t1_1, hf.ovvv)
            # N^4: V^4 / N^4: V^4
            + 1 * einsum("cd,acbd->ab", einsum("id,ic->cd", t1_1, t1_2), hf.vvvv)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("jd,jabd->ab", einsum("ijcd,ic->jd", t2_1, t1_2), hf.ovvv)
            # N^3: O^1V^2 / N^2: V^2
            + 0.5 * einsum("ia,ib->ab", einsum("jc,jc->", t1_1, t1_1) * t1_1, F.ov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ja->ab", einsum("ic,ibjc->jb", t1_2, hf.ovov), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ib,ia->ab", einsum("jc,icjb->ib", t1_1, hf.ovov), t1_2)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikad,idkb->ab", einsum("jkcd,ijac->ikad", t2_1, t2_2), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,ibka->ab", einsum("jkcd,ijcd->ik", t2_1, t2_2), hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,iajb->ab", einsum("jc,ic->ij", t1_1, t1_2), hf.ovov)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 1 * einsum("id,iabd->ab", einsum("jc,ijcd->id", t1_1, t2_2), hf.ovvv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ja,jb->ab", einsum("ic,ijac->ja", t1_1, t2_2), F.ov)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijbc,ijac->ab", einsum("id,jdbc->ijbc", t1_1, hf.ovvv), t2_2)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ja,jb->ab", einsum("ijac,ic->ja", t2_1, t1_2), F.ov)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("jkad,jbkd->ab", einsum("ikac,ijcd->jkad", t2_1, t2_2), hf.ovov)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijbd,ijad->ab", einsum("ic,jcbd->ijbd", t1_2, hf.ovvv), t2_1)
            # N^5: O^2V^3 / N^4: V^4
            - 0.5 * einsum("ce,acbe->ab", einsum("ijde,ijcd->ce", t2_1, t2_2), hf.vvvv)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("kd,kd->", t1_1, t1_1) * einsum("ijac,ijbc->ab", t2_1, hf.oovv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ib,ia->ab", einsum("ijcd,jbcd->ib", t2_2, hf.ovvv), t1_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ib,ia->ab", einsum("ijcd,jbcd->ib", t2_1, hf.ovvv), t1_2)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 1 / 8 * einsum("ijbd,ijad->ab", einsum("ijkcde,kbce->ijbd", t3_2, hf.ovvv), t2_1)
            # N^6: O^2V^4 / N^4: V^4
            - 1 / 8 * einsum("ijbe,ijae->ab", einsum("ijcd,becd->ijbe", t2_2, hf.vvvv), t2_1)
            # N^6: O^2V^4 / N^4: V^4
            - 1 / 8 * einsum("ijbc,ijac->ab", einsum("ijde,bcde->ijbc", t2_1, hf.vvvv), t2_2)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 0.25 * einsum("ic,iabc->ab", einsum("jkde,jkde->", t2_1, t2_1) * t1_1, hf.ovvv)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.25 * einsum("jkad,jkbd->ab", einsum("ic,ijkacd->jkad", t1_1, t3_2), hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijka,ijkb->ab", einsum("kc,ijac->ijka", t1_1, t2_2), hf.ooov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijka,jkib->ab", einsum("jkac,ic->ijka", t2_1, t1_2), hf.ooov)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("kade,kebd->ab", einsum("ijce,ijkacd->kade", t2_1, t3_2), hf.ovvv)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.25 * einsum("ic,iabc->ab", einsum("jkde,ijkcde->ic", t2_1, t3_2), hf.ovvv)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ia,ib->ab", einsum("jkcd,jkcd->", t2_1, t2_1) * t1_1, F.ov)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 1 / 8 * einsum("ia,ib->ab", einsum("jkcd,ijkacd->ia", t2_1, t3_2), F.ov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 1 / 8 * einsum("ikla,iklb->ab", einsum("jlcd,ijkacd->ikla", t2_1, t3_2), hf.ooov)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 0.5 * einsum("ib,ia->ab",
                           einsum("bd,id->ib",
                                  einsum("jc,jbcd->bd", t1_1, hf.ovvv), t1_1), t1_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ijac,ijbc->ab",
                           einsum("jkae,ikce->ijac",
                                  einsum("jlad,klde->jkae", t2_1, t2_1), t2_1), hf.oovv)
            # N^3: O^1V^2 / N^2: V^2
            - 3 / 8 * einsum("ia,ib->ab",
                             einsum("ij,ja->ia", einsum("ic,jc->ij", t1_1, t1_1), t1_1), F.ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 3 / 16 * einsum("kb,ka->ab",
                              einsum("ijkc,ijbc->kb",
                                     einsum("kd,ijcd->ijkc", t1_1, t2_1), hf.oovv), t1_1)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("ib,ia->ab",
                           einsum("bd,id->ib", einsum("jc,jdbc->bd", t1_1, hf.ovvv), t1_1), t1_1)
            # N^3: O^1V^2 / N^2: V^2
            - 1 / 8 * einsum("ia,ib->ab",
                             einsum("ij,ja->ia", einsum("ic,jc->ij", t1_1, F.ov), t1_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("kb,ka->ab",
                              einsum("ijkc,ijbc->kb",
                                     einsum("kd,ijcd->ijkc", t1_1, hf.oovv), t2_1), t1_1)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 1 / 32 * einsum("bd,ad->ab",
                              einsum("ijbc,ijcd->bd", t2_1, t2_1),
                              einsum("klae,klde->ad", t2_1, hf.oovv))
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("ae,be->ab",
                            einsum("ijac,ijce->ae", t2_1, t2_1),
                            einsum("kd,kebd->be", t1_1, hf.ovvv))
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("ad,bd->ab",
                            einsum("ijac,ijcd->ad", t2_1, t2_1),
                            einsum("ke,kbde->bd", t1_1, hf.ovvv))
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("ijac,ijbc->ab",
                            einsum("ik,jkac->ijac",
                                   einsum("ilde,klde->ik", t2_1, t2_1), t2_1), hf.oovv)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 1 / 8 * einsum("ijac,ijbc->ab",
                             einsum("ce,ijae->ijac",
                                    einsum("klcd,klde->ce", t2_1, t2_1), t2_1), hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("kb,ka->ab",
                              einsum("ijkc,ijbc->kb",
                                     einsum("ijcd,kd->ijkc", t2_1, F.ov), t2_1), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("ijac,ijbc->ab",
                              einsum("ijkl,klac->ijac",
                                     einsum("ijde,klde->ijkl", t2_1, t2_1), t2_1), hf.oovv)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 1 / 32 * einsum("bd,ad->ab",
                              einsum("ijcd,ijbc->bd", t2_1, hf.oovv),
                              einsum("klae,klde->ad", t2_1, t2_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 3 / 16 * einsum("ka,kb->ab",
                              einsum("ijkc,ijac->ka",
                                     einsum("kd,ijcd->ijkc", t1_1, t2_1), t2_1), F.ov)
        ).symmetrise())
        # fourth order contributions with delta_ab
        + 1 * einsum("ab,ij->iajb", d_vv, 2 * (
            # Apply (1 + P_ij) to:
            - 1 * einsum("ka,ikja->ij", t1_3, hf.ooov)  # N^4: O^3V^1 / N^4: O^3V^1
            - 0.5 * einsum("ia,ja->ij", t1_3, F.ov)  # N^3: O^2V^1 / N^2: O^1V^1
            + 0.25 * einsum("ikab,jkab->ij", t2_3, hf.oovv)  # N^5: O^3V^2 / N^4: O^2V^2
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("lb,lb->", t1_1, t1_1) * einsum("ka,ikja->ij", t1_1, hf.ooov)
            # N^4: O^4 / N^4: O^4
            + 1 * einsum("kl,ikjl->ij", einsum("la,ka->kl", t1_1, t1_2), hf.oooo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("lb,iljb->ij", einsum("klab,ka->lb", t2_1, t1_2), hf.ooov)
            # N^3: O^2V^1 / N^2: O^1V^1
            + 0.5 * einsum("kb,kb->", t1_1, t1_1) * einsum("ia,ja->ij", t1_1, F.ov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ib->ij", einsum("ka,jakb->jb", t1_2, hf.ovov), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ja,ia->ij", einsum("kb,jbka->ja", t1_1, hf.ovov), t1_2)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikbc,jbkc->ij", einsum("ilac,klab->ikbc", t2_1, t2_2), hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("kb,ikjb->ij", einsum("la,klab->kb", t1_1, t2_2), hf.ooov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ab,iajb->ij", einsum("kb,ka->ab", t1_1, t1_2), hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ib,jb->ij", einsum("ka,ikab->ib", t1_1, t2_2), F.ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("iklb,jklb->ij", einsum("la,ikab->iklb", t1_1, t2_2), hf.ooov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ib,jb->ij", einsum("ikab,ka->ib", t2_1, t1_2), F.ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("iklb,jlkb->ij", einsum("ilab,ka->iklb", t2_1, t1_2), hf.ooov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("km,ikjm->ij", einsum("lmab,klab->km", t2_1, t2_2), hf.oooo)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("ilbc,jclb->ij", einsum("klac,ikab->ilbc", t2_1, t2_2), hf.ovov)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.5 * einsum("bc,ibjc->ij", einsum("klac,klab->bc", t2_1, t2_2), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("lc,lc->", t1_1, t1_1) * einsum("ikab,jkab->ij", t2_1, hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ja,ia->ij", einsum("klab,kljb->ja", t2_2, hf.ooov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ja,ia->ij", einsum("klab,kljb->ja", t2_1, hf.ooov), t1_2)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("iklm,jmkl->ij", einsum("imab,klab->iklm", t2_1, t2_2), hf.oooo)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("iklm,jklm->ij", einsum("lmab,ikab->iklm", t2_1, t2_2), hf.oooo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.25 * einsum("lmbc,lmbc->", t2_1, t2_1) * einsum("ka,ikja->ij", t1_1, hf.ooov)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.25 * einsum("ilbc,jlbc->ij", einsum("ka,iklabc->ilbc", t1_1, t3_2), hf.oovv)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("jkab,ikab->ij", einsum("kc,jcab->jkab", t1_1, hf.ovvv), t2_2)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.25 * einsum("ilmc,jlmc->ij", einsum("kmab,iklabc->ilmc", t2_1, t3_2), hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("jkbc,ikbc->ij", einsum("ka,jabc->jkbc", t1_2, hf.ovvv), t2_1)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.25 * einsum("ka,ikja->ij", einsum("lmbc,klmabc->ka", t2_1, t3_2), hf.ooov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("klbc,klbc->", t2_1, t2_1) * einsum("ia,ja->ij", t1_1, F.ov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 1 / 8 * einsum("ilma,lmja->ij", einsum("ikbc,klmabc->ilma", t2_1, t3_2), hf.ooov)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 1 / 8 * einsum("ia,ja->ij", einsum("klbc,iklabc->ia", t2_1, t3_2), F.ov)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 1 / 8 * einsum("iacd,jdac->ij", einsum("klbd,iklabc->iacd", t2_1, t3_2), hf.ovvv)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 0.5 * einsum("jl,il->ij",
                           einsum("kb,kljb->jl", t1_1, hf.ooov),
                           einsum("ia,la->il", t1_1, t1_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jlbd,ilbd->ij",
                           einsum("jmbc,lmcd->jlbd",
                                  einsum("jkab,kmac->jmbc", t2_1, t2_1), t2_1), hf.oovv)
            # N^3: O^2V^1 / N^2: O^1V^1
            - 3 / 8 * einsum("ik,jk->ij",
                             einsum("kb,ib->ik", t1_1, F.ov), einsum("ja,ka->jk", t1_1, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 3 / 16 * einsum("jl,il->ij",
                              einsum("klab,jkab->jl", t2_1, hf.oovv),
                              einsum("ic,lc->il", t1_1, t1_1))
            # N^4: O^3V^1 / N^4: O^3V^1
            - 0.5 * einsum("jl,il->ij",
                           einsum("kb,jklb->jl", t1_1, hf.ooov), einsum("ia,la->il", t1_1, t1_1))
            # N^3: O^2V^1 / N^2: O^1V^1
            - 1 / 8 * einsum("ik,jk->ij",
                             einsum("ib,kb->ik", t1_1, F.ov), einsum("ja,ka->jk", t1_1, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 16 * einsum("jl,il->ij",
                              einsum("jkab,klab->jl", t2_1, hf.oovv),
                              einsum("ic,lc->il", t1_1, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 / 32 * einsum("il,jl->ij",
                              einsum("imcd,lmcd->il", t2_1, hf.oovv),
                              einsum("jkab,klab->jl", t2_1, t2_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("im,jm->ij",
                            einsum("ikab,kmab->im", t2_1, t2_1),
                            einsum("lc,jlmc->jm", t1_1, hf.ooov))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("il,jl->ij",
                            einsum("ikab,klab->il", t2_1, t2_1),
                            einsum("mc,lmjc->jl", t1_1, hf.ooov))
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("jkbc,ikbc->ij",
                            einsum("ac,jkab->jkbc",
                                   einsum("lmad,lmcd->ac", t2_1, t2_1), t2_1), hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ikab,jkab->ij",
                             einsum("km,imab->ikab",
                                    einsum("klcd,lmcd->km", t2_1, t2_1), t2_1), hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("jl,il->ij",
                              einsum("jkab,klab->jl", t2_1, t2_1), einsum("ic,lc->il", t1_1, F.ov))
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 16 * einsum("iklm,jklm->ij",
                              einsum("lmcd,ikcd->iklm", t2_1, hf.oovv),
                              einsum("jkab,lmab->jklm", t2_1, t2_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 / 32 * einsum("il,jl->ij",
                              einsum("lmcd,imcd->il", t2_1, hf.oovv),
                              einsum("jkab,klab->jl", t2_1, t2_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 3 / 16 * einsum("il,jl->ij",
                              einsum("ikab,klab->il", t2_1, t2_1), einsum("lc,jc->jl", t1_1, F.ov))
        ).symmetrise())
        # fourth order contributions without delta
        # Apply (1 + P_ijP_ab) to:
        + 2 * (
            + 1 * einsum("ka,jkib->iajb", t1_3, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ic,jabc->iajb", t1_3, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ikac,jkbc->iajb", t2_3, hf.oovv)  # N^6: O^3V^3 / N^4: O^2V^2
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("id,jabd->iajb", einsum("kc,ikcd->id", t1_1, t2_2), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("la,jlib->iajb", einsum("klac,kc->la", t2_1, t1_2), hf.ooov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 * einsum("kmab,ikjm->iajb", einsum("lmac,klbc->kmab", t2_1, t2_2), hf.oooo)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("id,jabd->iajb", einsum("ikcd,kc->id", t2_1, t1_2), hf.ovvv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 * einsum("ikbd,jakd->iajb", einsum("ilcd,klbc->ikbd", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb", einsum("ld,ld->", t1_1, t1_1) * t2_1, hf.oovv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb", einsum("kc,ibjc->ijkb", t1_2, hf.ovov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb", einsum("jkbc,ic->ijkb", t2_2, F.ov), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijlb,la->iajb", einsum("jkbc,iklc->ijlb", t2_2, hf.ooov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jakb->iajb", einsum("ic,kc->ik", t1_1, t1_2), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb", einsum("ic,jkbc->ijkb", t1_1, t2_2), F.ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb", einsum("kc,ibjc->ijkb", t1_1, hf.ovov), t1_2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jakb->iajb", einsum("kc,ic->ik", t1_1, t1_2), hf.ovov)
            # N^6: O^3V^3 / N^6: O^3V^3
            + 0.5 * einsum("ilad,jlbd->iajb", einsum("kc,iklacd->ilad", t1_1, t3_2), hf.oovv)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 0.5 * einsum("ikac,jkbc->iajb", einsum("id,kdac->ikac", t1_1, hf.ovvv), t2_2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijka,kb->iajb", einsum("ikac,jc->ijka", t2_1, F.ov), t1_2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijka,kb->iajb", einsum("ikac,jc->ijka", t2_1, t1_2), F.ov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijka,kb->iajb", einsum("ilac,jlkc->ijka", t2_1, hf.ooov), t1_2)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jlbc,ilac->iajb", einsum("jkcd,kbld->jlbc", t2_2, hf.ovov), t2_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 0.5 * einsum("jkbd,ikad->iajb", einsum("jc,kcbd->jkbd", t1_2, hf.ovvv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbd,ikad->iajb", einsum("klbc,jcld->jkbd", t2_2, hf.ovov), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb", einsum("klad,idlc->ikac", t2_1, hf.ovov), t2_2)
            # N^6: O^2V^4 / N^4: V^4
            + 0.5 * einsum("abcd,idjc->iajb", einsum("klad,klbc->abcd", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb", einsum("ilcd,kdla->ikac", t2_1, hf.ovov), t2_2)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkl,kbla->iajb", einsum("ilcd,jkcd->ijkl", t2_1, t2_2), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ka,jkib->iajb", einsum("lc,lc->", t1_1, t1_1) * t1_1, hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ic,jabc->iajb", einsum("kd,kd->", t1_1, t1_1) * t1_1, hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,ka->iajb", einsum("jc,ibkc->ijkb", t1_2, hf.ovov), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,ka->iajb", einsum("klbc,iljc->ijkb", t2_2, hf.ooov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijka,kb->iajb", einsum("la,ikjl->ijka", t1_1, hf.oooo), t1_2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijka,kb->iajb", einsum("ic,jakc->ijka", t1_1, hf.ovov), t1_2)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 1 * einsum("ijkd,kbad->iajb", einsum("ic,jkcd->ijkd", t1_1, t2_2), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ka,jkib->iajb", einsum("lc,klac->ka", t1_1, t2_2), hf.ooov)
            # N^5: O^1V^4 / N^4: V^4
            - 1 * einsum("iabc,jc->iajb", einsum("id,acbd->iabc", t1_1, hf.vvvv), t1_2)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijka,kb->iajb", einsum("klac,jlic->ijka", t2_1, hf.ooov), t1_2)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 * einsum("jlad,ibld->iajb", einsum("klac,jkcd->jlad", t2_1, t2_2), hf.ovov)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 1 * einsum("ijkd,kabd->iajb", einsum("ikcd,jc->ijkd", t2_1, t1_2), hf.ovvv)
            # N^6: O^2V^4 / N^4: V^4
            - 1 * einsum("ijde,adbe->iajb", einsum("ikce,jkcd->ijde", t2_1, t2_2), hf.vvvv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("jb,ia->iajb", einsum("kc,jkbc->jb", t1_2, hf.oovv), t1_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijkb,ka->iajb", einsum("jkcd,ibcd->ijkb", t2_2, hf.ovvv), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("iklb,klja->iajb", einsum("ic,klbc->iklb", t1_1, t2_2), hf.ooov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("jb,ia->iajb", einsum("kc,jkbc->jb", t1_1, hf.oovv), t1_2)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ikla,jklb->iajb", einsum("lc,ikac->ikla", t1_1, t2_2), hf.ooov)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("jkbc,ikac->iajb", einsum("kd,jdbc->jkbc", t1_1, hf.ovvv), t2_2)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ikla,jlkb->iajb", einsum("ilac,kc->ikla", t2_1, t1_2), hf.ooov)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("jkla,klib->iajb", einsum("klac,jc->jkla", t2_1, t1_2), hf.ooov)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 0.5 * einsum("mabd,imjd->iajb", einsum("klac,klmbcd->mabd", t2_1, t3_2), hf.ooov)
            # N^8: O^5V^3 / N^6: O^3V^3
            - 0.5 * einsum("ijkmbc,kmac->iajb", einsum("jklbcd,ilmd->ijkmbc", t3_2, hf.ooov), t2_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("jkbd,ikad->iajb", einsum("kc,jcbd->jkbd", t1_2, hf.ovvv), t2_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijka,kb->iajb", einsum("ikcd,jacd->ijka", t2_1, hf.ovvv), t1_2)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.5 * einsum("ijle,lbae->iajb", einsum("ikcd,jklcde->ijle", t2_1, t3_2), hf.ovvv)
            # N^8: O^4V^4 / N^6: O^3V^3
            - 0.5 * einsum("ijlbde,lead->iajb", einsum("ikce,jklbcd->ijlbde", t2_1, t3_2), hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ka,jkib->iajb", einsum("lmcd,lmcd->", t2_1, t2_1) * t1_1, hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ic,jabc->iajb", einsum("klde,klde->", t2_1, t2_1) * t1_1, hf.ovvv)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("jkbc,ikac->iajb", einsum("klmbcd,lmjd->jkbc", t3_2, hf.ooov), t2_1)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 0.25 * einsum("jkbc,ikac->iajb", einsum("jklcde,lbde->jkbc", t3_2, hf.ovvv), t2_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("ijklbc,klac->iajb", einsum("jklbcd,id->ijklbc", t3_2, F.ov), t2_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("ijlb,la->iajb", einsum("ikcd,jklbcd->ijlb", t2_1, t3_2), F.ov)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ad,ibjd->iajb", einsum("klcd,klac->ad", t2_1, t2_2), hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("il,jalb->iajb", einsum("klcd,ikcd->il", t2_1, t2_2), hf.ovov)
            # N^7: O^4V^3 / N^6: O^3V^3
            - 0.25 * einsum("ilma,jlmb->iajb", einsum("kmcd,iklacd->ilma", t2_1, t3_2), hf.ooov)
            # N^6: O^3V^3 / N^6: O^3V^3
            - 0.25 * einsum("ka,jkib->iajb", einsum("lmcd,klmacd->ka", t2_1, t3_2), hf.ooov)
            # N^7: O^3V^4 / N^6: O^3V^3
            - 0.25 * einsum("iade,jebd->iajb", einsum("klce,iklacd->iade", t2_1, t3_2), hf.ovvv)
            # N^6: O^3V^3 / N^6: O^3V^3
            - 0.25 * einsum("ic,jabc->iajb", einsum("klde,iklcde->ic", t2_1, t3_2), hf.ovvv)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.25 * einsum("ijlb,la->iajb", einsum("jklbcd,ikcd->ijlb", t3_2, hf.oovv), t1_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.25 * einsum("ijklbc,klac->iajb", einsum("id,jklbcd->ijklbc", t1_1, t3_2), hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("jmbc,imac->iajb", einsum("klbc,jmkl->jmbc", t2_2, hf.oooo), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb", einsum("lmac,iklm->ikac", t2_1, hf.oooo), t2_2)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("ac,ibjc->iajb", einsum("klad,klcd->ac", t2_1, t2_2), hf.ovov)
            # N^7: O^3V^4 / N^6: O^3V^3
            + 0.25 * einsum("jace,ibce->iajb", einsum("klad,jklcde->jace", t2_1, t3_2), hf.ovvv)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("jkbe,ikae->iajb", einsum("jkcd,becd->jkbe", t2_2, hf.vvvv), t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ik,jakb->iajb", einsum("ilcd,klcd->ik", t2_1, t2_2), hf.ovov)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 0.25 * einsum("ikmb,kmja->iajb", einsum("ilcd,klmbcd->ikmb", t2_1, t3_2), hf.ooov)
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("ikac,jkbc->iajb", einsum("ikde,acde->ikac", t2_1, hf.vvvv), t2_2)
            # N^8: O^4V^4 / N^6: O^3V^3
            + 1 / 8 * einsum("ijklbe,klae->iajb",
                             einsum("jklbcd,iecd->ijklbe", t3_2, hf.ovvv), t2_1)
            # N^8: O^5V^3 / N^6: O^3V^3
            + 1 / 8 * einsum("ijklmb,klma->iajb",
                             einsum("imcd,jklbcd->ijklmb", t2_1, t3_2), hf.ooov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("jk,kb->jb", einsum("kc,jc->jk", t1_1, F.ov), t1_1), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("jl,lb->jb", einsum("kc,jklc->jl", t1_1, hf.ooov), t1_1), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("jk,kb->jb", einsum("jc,kc->jk", t1_1, t1_1), F.ov), t1_1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("bd,jd->jb", einsum("kc,kdbc->bd", t1_1, hf.ovvv), t1_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("jkld,klbd->jb",
                                  einsum("kc,jdlc->jkld", t1_1, hf.ovov), t2_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("jkld,kdlb->jb",
                                  einsum("kc,jlcd->jkld", t1_1, t2_1), hf.ovov), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("kc,jkbc->jb", einsum("ld,klcd->kc", t1_1, t2_1), hf.oovv), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("jlmc,lmbc->jb",
                                  einsum("klcd,jkmd->jlmc", t2_1, hf.ooov), t2_1), t1_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 0.5 * einsum("jb,ia->iajb",
                           einsum("jkde,kebd->jb",
                                  einsum("klcd,jlce->jkde", t2_1, t2_1), hf.ovvv), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijla,lb->iajb",
                           einsum("ijkl,ka->ijla",
                                  einsum("kc,iljc->ijkl", t1_1, hf.ooov), t1_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkb,ka->iajb",
                           einsum("il,jklb->ijkb", einsum("ic,lc->il", t1_1, t1_1), hf.ooov), t1_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ijkb,ka->iajb",
                           einsum("ijbd,kd->ijkb",
                                  einsum("ic,jdbc->ijbd", t1_1, hf.ovvv), t1_1), t1_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ijkb,ka->iajb",
                           einsum("ikbd,jd->ijkb",
                                  einsum("kc,ibcd->ikbd", t1_1, hf.ovvv), t1_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijlb,la->iajb",
                           einsum("jk,klib->ijlb", einsum("jc,kc->jk", t1_1, t1_1), hf.ooov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ka,jkib->iajb",
                           einsum("kl,la->ka", einsum("kc,lc->kl", t1_1, t1_1), t1_1), hf.ooov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijlb,la->iajb",
                           einsum("iklc,jkbc->ijlb",
                                  einsum("kmcd,lmid->iklc", t2_1, hf.ooov), t2_1), t1_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("jkab,ik->iajb",
                           einsum("jd,kbad->jkab", t1_1, hf.ovvv), einsum("ic,kc->ik", t1_1, t1_1))
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ic,jabc->iajb",
                           einsum("ik,kc->ic", einsum("id,kd->ik", t1_1, t1_1), t1_1), hf.ovvv)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ia,jb->iajb",
                           einsum("ld,ilad->ia", t1_1, t2_1), einsum("kc,jkbc->jb", t1_1, hf.oovv))
            # N^6: O^3V^3 / N^4: O^1V^3
            + 0.5 * einsum("ikac,jkbc->iajb",
                           einsum("ilae,klce->ikac",
                                  einsum("id,lade->ilae", t1_1, hf.ovvv), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("jm,kmbc->jkbc",
                                  einsum("ld,lmjd->jm", t1_1, hf.ooov), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("be,jkce->jkbc",
                                  einsum("ld,lbde->be", t1_1, hf.ovvv), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("klbd,jlcd->jkbc",
                                  einsum("kmbe,lmde->klbd", t2_1, t2_1), hf.oovv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("jmce,kmbe->jkbc",
                                  einsum("jlcd,lmde->jmce", t2_1, t2_1), hf.oovv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikbd,jkad->iajb",
                           einsum("kmbe,imde->ikbd", t2_1, hf.oovv),
                           einsum("klac,jlcd->jkad", t2_1, t2_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikbd,jkad->iajb",
                           einsum("imde,kmbe->ikbd", t2_1, hf.oovv),
                           einsum("klac,jlcd->jkad", t2_1, t2_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.5 * einsum("ikac,jkbc->iajb",
                           einsum("kmae,imce->ikac",
                                  einsum("klad,lmde->kmae", t2_1, t2_1), t2_1), hf.oovv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("lc,iljc->ij", t1_1, hf.ooov),
                         einsum("ka,kb->ab", t1_1, t1_1))
            # N^4: O^1V^3 / N^4: O^1V^3
            - 1 * einsum("ab,ij->iajb",
                         einsum("kd,kabd->ab", t1_1, hf.ovvv), einsum("ic,jc->ij", t1_1, t1_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            - 3 / 8 * einsum("jkbc,ikac->iajb",
                             einsum("jl,klbc->jkbc", einsum("ld,jd->jl", t1_1, F.ov), t2_1), t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 3 / 8 * einsum("ijla,lb->iajb",
                             einsum("jklc,ikac->ijla",
                                    einsum("ld,jkcd->jklc", t1_1, t2_1), t2_1), F.ov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("jb,ia->iajb",
                           einsum("kl,jklb->jb", einsum("kc,lc->kl", t1_1, t1_1), hf.ooov), t1_1)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("jb,ia->iajb",
                           einsum("cd,jdbc->jb", einsum("kc,kd->cd", t1_1, t1_1), hf.ovvv), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ic,jc->ij", t1_1, F.ov), einsum("ka,kb->ab", t1_1, t1_1))
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ic,jc->ij", t1_1, t1_1), einsum("ka,kb->ab", t1_1, F.ov))
            # N^6: O^3V^3 / N^4: O^2V^2
            - 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("jm,kmbc->jkbc",
                                  einsum("ld,jlmd->jm", t1_1, hf.ooov), t2_1), t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ilma,jlmb->iajb",
                           einsum("klmc,ikac->ilma",
                                  einsum("ld,kmcd->klmc", t1_1, t2_1), t2_1), hf.ooov)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("be,jkce->jkbc",
                                  einsum("ld,lebd->be", t1_1, hf.ovvv), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 0.5 * einsum("jkbc,ikac->iajb",
                           einsum("jlbe,klce->jkbc",
                                  einsum("ld,jebd->jlbe", t1_1, hf.ovvv), t2_1), t2_1)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("klac,klbc->ab", t2_1, t2_1),
                           einsum("md,imjd->ij", t1_1, hf.ooov))
            # N^5: O^3V^2 / N^4: O^1V^3
            - 0.5 * einsum("ij,ab->iajb",
                           einsum("ikcd,jkcd->ij", t2_1, t2_1),
                           einsum("le,labe->ab", t1_1, hf.ovvv))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("jb,ia->iajb",
                            einsum("jklc,klbc->jb",
                                   einsum("klcd,jd->jklc", t2_1, F.ov), t2_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("jb,ia->iajb",
                            einsum("jl,lb->jb", einsum("jkcd,klcd->jl", t2_1, t2_1), F.ov), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ij,ab->iajb",
                            einsum("ilcd,jlcd->ij", t2_1, t2_1),
                            einsum("ka,kb->ab", t1_1, F.ov))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ijlb,la->iajb",
                            einsum("im,jlmb->ijlb",
                                   einsum("ikcd,kmcd->im", t2_1, t2_1), hf.ooov), t1_1)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ijmb,ma->iajb",
                            einsum("bd,jmid->ijmb",
                                   einsum("klbc,klcd->bd", t2_1, t2_1), hf.ooov), t1_1)
            # N^5: O^2V^3 / N^4: O^2V^2
            - 0.25 * einsum("ab,ij->iajb",
                            einsum("klad,klbd->ab", t2_1, t2_1),
                            einsum("ic,jc->ij", t1_1, F.ov))
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ic,jabc->iajb",
                            einsum("il,lc->ic", einsum("ikde,klde->il", t2_1, t2_1), t1_1), hf.ovvv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ae,ijbe->iajb",
                            einsum("klac,klce->ae", t2_1, t2_1),
                            einsum("id,jebd->ijbe", t1_1, hf.ovvv))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ka,jkib->iajb",
                            einsum("klmc,lmac->ka",
                                   einsum("kd,lmcd->klmc", t1_1, t2_1), t2_1), hf.ooov)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("jl,ilab->iajb",
                            einsum("jkcd,klcd->jl", t2_1, t2_1),
                            einsum("ie,labe->ilab", t1_1, hf.ovvv))
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 / 8 * einsum("ijlb,la->iajb",
                             einsum("iklc,jkbc->ijlb",
                                    einsum("ikcd,ld->iklc", t2_1, F.ov), t2_1), t1_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("il,klac->ikac", einsum("id,ld->il", t1_1, F.ov), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbc,ikac->iajb",
                             einsum("jklm,lmbc->jkbc",
                                    einsum("lmde,jkde->jklm", t2_1, hf.oovv), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("jkbc,ikac->iajb",
                             einsum("jklm,lmbc->jkbc",
                                    einsum("jkde,lmde->jklm", t2_1, t2_1), hf.oovv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("iklm,lmac->ikac",
                                    einsum("ikde,lmde->iklm", t2_1, t2_1), t2_1), hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("imad,kmcd->ikac",
                                    einsum("ilae,lmde->imad", t2_1, hf.oovv), t2_1), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("jm,imab->iajb",
                              einsum("jlde,lmde->jm", t2_1, hf.oovv),
                              einsum("ikac,kmbc->imab", t2_1, t2_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 / 16 * einsum("be,ijae->iajb",
                              einsum("lmbd,lmde->be", t2_1, hf.oovv),
                              einsum("ikac,jkce->ijae", t2_1, t2_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("jl,lb->jb", einsum("klcd,jkcd->jl", t2_1, hf.oovv), t1_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("klmb,jklm->jb",
                                   einsum("kc,lmbc->klmb", t1_1, t2_1), hf.oooo), t1_1)
            # N^5: O^1V^4 / N^4: V^4
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("jcde,bcde->jb",
                                   einsum("kc,jkde->jcde", t1_1, t2_1), hf.vvvv), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("jklc,klbc->jb",
                                   einsum("jd,klcd->jklc", t1_1, t2_1), hf.oovv), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("km,jkmb->jb",
                                   einsum("klcd,lmcd->km", t2_1, t2_1), hf.ooov), t1_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("ce,jebc->jb",
                                   einsum("klcd,klde->ce", t2_1, t2_1), hf.ovvv), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ij,ab->iajb",
                            einsum("ilcd,jlcd->ij", t2_1, hf.oovv),
                            einsum("ka,kb->ab", t1_1, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijlb,la->iajb",
                            einsum("jm,lmib->ijlb",
                                   einsum("jkcd,kmcd->jm", t2_1, t2_1), hf.ooov), t1_1)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.25 * einsum("ab,ij->iajb",
                            einsum("klad,klbd->ab", t2_1, hf.oovv), einsum("ic,jc->ij", t1_1, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("lmbd,lmjd->jb", t2_1, hf.ooov),
                            einsum("kc,ikac->ia", t1_1, t2_1))
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("jb,ia->iajb",
                            einsum("jlde,lbde->jb", t2_1, hf.ovvv),
                            einsum("kc,ikac->ia", t1_1, t2_1))
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.25 * einsum("be,ijae->iajb",
                            einsum("klbc,klce->be", t2_1, t2_1),
                            einsum("id,jade->ijae", t1_1, hf.ovvv))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb",
                            einsum("kl,ilac->ikac",
                                   einsum("kmde,lmde->kl", t2_1, t2_1), t2_1), hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ijlm,lmab->iajb",
                            einsum("imde,jlde->ijlm", t2_1, hf.oovv),
                            einsum("klac,kmbc->lmab", t2_1, t2_1))
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("ijcd,abcd->iajb",
                            einsum("jmce,imde->ijcd", t2_1, hf.oovv),
                            einsum("klac,klbd->abcd", t2_1, t2_1))
            # N^6: O^2V^4 / N^4: V^4
            + 0.25 * einsum("ijcd,abcd->iajb",
                            einsum("jmce,imde->ijcd", t2_1, t2_1),
                            einsum("klac,klbd->abcd", t2_1, hf.oovv))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ijlm,lmab->iajb",
                            einsum("imde,jlde->ijlm", t2_1, t2_1),
                            einsum("klac,kmbc->lmab", t2_1, hf.oovv))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 0.25 * einsum("ikac,jkbc->iajb",
                            einsum("cd,ikad->ikac",
                                   einsum("lmce,lmde->cd", t2_1, t2_1), t2_1), hf.oovv)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 1 / 8 * einsum("jb,ia->iajb",
                             einsum("jkle,klbe->jb",
                                    einsum("klcd,jecd->jkle", t2_1, hf.ovvv), t2_1), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("jb,ia->iajb",
                             einsum("jklm,klmb->jb",
                                    einsum("jmcd,klcd->jklm", t2_1, t2_1), hf.ooov), t1_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 / 8 * einsum("ijlb,la->iajb",
                             einsum("iklc,jkbc->ijlb",
                                    einsum("ld,ikcd->iklc", t1_1, hf.oovv), t2_1), t1_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("il,klac->ikac",
                                    einsum("id,ld->il", t1_1, t1_1), hf.oovv), t2_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("imad,kmcd->ikac",
                                    einsum("ilae,lmde->imad", t2_1, t2_1), t2_1), hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 16 * einsum("jm,imab->iajb",
                              einsum("lmde,jlde->jm", t2_1, hf.oovv),
                              einsum("ikac,kmbc->imab", t2_1, t2_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 16 * einsum("be,ijae->iajb",
                              einsum("lmde,lmbd->be", t2_1, hf.oovv),
                              einsum("ikac,jkce->ijae", t2_1, t2_1))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 16 * einsum("im,jmab->iajb",
                              einsum("ilde,lmde->im", t2_1, t2_1),
                              einsum("kmac,jkbc->jmab", t2_1, hf.oovv))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 1 / 16 * einsum("ikac,jkbc->iajb",
                              einsum("ae,ikce->ikac",
                                     einsum("lmad,lmde->ae", t2_1, t2_1), t2_1), hf.oovv)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 3 / 8 * einsum("ijlb,la->iajb",
                             einsum("iklc,jkbc->ijlb",
                                    einsum("ld,ikcd->iklc", t1_1, t2_1), hf.oovv), t1_1)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 3 / 8 * einsum("ikac,jkbc->iajb",
                             einsum("il,klac->ikac",
                                    einsum("id,ld->il", t1_1, t1_1), t2_1), hf.oovv)
            # N^6: O^3V^3 / N^4: O^2V^2
            + 3 / 16 * einsum("be,ijae->iajb",
                              einsum("lmbd,lmde->be", t2_1, t2_1),
                              einsum("ikac,jkce->ijae", t2_1, hf.oovv))
            # N^6: O^3V^3 / N^4: O^2V^2
            + 3 / 16 * einsum("jm,imab->iajb",
                              einsum("jlde,lmde->jm", t2_1, t2_1),
                              einsum("ikac,kmbc->imab", t2_1, hf.oovv))
        )
    ).symmetrise((0, 2), (1, 3))  # fmt: skip
