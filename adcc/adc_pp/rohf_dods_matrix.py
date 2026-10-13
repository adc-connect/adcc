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


def block_pphh_pphh_1(hf, mp, intermediates):
    F = hf.operators.fock
    # TODO first-order diagonal, for now zeroth order only
    diagonal = diagonal_pphh_pphh_0(hf)

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
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    t1_2 = mp.ts2(b.ov)
    t2_2 = mp.td2(b.oovv)
    i1 = intermediates.rohf_dods_adc3_i1
    i2 = intermediates.rohf_dods_adc3_i2
    lower = block_ph_ph_2(hf, mp, intermediates)
    # TODO third-order diagonal, for now zeroth order only
    diagonal = lower.diagonal

    def apply(ampl):
        ur1 = ampl.ph
        return lower.apply(ampl) + AmplitudeVector(ph=(
            # 3 (via i1 and i2)
            + 1 * einsum("ib,ab->ia", ur1, i1)  # N^3: O^1V^2 / N^2: V^2
            - 1 * einsum("ij,ja->ia", i2, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
            # 3
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("jk,ijka->ia", einsum("kb,jb->jk", ur1, t1_2), hf.ooov)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("bc,icab->ia", einsum("jc,jb->bc", ur1, t1_2), hf.ovvv)
            # N^4: O^3V^1 / N^4: O^3V^1
            - 1 * einsum("ij,ja->ia", einsum("kb,jkib->ij", ur1, hf.ooov), t1_2)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 1 * einsum("ab,ib->ia", einsum("jc,jabc->ab", ur1, hf.ovvv), t1_2)
            # N^3: O^2V^1 / N^2: O^1V^1
            - 0.5 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, F.ov), t1_2)
            # N^3: O^2V^1 / N^2: O^1V^1
            - 0.5 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, t1_2), F.ov)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,ikac->ia", einsum("jb,jkbc->kc", ur1, t2_2), hf.oovv)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("jb,ijab->ia", einsum("kc,jkbc->jb", ur1, hf.oovv), t2_2)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ijkb,jkab->ia", einsum("ic,jkbc->ijkb", ur1, hf.oovv), t2_2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkc,jkac->ia", einsum("ib,jkbc->ijkc", ur1, t2_2), hf.oovv)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ij,ja->ia",
                           einsum("jb,ib->ij", einsum("kc,jckb->jb", t1_1, hf.ovov), ur1),
                           t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ja,ij->ia",
                           einsum("kc,jcka->ja", t1_1, hf.ovov),
                           einsum("ib,jb->ij", ur1, t1_1))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ja,ij->ia",
                           einsum("kc,jkac->ja", t1_1, t2_1),
                           einsum("ib,jb->ij", ur1, F.ov))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ij,ja->ia",
                           einsum("jb,ib->ij", einsum("kc,jkbc->jb", t1_1, t2_1), ur1),
                           F.ov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ijkc,jkac->ia",
                           einsum("ijcd,kd->ijkc", einsum("ib,jdbc->ijcd", ur1, hf.ovvv), t1_1),
                           t2_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("ijcd,jdac->ia",
                           einsum("ijkc,kd->ijcd", einsum("ib,jkbc->ijkc", ur1, t2_1), t1_1),
                           hf.ovvv)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkc,jkac->ia",
                           einsum("ijld,klcd->ijkc", einsum("ib,jdlb->ijld", ur1, hf.ovov), t2_1),
                           t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijld,jdla->ia",
                           einsum("ijkc,klcd->ijld", einsum("ib,jkbc->ijkc", ur1, t2_1), t2_1),
                           hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,ka->ia",
                           einsum("ic,kc->ik", einsum("jb,ibjc->ic", ur1, hf.ovov), t1_1),
                           t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,ka->ia",
                           einsum("kc,ic->ik", einsum("jb,jkbc->kc", ur1, t2_1), F.ov),
                           t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("il,la->ia",
                           einsum("kc,iklc->il", einsum("jb,jkbc->kc", ur1, t2_1), hf.ooov),
                           t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jc,icja->ia",
                           einsum("jk,kc->jc", einsum("jb,kb->jk", ur1, t1_1), t1_1),
                           hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("kc,ikac->ia",
                           einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, t1_1), F.ov),
                           t2_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("kc,ikac->ia",
                           einsum("jl,jklc->kc", einsum("jb,lb->jl", ur1, t1_1), hf.ooov),
                           t2_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,ka->ia",
                           einsum("ic,kc->ik", t1_1, t1_1),
                           einsum("jb,jakb->ka", ur1, hf.ovov))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,ka->ia",
                           einsum("kc,ic->ik", einsum("jb,jkbc->kc", ur1, t2_1), t1_1),
                           F.ov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("kb,ibka->ia",
                           einsum("jk,jb->kb", einsum("jc,kc->jk", t1_1, t1_1), ur1),
                           hf.ovov)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("kc,ikac->ia",
                           einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, F.ov), t1_1),
                           t2_1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 0.5 * einsum("ad,id->ia",
                           einsum("kc,kdac->ad", einsum("jb,jkbc->kc", ur1, t2_1), hf.ovvv),
                           t1_1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 0.5 * einsum("kc,ikac->ia",
                           einsum("bd,kdbc->kc", einsum("jb,jd->bd", ur1, t1_1), hf.ovvv),
                           t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("kc,ikac->ia",
                           einsum("jkld,jdlc->kc", einsum("jb,klbd->jkld", ur1, t2_1), hf.ovov),
                           t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("kc,ikac->ia",
                           einsum("jkld,jlcd->kc", einsum("jb,kdlb->jkld", ur1, hf.ovov), t2_1),
                           t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("iklc,klac->ia",
                           einsum("jkld,icjd->iklc", einsum("jb,klbd->jkld", ur1, t2_1), hf.ovov),
                           t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ikld,klad->ia",
                           einsum("kc,idlc->ikld", einsum("jb,jkbc->kc", ur1, t2_1), hf.ovov),
                           t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ikld,kdla->ia",
                           einsum("kc,ilcd->ikld", einsum("jb,jkbc->kc", ur1, t2_1), t2_1),
                           hf.ovov)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijkl,jkla->ia",
                           einsum("ikcd,jlcd->ijkl", t2_1, t2_1),
                           einsum("jb,kalb->jkla", ur1, hf.ovov))
            # N^4: O^4 / N^4: O^4
            - 1 * einsum("ik,ka->ia",
                         einsum("jl,iljk->ik", einsum("jb,lb->jl", ur1, t1_1), hf.oooo),
                         t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ik,ka->ia",
                         einsum("bc,ibkc->ik", einsum("jb,jc->bc", ur1, t1_1), hf.ovov),
                         t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ik,ka->ia",
                         einsum("jklc,iljc->ik", einsum("jb,klbc->jklc", ur1, t2_1), hf.ooov),
                         t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ac,ic->ia",
                         einsum("jk,jakc->ac", einsum("jb,kb->jk", ur1, t1_1), hf.ovov),
                         t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("iklc,klac->ia",
                         einsum("jk,jlic->iklc", einsum("jb,kb->jk", ur1, t1_1), hf.ooov),
                         t2_1)
            # N^4: V^4 / N^4: V^4
            - 1 * einsum("ac,ic->ia",
                         einsum("bd,adbc->ac", einsum("jb,jd->bd", ur1, t1_1), hf.vvvv),
                         t1_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ijkd,jkad->ia",
                         einsum("ic,jkcd->ijkd", t1_1, t2_1),
                         einsum("jb,kbad->jkad", ur1, hf.ovvv))
            # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ijkd,jkad->ia",
                         einsum("jc,ikcd->ijkd", t1_1, t2_1),
                         einsum("jb,kabd->jkad", ur1, hf.ovvv))
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ka,ik->ia",
                         einsum("lc,klac->ka", t1_1, t2_1),
                         einsum("jb,jkib->ik", ur1, hf.ooov))
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("jk,ikja->ia",
                         einsum("kb,jb->jk", einsum("lc,klbc->kb", t1_1, t2_1), ur1),
                         hf.ooov)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 1 * einsum("ic,ac->ia",
                         einsum("kd,ikcd->ic", t1_1, t2_1),
                         einsum("jb,jabc->ac", ur1, hf.ovvv))
            # N^4: O^1V^3 / N^4: O^1V^3
            - 1 * einsum("bc,ibac->ia",
                         einsum("jc,jb->bc", einsum("kd,jkcd->jc", t1_1, t2_1), ur1),
                         hf.ovvv)
            # N^6: O^5V^1 / N^4: O^2V^2
            - 1 * einsum("iklc,klac->ia",
                         einsum("jkmc,imjl->iklc", einsum("jb,kmbc->jkmc", ur1, t2_1), hf.oooo),
                         t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("iklc,klac->ia",
                         einsum("ijld,jkcd->iklc", einsum("jb,ibld->ijld", ur1, hf.ovov), t2_1),
                         t2_1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijld,jald->ia",
                         einsum("jklc,ikcd->ijld", einsum("jb,klbc->jklc", ur1, t2_1), t2_1),
                         hf.ovov)
            # N^6: O^3V^3 / N^4: V^4
            - 1 * einsum("ijde,jade->ia",
                         einsum("ikcd,jkce->ijde", t2_1, t2_1),
                         einsum("jb,aebd->jade", ur1, hf.vvvv))
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,kc->", einsum("jb,jkbc->kc", ur1, hf.oovv), t1_1) * t1_1
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ik,ka->ia",
                           einsum("ijcd,jkcd->ik", einsum("jb,ibcd->ijcd", ur1, hf.ovvv), t2_1),
                           t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,ikac->ia", einsum("jb,jb->", ur1, t1_1) * t1_1, hf.oovv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.5 * einsum("ijcd,jacd->ia",
                           einsum("jk,ikcd->ijcd", einsum("jb,kb->jk", ur1, t1_1), t2_1),
                           hf.ovvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijkl,klja->ia",
                           einsum("jklc,ic->ijkl", einsum("jb,klbc->jklc", ur1, t2_1), t1_1),
                           hf.ooov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.5 * einsum("iklc,klac->ia",
                           einsum("ijkl,jc->iklc", einsum("jb,klib->ijkl", ur1, hf.ooov), t1_1),
                           t2_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kc,ikac->ia",
                           einsum("kl,lc->kc", einsum("jb,jklb->kl", ur1, hf.ooov), t1_1),
                           t2_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("kl,ikla->ia",
                           einsum("kc,lc->kl", einsum("jb,jkbc->kc", ur1, t2_1), t1_1),
                           hf.ooov)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("kc,ikac->ia",
                           einsum("cd,kd->kc", einsum("jb,jdbc->cd", ur1, hf.ovvv), t1_1),
                           t2_1)
            # N^4: O^1V^3 / N^4: O^1V^3
            - 0.5 * einsum("cd,idac->ia",
                           einsum("kc,kd->cd", einsum("jb,jkbc->kc", ur1, t2_1), t1_1),
                           hf.ovvv)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ij,ja->ia",
                            einsum("jb,ib->ij", einsum("jkcd,kbcd->jb", t2_1, hf.ovvv), ur1),
                            t1_1)
            # N^5: O^2V^3 / N^4: O^1V^3
            - 0.25 * einsum("ja,ij->ia",
                            einsum("jkcd,kacd->ja", t2_1, hf.ovvv),
                            einsum("ib,jb->ij", ur1, t1_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("iklc,klac->ia",
                            einsum("id,klcd->iklc", einsum("jb,ibjd->id", ur1, hf.ovov), t2_1),
                            t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("jd,idja->ia",
                            einsum("jklc,klcd->jd", einsum("jb,klbc->jklc", ur1, t2_1), t2_1),
                            hf.ovov)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("il,la->ia",
                            einsum("ikcd,klcd->il", t2_1, t2_1),
                            einsum("jb,jalb->la", ur1, hf.ovov))
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("lb,ibla->ia",
                            einsum("jl,jb->lb", einsum("jkcd,klcd->jl", t2_1, t2_1), ur1),
                            hf.ovov)
            # N^6: O^3V^3 / N^4: V^4
            - 1 / 8 * einsum("ijkc,jkac->ia",
                             einsum("icde,jkde->ijkc", einsum("ib,bcde->icde", ur1, hf.vvvv), t2_1),
                             t2_1)
            # N^6: O^3V^3 / N^4: V^4
            - 1 / 8 * einsum("icde,acde->ia",
                             einsum("ijkc,jkde->icde", einsum("ib,jkbc->ijkc", ur1, t2_1), t2_1),
                             hf.vvvv)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkc,jkac->ia",
                            einsum("ijkl,lc->ijkc", einsum("ib,jklb->ijkl", ur1, hf.ooov), t1_1),
                            t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkl,jkla->ia",
                            einsum("ijkc,lc->ijkl", einsum("ib,jkbc->ijkc", ur1, t2_1), t1_1),
                            hf.ooov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("kc,ikac->ia",
                            einsum("jlmc,jklm->kc", einsum("jb,lmbc->jlmc", ur1, t2_1), hf.oooo),
                            t2_1)
            # N^5: O^1V^4 / N^4: V^4
            + 0.25 * einsum("kc,ikac->ia",
                            einsum("kbde,bcde->kc", einsum("jb,jkde->kbde", ur1, t2_1), hf.vvvv),
                            t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.25 * einsum("klma,iklm->ia",
                            einsum("kc,lmac->klma", einsum("jb,jkbc->kc", ur1, t2_1), t2_1),
                            hf.oooo)
            # N^5: O^1V^4 / N^4: V^4
            + 0.25 * einsum("icde,acde->ia",
                            einsum("kc,ikde->icde", einsum("jb,jkbc->kc", ur1, t2_1), t2_1),
                            hf.vvvv)
        ))  # fmt: skip

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
            - 1.0 * einsum("jc,jabc->ab", t1_1, hf.ovvv)
            - 0.5 * einsum("ia,ib->ab", t1_1, F.ov)
            + 0.25 * einsum("ijac,ijbc->ab", t2_1, hf.oovv)
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
            - 1.0 * einsum("ka,ikja->ij", t1_1, hf.ooov)
            - 0.5 * einsum("ia,ja->ij", t1_1, F.ov)
            + 0.25 * einsum("ikab,jkab->ij", t2_1, hf.oovv)  # N^5: O^3V^2 / N^4: O^2V^2
        )
    ).symmetrise()  # fmt: skip


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
def rohf_dods_adc3_i1(hf, mp, intermediates):
    # Third-order vv part of block_ph_ph_3 that is independent of the trial vector
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    t1_2 = mp.ts2(b.ov)
    return (
        # N^4: O^1V^3 / N^4: O^1V^3
        + 1 * einsum("jb,jabc->ac", t1_2, hf.ovvv)
        # N^4: O^1V^3 / N^4: O^1V^3
        - 1 * einsum("jb,jcab->ac", t1_2, hf.ovvv)
        # N^4: V^4 / N^4: V^4
        + 1 * einsum("cd,adbc->ab", einsum("jc,jd->cd", t1_1, t1_1), hf.vvvv)
        # N^4: O^1V^3 / N^4: O^1V^3
        + 1 * einsum("jc,jabc->ab", einsum("kd,jkcd->jc", t1_1, t2_1), hf.ovvv)
        # N^4: O^1V^3 / N^4: O^1V^3
        + 1 * einsum("jc,jbac->ab", einsum("kd,jkcd->jc", t1_1, t2_1), hf.ovvv)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jk,jakb->ab", einsum("jc,kc->jk", t1_1, t1_1), hf.ovov)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("kl,kalb->ab", einsum("jkcd,jlcd->kl", t2_1, t2_1), hf.ovov)
        # N^5: O^2V^3 / N^4: V^4
        - 0.5 * einsum("ce,aebc->ab", einsum("jkcd,jkde->ce", t2_1, t2_1), hf.vvvv)
    )  # fmt: skip


@register_as_intermediate
def rohf_dods_adc3_i2(hf, mp, intermediates):
    # Third-order oo part of block_ph_ph_3 that is independent of the trial vector
    F = hf.operators.fock
    t1_1 = mp.ts1(b.ov)
    t2_1 = mp.t2oo
    t1_2 = mp.ts2(b.ov)
    t2_2 = mp.td2(b.oovv)
    return (
        # N^4: O^3V^1 / N^4: O^3V^1
        - 1 * einsum("jb,jkib->ik", t1_2, hf.ooov)
        # N^4: O^3V^1 / N^4: O^3V^1
        + 1 * einsum("jb,ijkb->ik", t1_2, hf.ooov)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("ib,jb->ij", t1_2, F.ov)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("jb,ib->ij", t1_2, F.ov)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ijbc,jkbc->ik", t2_2, hf.oovv)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("jkbc,ikbc->ij", t2_2, hf.oovv)
        # N^4: O^4 / N^4: O^4
        - 1 * einsum("kl,iljk->ij", einsum("kb,lb->kl", t1_1, t1_1), hf.oooo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kb,ikjb->ij", einsum("lc,klbc->kb", t1_1, t2_1), hf.ooov)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kb,jkib->ij", einsum("lc,klbc->kb", t1_1, t2_1), hf.ooov)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("jb,ib->ij", einsum("kc,jckb->jb", t1_1, hf.ovov), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ib,jb->ij", einsum("kc,ickb->ib", t1_1, hf.ovov), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ib,jb->ij", einsum("kc,ikbc->ib", t1_1, t2_1), F.ov)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("jb,ib->ij", einsum("kc,jkbc->jb", t1_1, t2_1), F.ov)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("iklb,jklb->ij", einsum("lc,ikbc->iklb", t1_1, t2_1), hf.ooov)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("jklb,iklb->ij", einsum("lc,jkbc->jklb", t1_1, t2_1), hf.ooov)
        # N^6: O^3V^3 / N^4: O^2V^2
        - 0.5 * einsum("ilbd,jdlb->ij", einsum("ikbc,klcd->ilbd", t2_1, t2_1), hf.ovov)
        # N^6: O^3V^3 / N^4: O^2V^2
        - 0.5 * einsum("jlbd,idlb->ij", einsum("jkbc,klcd->jlbd", t2_1, t2_1), hf.ovov)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("bc,ibjc->ij", einsum("kb,kc->bc", t1_1, t1_1), hf.ovov)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("km,imjk->ij", einsum("klbc,lmbc->km", t2_1, t2_1), hf.oooo)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 0.5 * einsum("cd,icjd->ij", einsum("klbc,klbd->cd", t2_1, t2_1), hf.ovov)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("jb,ib->ij", einsum("klbc,kljc->jb", t2_1, hf.ooov), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ib,jb->ij", einsum("klbc,klic->ib", t2_1, hf.ooov), t1_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1 / 8 * einsum("iklm,jklm->ij", einsum("ikbc,lmbc->iklm", t2_1, t2_1), hf.oooo)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1 / 8 * einsum("jklm,iklm->ij", einsum("jkbc,lmbc->jklm", t2_1, t2_1), hf.oooo)
        # N^5: O^2V^3 / N^4: O^1V^3
        - 0.25 * einsum("jkbc,ikbc->ij", einsum("kd,jdbc->jkbc", t1_1, hf.ovvv), t2_1)
        # N^5: O^2V^3 / N^4: O^1V^3
        - 0.25 * einsum("ikbc,jkbc->ij", einsum("kd,idbc->ikbc", t1_1, hf.ovvv), t2_1)
    )  # fmt: skip
