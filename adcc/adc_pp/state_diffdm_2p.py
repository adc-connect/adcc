#!/usr/bin/env python3
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
from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import IsrMethod
from adcc.functions import einsum, zeros_like
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.TwoParticleDensity import TwoParticleDensity
from adcc.NParticleOperator import OperatorSymmetry

from .util import check_doubles_amplitudes, check_singles_amplitudes


def diffdm_isr0_2p(mp, amplitude, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude)
    u1 = amplitude.ph

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    dm = TwoParticleDensity(mp, symmetry=OperatorSymmetry.HERMITIAN)

    # one-particle ISR(0) diffdm
    p1_oo = -einsum("ia,la->il", u1, u1).evaluate()
    p1_vv = einsum("ka,kb->ab", u1, u1).evaluate()

    dm.oooo = (
        # N^4: O^4 / N^4: O^4
        - 4.0 * einsum("il,jk->ijkl", p1_oo, d_oo)
    ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    dm.ovov = (
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1.0 * einsum("ja,ib->iajb", u1, u1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", p1_vv, d_oo)
    )
    return dm


def diffdm_isr1s_2p(mp, amplitude, intermediates):
    dm = diffdm_isr0_2p(mp, amplitude, intermediates)  # Get ISR(0) result
    u1 = amplitude.ph

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)
    # TODO move to intermediates!
    # one-particle ISR(0) diffdm
    p1_oo = -einsum("ia,la->il", u1, u1).evaluate()

    # ISR(2) ISR intermediate (TODO Move to intermediates)
    ru1 = einsum("ijab,jb->ia", t2, u1).evaluate()
    # new ones
    ru1_ooov = einsum("kc,ijac->ijka", u1, t2).evaluate()

    dm.oovv += (
        # N^4: O^2V^2 / N^4: O^2V^2
        + 4.0 * (
            einsum("ib,ja->ijab", ru1, u1)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2.0 * (
            einsum("ijka,kb->ijab", ru1_ooov, u1)
        ).antisymmetrise(2, 3)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 2.0 * (
            einsum("jk,ikab->ijab", p1_oo, t2)
        ).antisymmetrise(0, 1)
    )

    return dm


def diffdm_isr1_2p(mp, amplitude, intermediates):
    dm = diffdm_isr1s_2p(mp, amplitude, intermediates)  # Get ISR(1)-s result
    u1 = amplitude.ph

    try:
        # ISR(1)-d
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
        u2 = amplitude.pphh

        dm.ooov += (
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2.0 * einsum("kb,ijab->ijka", u1, u2)
        )

        dm.ovvv += (
            # N^5: O^2V^3 / N^4: O^1V^3
            - 2.0 * einsum("ja,ijbc->iabc", u1, u2)
        )
    except ValueError:
        # no doubles contribution
        pass
    return dm


def diffdm_isr2_2p(mp, amplitude, intermediates):
    dm = diffdm_isr1_2p(mp, amplitude, intermediates)  # Get ISR(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.ph, amplitude.pphh
    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    dm.oooo += (
        # N^6: O^4V^2 / N^4: O^2V^2
        + 2.0 * einsum("ijab,klab->ijkl", u2, u2)
        # N^6: O^5V^1 / N^4: O^2V^2
        - 1.0 * einsum("klmc,ijmc->ijkl", einsum("mb,klbc->klmc", u1, t2),
                       einsum("ma,ijac->ijmc", u1, t2))
        + (
            # N^5: O^3V^2 / N^4: O^2V^2
            - 8.0 * einsum("ik,jl->ijkl", einsum("imab,kmab->ik", u2, u2), d_oo)
            # N^4: O^4 / N^4: O^4
            + 4.0 * einsum("il,jk->ijkl", einsum("ia,la->il", u1, u1), p0.oo)
            # N^6: O^5V^1 / N^4: O^2V^2
            + 4.0 * einsum("jlmc,ikmc->ijkl", einsum("lb,jmbc->jlmc", u1, t2),
                           einsum("ia,kmac->ikmc", u1, t2))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 4.0 * einsum("il,jk->ijkl",
                           einsum("lc,ic->il", einsum("nb,lnbc->lc", u1, t2),
                                  einsum("ma,imac->ic", u1, t2)), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 4.0 * einsum("ik,jl->ijkl", einsum("kmnc,imnc->ik",
                                                 einsum("nb,kmbc->kmnc", u1, t2),
                                                 einsum("na,imac->imnc", u1, t2)),
                           d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 2.0 * einsum("ik,jl->ijkl",
                           einsum("inbc,knbc->ik",
                                  einsum("mn,imbc->inbc",
                                         einsum("ia,la->il", u1, u1), t2), t2),
                           d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 4.0 * einsum("iklb,jb->ijkl",
                           einsum("ic,klbc->iklb", einsum("ma,imac->ic", u1, t2),
                                  t2), u1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 2.0 * einsum("jklm,im->ijkl",
                           einsum("jmbc,klbc->jklm", t2, t2),
                           einsum("ia,la->il", u1, u1))
        ).antisymmetrise(0, 1).symmetrise([(0, 2), (1, 3)])
        + (
            # N^4: O^4 / N^4: O^4
            + 4.0 * einsum("il,jk->ijkl", einsum("im,lm->il",
                                                 einsum("ia,la->il", u1, u1),
                                                 p0.oo), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 4.0 * einsum("ik,jl->ijkl",
                           einsum("kb,ib->ik",
                                  einsum("mc,kmbc->kb",
                                         einsum("na,mnac->mc", u1, t2), t2), u1),
                           d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3).symmetrise([(0, 2), (1, 3)])
    )
    dm.ooov += (
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1.0 * einsum("ijkl,la->ijka", einsum("klbc,ijbc->ijkl", u2, t2), u1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2.0 * einsum("kb,ijab->ijka", einsum("lc,klbc->kb", u1, u2), t2)
        + (
            # N^4: O^2V^2 / N^4: O^2V^2
            - 4.0 * einsum("ja,ik->ijka", einsum("lb,jlab->ja", u1, u2), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 2.0 * einsum("jk,ia->ijka", einsum("klbc,jlbc->jk", u2, t2), u1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 4.0 * einsum("jklb,ilab->ijka", einsum("jc,klbc->jklb", u1, u2), t2)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 2.0 * einsum("ja,ik->ijka",
                           einsum("jlmb,lmab->ja",
                                  einsum("jc,lmbc->jlmb", u1, u2), t2), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 4.0 * einsum("ia,jk->ijka",
                           einsum("lb,ilab->ia",
                                  einsum("mc,lmbc->lb", u1, u2), t2), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 2.0 * einsum("ia,jk->ijka",
                           einsum("il,la->ia",
                                  einsum("lmbc,imbc->il", u2, t2), u1), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 2.0 * einsum("jk,ia->ijka", einsum("kb,jb->jk", u1, p0.ov), u1)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 2.0 * einsum("jk,ia->ijka", einsum("jb,kb->jk", u1, u1), p0.ov)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 2.0 * einsum("ia,jk->ijka",
                           einsum("il,la->ia",
                                  einsum("lb,ib->il", u1, p0.ov), u1), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 2.0 * einsum("ia,jk->ijka",
                           einsum("il,la->ia", einsum("ia,la->il", u1, u1),
                                  p0.ov), d_oo)
        ).antisymmetrise(0, 1)
    )
    dm.oovv += (
        # N^4: O^2V^2 / N^4: O^2V^2
        + 4.0 * einsum("ib,ja->ijab",
                       einsum("kc,ikbc->ib", u1, td2),
                       u1).antisymmetrise(0, 1).antisymmetrise(2, 3)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2.0 * einsum("ijka,kb->ijab",
                       einsum("kc,ijac->ijka", u1, td2), u1).antisymmetrise(2, 3)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2.0 * einsum("jk,ikab->ijab", einsum("jc,kc->jk", u1, u1), td2)
    )
    dm.ovov += (
        # N^6: O^3V^3 / N^4: O^2V^2
        - 4.0 * einsum("jkac,ikbc->iajb", u2, u2)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 2.0 * einsum("ab,ij->iajb", einsum("klac,klbc->ab", u2, u2), d_oo)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1.0 * einsum("iklb,jkla->iajb", einsum("ld,ikbd->iklb", u1, t2),
                       einsum("lc,jkac->jkla", u1, t2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", einsum("ka,kb->ab", u1, u1), p0.oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1.0 * einsum("ij,ab->iajb", einsum("ic,jc->ij", u1, u1), p0.vv)
        # N^6: O^3V^3 / N^4: O^2V^2
        + 1.0 * einsum("jlac,ilbc->iajb",
                       einsum("kl,jkac->jlac", einsum("kd,ld->kl", u1, u1), t2), t2)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijla,lb->iajb",
                       einsum("ijkl,ka->ijla",
                              einsum("ikcd,jlcd->ijkl", t2, t2), u1), u1)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("jklb,ikla->iajb",
                       einsum("jd,klbd->jklb", u1, t2),
                       einsum("ic,klac->ikla", u1, t2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1.0 * einsum("ib,ja->iajb",
                       einsum("kd,ikbd->ib", u1, t2), einsum("lc,jlac->ja", u1, t2))
        # N^5: O^2V^3 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb",
                       einsum("lmad,lmbd->ab",
                              einsum("km,klad->lmad",
                                     einsum("kc,mc->km", u1, u1), t2), t2), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1.0 * einsum("ab,ij->iajb",
                       einsum("lb,la->ab",
                              einsum("kd,klbd->lb", u1, t2),
                              einsum("mc,lmac->la", u1, t2)), d_oo)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb",
                       einsum("klmb,klma->ab",
                              einsum("md,klbd->klmb", u1, t2),
                              einsum("mc,klac->klma", u1, t2)), d_oo)
        + (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1.0 * einsum("ib,ja->iajb", einsum("ic,bc->ib", u1, p0.vv), u1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1.0 * einsum("ib,ja->iajb", einsum("kb,ik->ib", u1, p0.oo), u1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 2.0 * einsum("ijlb,la->iajb",
                           einsum("jc,ilbc->ijlb",
                                  einsum("kd,jkcd->jc", u1, t2), t2), u1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 2.0 * einsum("kb,ijka->iajb",
                           einsum("ld,klbd->kb", u1, t2),
                           einsum("ic,jkac->ijka", u1, t2))
            # N^6: O^4V^2 / N^4: O^2V^2
            - 2.0 * einsum("ijkb,ka->iajb",
                           einsum("jklc,ilbc->ijkb",
                                  einsum("kd,jlcd->jklc", u1, t2), t2), u1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 2.0 * einsum("ijlb,la->iajb",
                           einsum("ijkc,klbc->ijlb",
                                  einsum("id,jkcd->ijkc", u1, t2), t2), u1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 2.0 * einsum("ikbc,jkac->iajb",
                           einsum("il,klbc->ikbc",
                                  einsum("id,ld->il", u1, u1), t2), t2)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1.0 * einsum("ib,ja->iajb",
                           einsum("kc,ikbc->ib",
                                  einsum("ld,klcd->kc", u1, t2), t2), u1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1.0 * einsum("ab,ij->iajb",
                           einsum("kb,ka->ab",
                                  einsum("kc,bc->kb", u1, p0.vv), u1), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1.0 * einsum("ab,ij->iajb",
                           einsum("mb,ma->ab",
                                  einsum("ld,lmbd->mb",
                                         einsum("kc,klcd->ld", u1, t2),
                                         t2), u1), d_oo)
        ).symmetrise([(0, 2), (1, 3)])
    )
    dm.ovvv += (
        # N^6: O^3V^3 / N^4: O^1V^3
        + 1.0 * einsum("ijka,jkbc->iabc", einsum("id,jkad->ijka", u1, u2), t2)
        # N^5: O^2V^3 / N^4: O^1V^3
        + 2.0 * einsum("ja,ijbc->iabc", einsum("kd,jkad->ja", u1, u2), t2)
        + (
            # N^5: O^2V^3 / N^4: O^1V^3
            + 2.0 * einsum("ac,ib->iabc", einsum("jkad,jkcd->ac", u2, t2), u1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 4.0 * einsum("ikab,kc->iabc", einsum("jkad,ijbd->ikab", u2, t2), u1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 2.0 * einsum("ac,ib->iabc", einsum("ja,jc->ac", u1, p0.ov), u1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 2.0 * einsum("ac,ib->iabc", einsum("ja,jc->ac", u1, u1), p0.ov)
        ).antisymmetrise(2, 3)
    )
    dm.vvvv += (
        # N^6: O^2V^4 / N^4: V^4
        + 2 * einsum("ijab,ijcd->abcd", u2, u2)
        # N^6: O^2V^4 / N^4: V^4
        - 1 * einsum("jkab,jkcd->abcd",
                     einsum("ij,ikab->jkab", einsum("ie,je->ij", u1, u1), t2), t2)
        + (
            # N^4: V^4 / N^4: V^4
            + 4.0 * einsum("ac,bd->abcd", einsum("ia,ic->ac", u1, u1), p0.vv)
            # N^6: O^3V^3 / N^4: V^4
            + 4.0 * einsum("jabc,jd->abcd",
                           einsum("ijbc,ia->jabc",
                                  einsum("jkbe,ikce->ijbc", t2, t2), u1), u1)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + (
            # N^5: O^1V^4 / N^4: V^4
            + 4.0 * einsum("ka,kbcd->abcd",
                           einsum("ie,ikae->ka", u1, t2),
                           einsum("jb,jkcd->kbcd", u1, t2))
            # N^6: O^3V^3 / N^4: V^4
            + 2.0 * einsum("ibcd,ia->abcd",
                           einsum("ijkb,jkcd->ibcd",
                                  einsum("ie,jkbe->ijkb", u1, t2), t2), u1)
        ).antisymmetrise(0, 1).symmetrise([(0, 2), (1, 3)])
    )
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "isr0": diffdm_isr0_2p,
    "isr1s": diffdm_isr1s_2p,
    "isr1": diffdm_isr1_2p,
    "isr2": diffdm_isr2_2p,
}


def state_diffdm_2p(method, ground_state, amplitude, intermediates=None):
    """
    Compute the two-particle difference density matrix of an excited state
    in the MO basis.

    Parameters
    ----------
    method : str, IsrMethod
        The method to use for the computation (e.g. "isr2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, IsrMethod):
        method = IsrMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state_diffdm_2p is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
