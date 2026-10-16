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

from .. import block as b
from ..AdcMethod import IsrMethod
from ..AmplitudeVector import AmplitudeVector
from ..functions import einsum, zeros_like
from ..Intermediates import Intermediates
from ..LazyRoDodsMp import LazyRoDodsMp
from ..NParticleOperator import OperatorSymmetry
from ..TwoParticleDensity import TwoParticleDensity
from .util import check_doubles_amplitudes, check_singles_amplitudes, check_triples_amplitudes


def diffdm_isr0_2p(
    ground_state: LazyRoDodsMp, amplitude: AmplitudeVector, intermediates: Intermediates
) -> TwoParticleDensity:
    check_singles_amplitudes([b.o, b.v], amplitude)

    ur1 = amplitude.ph

    d_oo = zeros_like(ground_state.reference_state.foo)
    d_oo.set_mask("ii", 1)

    dm = TwoParticleDensity(ground_state, symmetry=OperatorSymmetry.HERMITIAN)

    dm.oooo = 4 * (
        + 1 * einsum("il,jk->ijkl", einsum("ia,la->il", ur1, ur1), d_oo)  # N^4: O^4 / N^4: O^4
    ).antisymmetrise(0, 1).antisymmetrise(2, 3)  # fmt: skip
    dm.ovov = (
        - 1 * einsum("ja,ib->iajb", ur1, ur1)  # N^4: O^2V^2 / N^4: O^2V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ab,ij->iajb", einsum("ka,kb->ab", ur1, ur1), d_oo)
    )  # fmt: skip
    return dm


def diffdm_isr1s_2p(
    ground_state: LazyRoDodsMp, amplitude: AmplitudeVector, intermediates: Intermediates
) -> TwoParticleDensity:
    dm = diffdm_isr0_2p(ground_state, amplitude, intermediates)

    ur1 = amplitude.ph

    d_oo = zeros_like(ground_state.reference_state.foo)
    d_oo.set_mask("ii", 1)

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)

    dm.ooov += 2 * (
        # Apply (1 - P_ij) to:
        # N^4: O^3V^1 / N^4: O^3V^1
        + 1 * einsum("jk,ia->ijka", einsum("kb,jb->jk", ur1, t1_1), ur1)
        # N^4: O^3V^1 / N^4: O^3V^1
        + 1 * einsum("jk,ia->ijka", einsum("jb,kb->jk", ur1, ur1), t1_1)
        # N^4: O^3V^1 / N^4: O^3V^1
        + 1 * einsum("ia,jk->ijka", einsum("il,la->ia", einsum("lb,ib->il", ur1, t1_1), ur1), d_oo)
        # N^4: O^3V^1 / N^4: O^3V^1
        + 1 * einsum("ia,jk->ijka", einsum("il,la->ia", einsum("ib,lb->il", ur1, ur1), t1_1), d_oo)
    ).antisymmetrise(0, 1)  # fmt: skip
    dm.oovv += (
        # Apply (1 - P_ij - P_ab + P_ijP_ab) to:
        # N^4: O^2V^2 / N^4: O^2V^2
        + 4 * 1 * einsum("ib,ja->ijab", einsum("kc,ikbc->ib", ur1, t2_1), ur1)
        # Apply (1 - P_ab) to:
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * 1 * einsum("ijka,kb->ijab", einsum("kc,ijac->ijka", ur1, t2_1), ur1)
        # Apply (1 - P_ij) to:
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * 1 * einsum("jk,ikab->ijab", einsum("jc,kc->jk", ur1, ur1), t2_1)
    ).antisymmetrise(0, 1).antisymmetrise(2, 3)  # fmt: skip
    dm.ovvv += 2 * (
        # Apply (1 - P_bc) to:
        # N^4: O^1V^3 / N^4: O^1V^3
        + 1 * einsum("ac,ib->iabc", einsum("ja,jc->ac", ur1, t1_1), ur1)
        # N^4: O^1V^3 / N^4: O^1V^3
        + 1 * einsum("ac,ib->iabc", einsum("ja,jc->ac", ur1, ur1), t1_1)
    ).antisymmetrise(2, 3)  # fmt: skip
    return dm


def diffdm_isr1_2p(
    ground_state: LazyRoDodsMp, amplitude: AmplitudeVector, intermediates: Intermediates
) -> TwoParticleDensity:
    dm = diffdm_isr1s_2p(ground_state, amplitude, intermediates)

    try:
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    except ValueError:
        return dm

    ur1, ur2 = amplitude.ph, amplitude.pphh

    d_oo = zeros_like(ground_state.reference_state.foo)
    d_oo.set_mask("ii", 1)

    dm.ooov += (
        # Apply 1 to:
        - 2 * einsum("kb,ijab->ijka", ur1, ur2)  # N^5: O^3V^2 / N^4: O^2V^2
        # Apply (1 - P_ij) to:
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * 2 * einsum("ja,ik->ijka", einsum("lb,jlab->ja", ur1, ur2), d_oo)
    ).antisymmetrise(0, 1)  # fmt: skip
    dm.ovvv += (
        - 2 * einsum("ja,ijbc->iabc", ur1, ur2)  # N^5: O^2V^3 / N^4: O^1V^3
    )  # fmt: skip
    return dm


def diffdm_isr2d_2p(
    ground_state: LazyRoDodsMp, amplitude: AmplitudeVector, intermediates: Intermediates
) -> TwoParticleDensity:
    dm = diffdm_isr1_2p(ground_state, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)

    ur1, ur2 = amplitude.ph, amplitude.pphh

    d_oo = zeros_like(ground_state.reference_state.foo)
    d_oo.set_mask("ii", 1)

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)
    t1_2 = ground_state.ts2(b.ov)
    t2_2 = ground_state.td2(b.oovv)

    dm.oooo += (
        # Apply (1 - P_ij - P_kl + P_ijP_kl) to:
        4 * (
            # second order ph-ph
            # N^4: O^4 / N^4: O^4
            + 1 * einsum("jl,ik->ijkl",
                         einsum("jb,lb->jl", t1_1, t1_1), einsum("ia,ka->ik", ur1, ur1))
            # N^4: O^4 / N^4: O^4
            + 1 * einsum("jl,ik->ijkl",
                         einsum("lb,jb->jl", ur1, t1_1), einsum("ia,ka->ik", ur1, t1_1))
            # N^6: O^5V^1 / N^4: O^2V^2
            + 1 * einsum("jlmc,ikmc->ijkl",
                         einsum("lb,jmbc->jlmc", ur1, t2_1), einsum("ia,kmac->ikmc", ur1, t2_1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("jl,ik->ijkl",
                           einsum("jmbc,lmbc->jl", t2_1, t2_1), einsum("ia,ka->ik", ur1, ur1))
            # N^4: O^4 / N^4: O^4
            + 1 * einsum("ik,jl->ijkl",
                         einsum("km,im->ik",
                                einsum("mb,kb->km", ur1, t1_1),
                                einsum("ma,ia->im", ur1, t1_1)), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("il,jk->ijkl",
                         einsum("lc,ic->il",
                                einsum("nb,lnbc->lc", ur1, t2_1),
                                einsum("ma,imac->ic", ur1, t2_1)), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ik,jl->ijkl",
                         einsum("kmnc,imnc->ik",
                                einsum("nb,kmbc->kmnc", ur1, t2_1),
                                einsum("na,imac->imnc", ur1, t2_1)), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jl->ijkl",
                           einsum("inbc,knbc->ik",
                                  einsum("mn,imbc->inbc",
                                         einsum("ma,na->mn", ur1, ur1), t2_1), t2_1), d_oo)
            # zeroth order pphh-pphh
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ik,jl->ijkl", einsum("imab,kmab->ik", ur2, ur2), d_oo)
        )
        # Apply (1 - P_ij + P_ikP_jl + P_ilP_jk) to:
        + 4 * (
            # second order ph-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("iklb,jb->ijkl",
                         einsum("ic,klbc->iklb", einsum("ma,imac->ic", ur1, t2_1), t2_1), ur1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("jklm,im->ijkl",
                           einsum("jmbc,klbc->jklm", t2_1, t2_1), einsum("ia,ma->im", ur1, ur1))
            # first order ph-pphh/pphh-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("jkla,ia->ijkl", einsum("jb,klab->jkla", ur1, ur2), t1_1)
        )
        # Apply (1 - P_ij - P_kl + P_ijP_kl + P_ikP_jl + P_ilP_jk - P_ijP_ikP_jl - P_ijP_ilP_jk) to:
        + 8 * (
            # second order ph-ph
            # N^4: O^4 / N^4: O^4
            + 0.5 * einsum("ik,jl->ijkl",
                           einsum("km,im->ik",
                                  einsum("kb,mb->km", t1_1, t1_1),
                                  einsum("ia,ma->im", ur1, ur1)), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jl->ijkl",
                           einsum("kb,ib->ik",
                                  einsum("mc,kmbc->kb",
                                         einsum("na,mnac->mc", ur1, t2_1), t2_1), ur1), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ik,jl->ijkl",
                            einsum("kn,in->ik",
                                   einsum("kmbc,mnbc->kn", t2_1, t2_1),
                                   einsum("ia,na->in", ur1, ur1)), d_oo)
            # first order ph-pphh/pphh-ph
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ik,jl->ijkl",
                         einsum("ib,kb->ik", einsum("ma,imab->ib", ur1, ur2), t1_1), d_oo)
        )
        # Apply 1 to:
        # second order ph-ph
        # N^6: O^5V^1 / N^4: O^2V^2
        - 1 * einsum("klmc,ijmc->ijkl",
                     einsum("mb,klbc->klmc", ur1, t2_1), einsum("ma,ijac->ijmc", ur1, t2_1))
        # zeroth order pphh-pphh
        + 2 * einsum("ijab,klab->ijkl", ur2, ur2)  # N^6: O^4V^2 / N^4: O^2V^2
    ).antisymmetrise(0, 1).antisymmetrise(2, 3).symmetrise((0, 2), (1, 3))  # fmt: skip
    dm.ooov += (
        # Apply (1 - P_ij) to:
        2 * (
            # second order ph-ph
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("jk,ia->ijka", einsum("kb,jb->jk", ur1, t1_2), ur1)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("jk,ia->ijka", einsum("jb,kb->jk", ur1, ur1), t1_2)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ia->ijka",
                         einsum("jc,kc->jk", einsum("lb,jlbc->jc", ur1, t2_1), t1_1), ur1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ik,ja->ijka",
                         einsum("ib,kb->ik", einsum("lc,ilbc->ib", t1_1, t2_1), ur1), ur1)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ia,jk->ijka",
                         einsum("il,la->ia", einsum("lb,ib->il", ur1, t1_2), ur1), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ia,jk->ijka",
                         einsum("il,la->ia", einsum("ib,lb->il", ur1, ur1), t1_2), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jl,ikla->ijka",
                         einsum("jc,lc->jl", ur1, t1_1),
                         einsum("kb,ilab->ikla", ur1, t2_1))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ik,ja->ijka",
                         einsum("ic,kc->ik", ur1, t1_1), einsum("lb,jlab->ja", ur1, t2_1))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ik->ijka",
                         einsum("lb,jlab->ja", t1_1, t2_1), einsum("ic,kc->ik", ur1, ur1))
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ikla,jl->ijka",
                         einsum("kb,ilab->ikla", t1_1, t2_1),
                         einsum("jc,lc->jl", ur1, ur1))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ik->ijka",
                         einsum("jl,la->ja",
                                einsum("jb,lb->jl",
                                       einsum("mc,jmbc->jb", t1_1, t2_1), ur1), ur1), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ia,jk->ijka",
                         einsum("il,la->ia",
                                einsum("ic,lc->il",
                                       einsum("mb,imbc->ic", ur1, t2_1), t1_1), ur1), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ik->ijka",
                         einsum("mb,jmab->ja",
                                einsum("lm,lb->mb",
                                       einsum("lc,mc->lm", ur1, t1_1), ur1), t2_1), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ik->ijka",
                         einsum("jl,la->ja",
                                einsum("jc,lc->jl", ur1, t1_1),
                                einsum("mb,lmab->la", ur1, t2_1)), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ia,jk->ijka",
                         einsum("ma,im->ia",
                                einsum("lb,lmab->ma", t1_1, t2_1),
                                einsum("ic,mc->im", ur1, ur1)), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ik->ijka",
                         einsum("mb,jmab->ja",
                                einsum("lm,lb->mb",
                                       einsum("lc,mc->lm", ur1, ur1), t1_1), t2_1), d_oo)
            # first order ph-pphh/pphh-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ia->ijka", einsum("klbc,jlbc->jk", ur2, t2_1), ur1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 2 * einsum("jklb,ilab->ijka", einsum("jc,klbc->jklb", ur1, ur2), t2_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ia,jk->ijka",
                         einsum("il,la->ia", einsum("lmbc,imbc->il", ur2, t2_1), ur1), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ik->ijka",
                         einsum("jlmb,lmab->ja", einsum("jc,lmbc->jlmb", ur1, ur2), t2_1), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ia,jk->ijka",
                         einsum("lb,ilab->ia", einsum("mc,lmbc->lb", ur1, ur2), t2_1), d_oo)
        )
        # Apply 1 to:
        # second order ph-ph
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ijkl,la->ijka",
                     einsum("ijlc,kc->ijkl", einsum("lb,ijbc->ijlc", ur1, t2_1), t1_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("kb,ijab->ijka",
                     einsum("kl,lb->kb", einsum("lc,kc->kl", ur1, t1_1), ur1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("kb,ijab->ijka",
                     einsum("kl,lb->kb", einsum("kc,lc->kl", ur1, ur1), t1_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijkl,la->ijka",
                     einsum("ijkc,lc->ijkl", einsum("kb,ijbc->ijkc", ur1, t2_1), t1_1), ur1)
        # first order ph-pphh/pphh-ph
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1 * einsum("ijkl,la->ijka", einsum("klbc,ijbc->ijkl", ur2, t2_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("kb,ijab->ijka", einsum("lc,klbc->kb", ur1, ur2), t2_1)
    ).antisymmetrise(0, 1)  # fmt: skip
    dm.oovv += (
        # Apply (1 - P_ij - P_ab + P_ijP_ab) to:
        4 * (
            # second order ph-ph
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ib,ja->ijab", einsum("kc,ikbc->ib", ur1, t2_2), ur1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ib->ijab", einsum("kc,kc->", ur1, t1_1) * ur1, t1_1)
            # first order ph-pphh/pphh-ph
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ia,jb->ijab", einsum("kc,ikac->ia", ur1, ur2), t1_1)
        )
        # Apply (1 - P_ab) to:
        + 2 * (
            # second order ph-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijka,kb->ijab", einsum("kc,ijac->ijka", ur1, t2_2), ur1)
            # first order ph-pphh/pphh-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkb,ka->ijab", einsum("kc,ijbc->ijkb", ur1, ur2), t1_1)
        )
        # Apply (1 - P_ij) to:
        + 2 * (
            # second order ph-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ikab->ijab", einsum("jc,kc->jk", ur1, ur1), t2_2)
            # first order ph-pphh/pphh-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ik,jkab->ijab", einsum("kc,ic->ik", ur1, t1_1), ur2)
        )
    ).antisymmetrise(0, 1).antisymmetrise(2, 3)  # fmt: skip
    dm.ovov += (
        # Apply (1 + P_ijP_ab) to:
        2 * (
            # second order ph-ph
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ib->iajb",
                         einsum("jk,ka->ja", einsum("kc,jc->jk", ur1, t1_1), ur1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijlb,la->iajb",
                         einsum("jc,ilbc->ijlb", einsum("kd,jkcd->jc", ur1, t2_1), t2_1), ur1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ib,ja->iajb",
                         einsum("ik,kb->ib", einsum("ic,kc->ik", ur1, ur1), t1_1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("kb,ijka->iajb",
                         einsum("ld,klbd->kb", ur1, t2_1), einsum("ic,jkac->ijka", ur1, t2_1))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ib,ja->iajb",
                           einsum("ik,kb->ib", einsum("ic,kc->ik", t1_1, t1_1), ur1), ur1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ib,ja->iajb",
                           einsum("ik,kb->ib", einsum("ic,kc->ik", ur1, t1_1), t1_1), ur1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ij,ab->iajb",
                         einsum("ic,jc->ij", ur1, t1_1), einsum("ka,kb->ab", ur1, t1_1))
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijkb,ka->iajb",
                         einsum("jklc,ilbc->ijkb", einsum("kd,jlcd->jklc", ur1, t2_1), t2_1), ur1)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 1 * einsum("ijlb,la->iajb",
                         einsum("ijkc,klbc->ijlb", einsum("id,jkcd->ijkc", ur1, t2_1), t2_1), ur1)
            # N^6: O^3V^3 / N^4: O^2V^2
            - 1 * einsum("ikbc,jkac->iajb",
                         einsum("il,klbc->ikbc", einsum("id,ld->il", ur1, ur1), t2_1), t2_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ib,ja->iajb",
                           einsum("kc,ikbc->ib", einsum("ld,klcd->kc", ur1, t2_1), t2_1), ur1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ib,ja->iajb",
                            einsum("il,lb->ib", einsum("ikcd,klcd->il", t2_1, t2_1), ur1), ur1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ib,ja->iajb",
                            einsum("iklc,klbc->ib", einsum("id,klcd->iklc", ur1, t2_1), t2_1), ur1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("ka,kb->ab",
                                  einsum("kl,la->ka",
                                         einsum("lc,kc->kl", ur1, t1_1), ur1), t1_1), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb",
                           einsum("mb,ma->ab",
                                  einsum("ld,lmbd->mb",
                                         einsum("kc,klcd->ld", ur1, t2_1), t2_1), ur1), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ab,ij->iajb",
                            einsum("mb,ma->ab",
                                   einsum("klmd,klbd->mb",
                                          einsum("mc,klcd->klmd", ur1, t2_1), t2_1), ur1), d_oo)
            # first order ph-pphh/pphh-ph
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijkb,ka->iajb", einsum("ikbc,jc->ijkb", ur2, t1_1), ur1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ijka,kb->iajb", einsum("ic,jkac->ijka", ur1, ur2), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * einsum("ja,ib->iajb", einsum("kc,jkac->ja", ur1, ur2), t1_1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ab,ij->iajb",
                         einsum("ka,kb->ab", einsum("lc,klac->ka", ur1, ur2), t1_1), d_oo)
        )
        # Apply 1 to:
        # second order ph-ph
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1 * einsum("iklb,jkla->iajb",
                     einsum("ld,ikbd->iklb", ur1, t2_1), einsum("lc,jkac->jkla", ur1, t2_1))
        # N^6: O^3V^3 / N^4: O^2V^2
        + 1 * einsum("jlac,ilbc->iajb",
                     einsum("kl,jkac->jlac", einsum("kd,ld->kl", ur1, ur1), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijla,lb->iajb",
                       einsum("ijkl,ka->ijla", einsum("ikcd,jlcd->ijkl", t2_1, t2_1), ur1), ur1)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("jklb,ikla->iajb",
                       einsum("jd,klbd->jklb", ur1, t2_1), einsum("ic,klac->ikla", ur1, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ab->iajb",
                     einsum("ic,jc->ij", t1_1, t1_1), einsum("ka,kb->ab", ur1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ab,ij->iajb", einsum("ka,kb->ab", t1_1, t1_1), einsum("ic,jc->ij", ur1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ib,ja->iajb",
                     einsum("kd,ikbd->ib", ur1, t2_1), einsum("lc,jlac->ja", ur1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ij,ab->iajb",
                       einsum("ilcd,jlcd->ij", t2_1, t2_1), einsum("ka,kb->ab", ur1, ur1))
        # N^5: O^2V^3 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb",
                       einsum("klad,klbd->ab", t2_1, t2_1), einsum("ic,jc->ij", ur1, ur1))
        # N^5: O^2V^3 / N^4: O^2V^2
        + 1 * einsum("ab,ij->iajb",
                     einsum("lmad,lmbd->ab",
                            einsum("km,klad->lmad",
                                   einsum("kc,mc->km", ur1, ur1), t2_1), t2_1), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ab,ij->iajb",
                     einsum("la,lb->ab",
                            einsum("kl,ka->la", einsum("kc,lc->kl", ur1, ur1), t1_1), t1_1), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ab,ij->iajb",
                     einsum("lb,la->ab",
                            einsum("kd,klbd->lb", ur1, t2_1),
                            einsum("mc,lmac->la", ur1, t2_1)), d_oo)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb",
                       einsum("klmb,klma->ab",
                              einsum("md,klbd->klmb", ur1, t2_1),
                              einsum("mc,klac->klma", ur1, t2_1)), d_oo)
        # zeroth order pphh-pphh
        - 4 * einsum("jkac,ikbc->iajb", ur2, ur2)  # N^6: O^3V^3 / N^4: O^2V^2
        # N^5: O^2V^3 / N^4: O^2V^2
        + 2 * einsum("ab,ij->iajb", einsum("klac,klbc->ab", ur2, ur2), d_oo)
    ).symmetrise((0, 2), (1, 3))  # fmt: skip
    dm.ovvv += (
        # Apply (1 - P_bc) to:
        2 * (
            # second order ph-ph
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ac,ib->iabc", einsum("ja,jc->ac", ur1, t1_2), ur1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ac,ib->iabc", einsum("ja,jc->ac", ur1, ur1), t1_2)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ab,ic->iabc",
                         einsum("jb,ja->ab", einsum("kd,jkbd->jb", t1_1, t2_1), ur1), ur1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("ikab,kc->iabc",
                         einsum("ijkb,ja->ikab", einsum("kd,ijbd->ijkb", t1_1, t2_1), ur1), ur1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ic,ab->iabc",
                         einsum("jd,ijcd->ic", t1_1, t2_1), einsum("ka,kb->ab", ur1, ur1))
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ac,ib->iabc",
                         einsum("kc,ka->ac", einsum("jd,jkcd->kc", ur1, t2_1), t1_1), ur1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ic,ab->iabc",
                         einsum("jd,ijcd->ic", ur1, t2_1), einsum("kb,ka->ab", ur1, t1_1))
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("ijbc,ja->iabc",
                         einsum("ijkb,kc->ijbc", einsum("kd,ijbd->ijkb", ur1, t2_1), ur1), t1_1)
            # first order ph-pphh/pphh-ph
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("ac,ib->iabc", einsum("jkad,jkcd->ac", ur2, t2_1), ur1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 2 * einsum("ikab,kc->iabc", einsum("jkad,ijbd->ikab", ur2, t2_1), ur1)
        )
        # Apply 1 to:
        # second order ph-ph
        # N^5: O^2V^3 / N^4: O^1V^3
        + 1 * einsum("ja,ijbc->iabc",
                     einsum("jk,ka->ja", einsum("kd,jd->jk", ur1, t1_1), ur1), t2_1)
        # N^5: O^2V^3 / N^4: O^1V^3
        + 1 * einsum("ikbc,ka->iabc",
                     einsum("ij,jkbc->ikbc", einsum("id,jd->ij", ur1, ur1), t2_1), t1_1)
        # N^5: O^2V^3 / N^4: O^1V^3
        + 1 * einsum("ja,ijbc->iabc",
                     einsum("jk,ka->ja", einsum("jd,kd->jk", ur1, ur1), t1_1), t2_1)
        # N^5: O^2V^3 / N^4: O^1V^3
        - 1 * einsum("ijbc,ja->iabc",
                     einsum("ik,jkbc->ijbc", einsum("id,kd->ik", ur1, t1_1), t2_1), ur1)
        # first order ph-pphh/pphh-ph
        # N^6: O^3V^3 / N^4: O^1V^3
        + 1 * einsum("ijka,jkbc->iabc",
                     einsum("id,jkad->ijka", ur1, ur2), t2_1)
        # N^5: O^2V^3 / N^4: O^1V^3
        + 2 * einsum("ja,ijbc->iabc", einsum("kd,jkad->ja", ur1, ur2), t2_1)
    ).antisymmetrise(2, 3)  # fmt: skip
    dm.vvvv += (
        # Apply (1 - P_ab - P_cd + P_abP_cd) to:
        4 * (
            # second order ph-ph
            # N^4: V^4 / N^4: V^4
            + 1 * einsum("bd,ac->abcd",
                         einsum("jb,jd->bd", t1_1, t1_1), einsum("ia,ic->ac", ur1, ur1))
            # N^4: V^4 / N^4: V^4
            + 1 * einsum("bd,ac->abcd",
                         einsum("jd,jb->bd", ur1, t1_1), einsum("ia,ic->ac", ur1, t1_1))
            # N^6: O^3V^3 / N^4: V^4
            + 1 * einsum("jabc,jd->abcd",
                         einsum("ijbc,ia->jabc", einsum("jkbe,ikce->ijbc", t2_1, t2_1), ur1), ur1)
            # N^5: O^2V^3 / N^4: V^4
            + 0.5 * einsum("bd,ac->abcd",
                           einsum("jkbe,jkde->bd", t2_1, t2_1), einsum("ia,ic->ac", ur1, ur1))
        )
        # Apply (1 - P_ab + P_acP_bd + P_adP_bc) to:
        + 4 * (
            # second order ph-ph
            # N^5: O^1V^4 / N^4: V^4
            + 1 * einsum("ka,kbcd->abcd",
                         einsum("ie,ikae->ka", ur1, t2_1), einsum("jb,jkcd->kbcd", ur1, t2_1))
            # N^6: O^3V^3 / N^4: V^4
            + 0.5 * einsum("ibcd,ia->abcd",
                           einsum("ijkb,jkcd->ibcd", einsum("ie,jkbe->ijkb", ur1, t2_1), t2_1), ur1)
            # first order ph-pphh/pphh-ph
            # N^5: O^1V^4 / N^4: V^4
            - 2 * einsum("ibcd,ia->abcd", einsum("jb,ijcd->ibcd", ur1, ur2), t1_1)
        )
        # Apply 1 to:
        # N^6: O^2V^4 / N^4: V^4
        # second order ph-ph
        - 1 * einsum("jkab,jkcd->abcd",
                     einsum("ij,ikab->jkab", einsum("ie,je->ij", ur1, ur1), t2_1), t2_1)
        # zeroth order pphh-pphh
        + 2 * einsum("ijab,ijcd->abcd", ur2, ur2)  # N^6: O^2V^4 / N^4: V^4
    ).antisymmetrise(0, 1).antisymmetrise(2, 3).symmetrise((0, 2), (1, 3))  # fmt: skip
    return dm


def diffdm_isr2_2p(
    ground_state: LazyRoDodsMp, amplitude: AmplitudeVector, intermediates: Intermediates
) -> TwoParticleDensity:
    dm = diffdm_isr2d_2p(ground_state, amplitude, intermediates)

    try:
        check_triples_amplitudes([b.o, b.o, b.o, b.v, b.v, b.v], amplitude)
    except ValueError:
        return dm

    ur1, ur3 = amplitude.ph, amplitude.ppphhh

    # zeroth order ph-ppphhh/ppphhh-ph
    dm.oovv += -6 * einsum("kc,ijkabc->ijab", ur1, ur3)  # N^6: O^3V^3 / N^6: O^3V^3
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "isr0": diffdm_isr0_2p,
    "isr1s": diffdm_isr1s_2p,
    "isr1": diffdm_isr1_2p,
    "isr2d": diffdm_isr2d_2p,
    "isr2": diffdm_isr2_2p,
}


def rohf_dods_state_diffdm_2p(
    method: str | IsrMethod,
    ground_state: LazyRoDodsMp,
    amplitude: AmplitudeVector,
    intermediates: Intermediates | None = None,
) -> TwoParticleDensity:
    """
    Compute the two-particle difference density matrix of an excited state
    in the MO basis.

    Parameters
    ----------
    method : str, IsrMethod
        The method to use for the computation (e.g. "isr2")
    ground_state : LazyRoDodsMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, IsrMethod):
        method = IsrMethod(method)
    if not isinstance(ground_state, LazyRoDodsMp):
        raise TypeError("ground_state should be a LazyRoDodsMp object.")
    if not isinstance(amplitude, AmplitudeVector):
        raise TypeError("amplitude should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError(f"state_diffdm_2p is not implemented for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
