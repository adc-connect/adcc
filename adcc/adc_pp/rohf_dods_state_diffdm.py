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
from ..functions import einsum
from ..Intermediates import Intermediates
from ..LazyRoDodsMp import LazyRoDodsMp
from ..NParticleOperator import OperatorSymmetry
from ..OneParticleDensity import OneParticleDensity
from .util import check_doubles_amplitudes, check_singles_amplitudes, check_triples_amplitudes


def diffdm_isr0(ground_state, amplitude, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude)
    ur1 = amplitude.ph

    dm = OneParticleDensity(ground_state, symmetry=OperatorSymmetry.HERMITIAN)
    dm.oo = (
        - 1 * einsum("ia,ja->ij", ur1, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
    )  # fmt: skip
    dm.vv = (
        + 1 * einsum("ia,ib->ab", ur1, ur1)  # N^3: O^1V^2 / N^2: V^2
    )  # fmt: skip
    return dm


def diffdm_isr1s(ground_state, amplitude, intermediates):
    dm = diffdm_isr0(ground_state, amplitude, intermediates)

    ur1 = amplitude.ph
    t1_1 = ground_state.ts1(b.ov)
    # first order ph-ph contribution
    dm.ov += (
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ur1, t1_1), ur1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, ur1), t1_1)  # N^3: O^2V^1 / N^2: O^1V^1
    )  # fmt: skip
    return dm


def diffdm_isr1(ground_state, amplitude, intermediates):
    dm = diffdm_isr1s(ground_state, amplitude, intermediates)
    try:
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    except ValueError:
        return dm
    # zeroth order ph-pphh (and pphh-ph) contribution
    ur1, ur2 = amplitude.ph, amplitude.pphh
    dm.ov += (
        - 2 * einsum("jb,ijab->ia", ur1, ur2)  # N^4: O^2V^2 / N^4: O^2V^2
    )  # fmt: skip
    return dm


def diffdm_isr2(ground_state, amplitude, intermediates):
    dm = diffdm_isr1(ground_state, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)

    ur1, ur2 = amplitude.ph, amplitude.pphh

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)
    t1_2 = ground_state.ts2(b.ov)

    dm.oo += (
        # Apply 1 to:
        # second order ph-ph
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,ik->ij", einsum("kb,jb->jk", ur1, t1_1), einsum("ka,ia->ik", ur1, t1_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jklc,iklc->ij",
                     einsum("lb,jkbc->jklc", ur1, t2_1), einsum("la,ikac->iklc", ur1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ilbc,jlbc->ij",
                       einsum("kl,ikbc->ilbc", einsum("ka,la->kl", ur1, ur1), t2_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jc,ic->ij",
                     einsum("lb,jlbc->jc", ur1, t2_1), einsum("ka,ikac->ic", ur1, t2_1))
        # zeroth order pphh-pphh
        - 2 * einsum("ikab,jkab->ij", ur2, ur2)  # N^5: O^3V^2 / N^4: O^2V^2
        # Apply (1 + P_ij) to:
        + 2 * (
            # second order ph-ph
            # N^3: O^2V^1 / N^2: O^1V^1
            + 0.5 * einsum("jk,ik->ij",
                           einsum("jb,kb->jk", t1_1, t1_1), einsum("ia,ka->ik", ur1, ur1))
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("jb,ib->ij",
                           einsum("kc,jkbc->jb", einsum("la,klac->kc", ur1, t2_1), t2_1), ur1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("jl,il->ij",
                            einsum("jkbc,klbc->jl", t2_1, t2_1), einsum("ia,la->il", ur1, ur1))
            # first order ph-pphh/pphh-ph
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ur1, ur2), t1_1)
        )
    ).symmetrise()  # fmt: skip
    dm.ov += (
        # second order ph-ph
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ur1, t1_2), ur1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, ur1), t1_2)  # N^3: O^2V^1 / N^2: O^1V^1
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("jc,ijbc->ib", t1_1, t2_1), ur1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ia", einsum("jk,kb->jb", einsum("kc,jc->jk", ur1, t1_1), ur1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ia", einsum("kb,jkab->ja", t1_1, t2_1), einsum("ic,jc->ij", ur1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ia", einsum("jk,kb->jb", einsum("jc,kc->jk", ur1, ur1), t1_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ia", einsum("ic,kc->ik", einsum("jb,ijbc->ic", ur1, t2_1), t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ia", einsum("ic,kc->ik", ur1, t1_1), einsum("jb,jkab->ka", ur1, t2_1))
        # first order ph-pphh/pphh-ph
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("jkbc,ijbc->ik", ur2, t2_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ijkb,jkab->ia", einsum("ic,jkbc->ijkb", ur1, ur2), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("kb,ikab->ia", einsum("jc,jkbc->kb", ur1, ur2), t2_1)
    )  # fmt: skip
    dm.vv += (
        # Apply 1 to:
        # second order ph-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kb,ka->ab",
                     einsum("jd,jkbd->kb", ur1, t2_1), einsum("ic,ikac->ka", ur1, t2_1))
        # N^3: O^1V^2 / N^2: V^2
        - 1 * einsum("ja,jb->ab", einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, ur1), t1_1), t1_1)
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum("jkad,jkbd->ab",
                     einsum("ij,ikad->jkad", einsum("ic,jc->ij", ur1, ur1), t2_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ijkb,ijka->ab",
                       einsum("id,jkbd->ijkb", ur1, t2_1), einsum("ic,jkac->ijka", ur1, t2_1))
        # zeroth order pphh-pphh
        + 2 * einsum("ijac,ijbc->ab", ur2, ur2)  # N^5: O^2V^3 / N^4: O^2V^2
        # Apply (1 + P_ab) to:
        + 2 * (
            # second order ph-ph
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ib,ia->ab",
                           einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_1), t2_1), ur1)
            # N^3: O^1V^2 / N^2: V^2
            - 0.5 * einsum("ja,jb->ab",
                           einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, t1_1), ur1), t1_1)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 0.25 * einsum("ib,ia->ab",
                            einsum("ijkd,jkbd->ib", einsum("ic,jkcd->ijkd", ur1, t2_1), t2_1), ur1)
            # first order ph-pphh/pphh-ph
            # N^4: O^2V^2 / N^4: O^2V^2
            + 2 * einsum("ja,jb->ab", einsum("ic,ijac->ja", ur1, ur2), t1_1)
        )
    ).symmetrise()  # fmt: skip
    return dm


def diffdm_isr3d(ground_state, amplitude, intermediates):
    dm = diffdm_isr2(ground_state, amplitude, intermediates)

    ur1, ur2 = amplitude.ph, amplitude.pphh

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)
    t1_2 = ground_state.ts2(b.ov)
    t2_2 = ground_state.td2(b.oovv)
    t3_2 = ground_state.tt2(b.ooovvv)
    t1_3 = ground_state.ts3(b.ov)

    dm.oo += 2 * (
        # Apply (1 + P_ij) to:
        # third order ph-ph
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,ik->ij", einsum("kb,jb->jk", ur1, t1_2), einsum("ka,ia->ik", ur1, t1_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jklc,iklc->ij",
                     einsum("lb,jkbc->jklc", ur1, t2_2), einsum("la,ikac->iklc", ur1, t2_1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("jk,ik->ij",
                       einsum("jb,kb->jk", t1_1, t1_2), einsum("ia,ka->ik", ur1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("jk,ik->ij",
                       einsum("kb,jb->jk", t1_1, t1_2), einsum("ia,ka->ik", ur1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ilbc,jlbc->ij",
                       einsum("kl,ikbc->ilbc", einsum("ka,la->kl", ur1, ur1), t2_1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ib->ij",
                       einsum("kc,jkbc->jb", einsum("la,klac->kc", ur1, t2_1), t2_2), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ib->ij",
                       einsum("kc,jkbc->jb", einsum("la,klac->kc", ur1, t2_2), t2_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jc,ic->ij",
                     einsum("lb,jlbc->jc", ur1, t2_2), einsum("ka,ikac->ic", ur1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("jl,il->ij",
                        einsum("jkbc,klbc->jl", t2_1, t2_2), einsum("ia,la->il", ur1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("jl,il->ij",
                        einsum("klbc,jkbc->jl", t2_1, t2_2), einsum("ia,la->il", ur1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ka,ka->", ur1, t1_1) * einsum("jc,ic->ij",
                                                    einsum("lb,jlbc->jc", ur1, t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("lc,lc->",
                       einsum("kb,klbc->lc", ur1, t2_1), t1_1) * einsum("ia,ja->ij", ur1, t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ka,ka->", ur1, t1_1) * einsum("jb,ib->ij",
                                                      einsum("lc,jlbc->jb", t1_1, t2_1), ur1)
        # second order ph-pphh/pphh-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ur1, ur2), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jk,ik->ij",
                     einsum("klbc,jlbc->jk", ur2, t2_1), einsum("ka,ia->ik", ur1, t1_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ikbc,jkbc->ij",
                     einsum("kl,ilbc->ikbc", einsum("la,ka->kl", ur1, t1_1), ur2), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 2 * einsum("jklb,iklb->ij",
                     einsum("kc,jlbc->jklb", t1_1, t2_1), einsum("ka,ilab->iklb", ur1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jc,ic->ij",
                     einsum("lb,jlbc->jc", einsum("ka,klab->lb", ur1, ur2), t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,ib->ij",
                     einsum("kc,jkbc->jb", t1_1, t2_1), einsum("la,ilab->ib", ur1, ur2))
    ).symmetrise()  # fmt: skip
    dm.ov += (
        # third order ph-ph
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ur1, t1_3), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, ur1), t1_3)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ia",
                     einsum("kc,kc->", t1_1, t1_1) * einsum("jb,ib->ij", ur1, t1_1), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ia",
                     einsum("kc,kc->", t1_1, t1_1) * einsum("ib,jb->ij", ur1, ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("jc,ijbc->ib", t1_1, t2_2), ur1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("ijbc,jc->ib", t2_1, t1_2), ur1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ia", einsum("jk,kb->jb", einsum("kc,jc->jk", ur1, t1_1), ur1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ia", einsum("jk,kb->jb", einsum("kc,jc->jk", ur1, t1_2), ur1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ia", einsum("kb,jkab->ja", t1_1, t2_2), einsum("ic,jc->ij", ur1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ia", einsum("jkab,kb->ja", t2_1, t1_2), einsum("ic,jc->ij", ur1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ia", einsum("jk,kb->jb", einsum("jc,kc->jk", ur1, ur1), t1_1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ia", einsum("jk,kb->jb", einsum("jc,kc->jk", ur1, ur1), t1_2), t2_1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("il,la->ia",
                       einsum("ikcd,klcd->il", einsum("jb,ijkbcd->ikcd", ur1, t3_2), t2_1), ur1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("iklc,klac->ia",
                       einsum("id,klcd->iklc", ur1, t2_1), einsum("jb,jklabc->klac", ur1, t3_2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ja->ia", einsum("ic,jc->ij", einsum("kb,ikbc->ic", ur1, t2_1), t1_2), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ia", einsum("ic,kc->ik", einsum("jb,ijbc->ic", ur1, t2_2), t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ka,ik->ia", einsum("jc,jkac->ka", ur1, t2_1), einsum("ib,kb->ik", ur1, t1_2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ia", einsum("ic,kc->ik", ur1, t1_1), einsum("jb,jkab->ka", ur1, t2_2))
        # N^6: O^3V^3 / N^6: O^3V^3
        - 1 * einsum("kc,ikac->ia",
                     einsum("ld,klcd->kc", ur1, t2_1), einsum("jb,ijkabc->ikac", ur1, t3_2))
        # N^6: O^3V^3 / N^6: O^3V^3
        - 0.5 * einsum("jkbc,ijkabc->ia",
                       einsum("jklc,lb->jkbc", einsum("ld,jkcd->jklc", ur1, t2_1), ur1), t3_2)
        # N^6: O^3V^3 / N^6: O^3V^3
        - 0.5 * einsum("jkbc,ijkabc->ia",
                       einsum("jl,klbc->jkbc", einsum("jd,ld->jl", ur1, ur1), t2_1), t3_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.25 * einsum("ij,ja->ia",
                        einsum("klcd,klcd->", t2_1, t2_1) * einsum("jb,ib->ij", ur1, t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.25 * einsum("ij,ja->ia",
                        einsum("klcd,klcd->", t2_1, t2_1) * einsum("ib,jb->ij", ur1, ur1), t1_1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.25 * einsum("ik,ka->ia",
                        einsum("ib,kb->ik", einsum("jlcd,ijlbcd->ib", t2_1, t3_2), ur1), ur1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.25 * einsum("ja,ij->ia",
                        einsum("klbd,jklabd->ja", t2_1, t3_2), einsum("ic,jc->ij", ur1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ia",
                     einsum("kb,kb->", ur1, t1_1) * einsum("ic,jc->ij", t1_1, t1_1), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ik,ka->ia",
                     einsum("jc,jc->", ur1, t1_1) * einsum("ib,kb->ik", ur1, t1_1), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,jk->",
                     einsum("jc,kc->jk", t1_1, t1_1), einsum("jb,kb->jk", ur1, ur1)) * t1_1
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,jk->",
                     einsum("jc,kc->jk", ur1, t1_1), einsum("jb,kb->jk", ur1, t1_1)) * t1_1
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("jc,ic->ij",
                              einsum("ld,jlcd->jc",
                                     einsum("kb,klbd->ld", ur1, t2_1), t2_1), t1_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("jl,il->ij",
                              einsum("jkcd,klcd->jl", t2_1, t2_1),
                              einsum("lb,ib->il", ur1, t1_1)), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("lb,lb->", ur1, t1_1) * einsum("ikcd,jkcd->ij", t2_1, t2_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ik,ka->ia",
                       einsum("kb,ib->ik",
                              einsum("ld,klbd->kb",
                                     einsum("jc,jlcd->ld", ur1, t2_1), t2_1), ur1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("id,jd->ij",
                              einsum("iklc,klcd->id",
                                     einsum("ib,klbc->iklc", ur1, t2_1), t2_1), ur1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("jd,jd->", ur1, t1_1) * einsum("iklc,klac->ia",
                                                      einsum("ib,klbc->iklc", ur1, t2_1), t2_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia",
                     einsum("jk,ik->ij",
                            einsum("jc,kc->jk", t1_1, t1_1), einsum("kb,ib->ik", ur1, t1_1)), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia",
                     einsum("jk,ik->ij",
                            einsum("jc,kc->jk", ur1, t1_1), einsum("ib,kb->ik", ur1, t1_1)), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ld,ld->",
                     einsum("kc,klcd->ld", einsum("jb,jkbc->kc", ur1, t2_1), t2_1), ur1) * t1_1
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kc,kc->",
                     einsum("ld,klcd->kc", ur1, t2_1), t1_1) * einsum("jb,ijab->ia", ur1, t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ld,ld->", ur1, t1_1) * einsum("kc,ikac->ia",
                                                    einsum("jb,jkbc->kc", ur1, t2_1), t2_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ia",
                       einsum("ik,jk->ij",
                              einsum("ic,kc->ik", t1_1, t1_1), einsum("jb,kb->jk", ur1, t1_1)), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("il,la->ia",
                       einsum("lc,ic->il", ur1, t1_1),
                       einsum("kd,klad->la", einsum("jb,jkbd->kd", ur1, t2_1), t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ik,ka->ia",
                       einsum("id,kd->ik",
                              einsum("lc,ilcd->id",
                                     einsum("jb,jlbc->lc", ur1, t2_1), t2_1), ur1), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ia",
                       einsum("jk,ik->ij",
                              einsum("kc,jc->jk", ur1, t1_1), einsum("kb,ib->ik", ur1, t1_1)), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ld,ld->",
                       einsum("jklc,jkcd->ld",
                              einsum("lb,jkbc->jklc", ur1, t2_1), t2_1), ur1) * t1_1
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ik,ka->ia",
                       einsum("jk,ij->ik",
                              einsum("jb,kb->jk", t1_1, t1_1), einsum("ic,jc->ij", ur1, ur1)), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ia",
                       einsum("ik,jk->ij",
                              einsum("ib,kb->ik", t1_1, t1_1), einsum("jc,kc->jk", ur1, ur1)), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("jl,jl->",
                       einsum("jkbc,klbc->jl", t2_1, t2_1),
                       einsum("jd,ld->jl", ur1, ur1)) * t1_1
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ij,ja->ia",
                        einsum("jc,ic->ij",
                               einsum("jkld,klcd->jc",
                                      einsum("jb,klbd->jkld", ur1, t2_1), t2_1), t1_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ik,ka->ia",
                        einsum("jk,ij->ik",
                               einsum("jlcd,klcd->jk", t2_1, t2_1),
                               einsum("ib,jb->ij", ur1, ur1)), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("il,la->ia",
                        einsum("lc,ic->il", ur1, t1_1),
                        einsum("jkld,jkad->la", einsum("lb,jkbd->jkld", ur1, t2_1), t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ik,ka->ia",
                        einsum("ij,jk->ik",
                               einsum("ilbc,jlbc->ij", t2_1, t2_1),
                               einsum("jd,kd->jk", ur1, ur1)), t1_1)
        # second order ph-pphh/pphh-ph
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("jkbc,ijbc->ik", ur2, t2_2), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ijkb,jkab->ia", einsum("ic,jkbc->ijkb", ur1, ur2), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("kb,ikab->ia", einsum("jc,jkbc->kb", ur1, ur2), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ij,ja->ia", einsum("ic,jc->ij", t1_1, t1_1), einsum("kb,jkab->ja", ur1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("jc,ijbc->ib", ur1, ur2), t1_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("il,la->ia",
                       einsum("ikcd,klcd->il", t2_1, t2_1), einsum("jb,jlab->la", ur1, ur2))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("kc,ikac->ia",
                       einsum("jk,jc->kc", einsum("jlbd,klbd->jk", ur2, t2_1), ur1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijkb,jkab->ia",
                       einsum("id,jkbd->ijkb", einsum("lc,ilcd->id", ur1, t2_1), t2_1), ur2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("il,la->ia",
                       einsum("ikbc,klbc->il", ur2, t2_1), einsum("jd,jlad->la", ur1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("iklc,klac->ia",
                       einsum("ib,klbc->iklc", einsum("jd,ijbd->ib", ur1, ur2), t2_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("lc,ilac->ia",
                       einsum("jklb,jkbc->lc", einsum("ld,jkbd->jklb", ur1, ur2), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijkc,jkac->ia",
                     einsum("ikld,jlcd->ijkc", einsum("kb,ilbd->ikld", ur1, t2_1), t2_1), ur2)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kc,ikac->ia",
                     einsum("lb,klbc->kc", einsum("jd,jlbd->lb", ur1, ur2), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("iklc,klac->ia",
                     einsum("ijkb,jlbc->iklc", einsum("kd,ijbd->ijkb", ur1, ur2), t2_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ik,ka->ia",
                     einsum("kc,ic->ik", einsum("jkbc,jb->kc", ur2, t1_1), t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ik,ka->ia", einsum("kc,ic->ik", einsum("jkbc,jb->kc", ur2, t1_1), ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,jb->", einsum("kc,jkbc->jb", ur1, ur2), t1_1) * t1_1
        # N^6: O^4V^2 / N^4: O^2V^2
        - 0.25 * einsum("ijkl,jkla->ia",
                        einsum("ilcd,jkcd->ijkl", t2_1, t2_1), einsum("lb,jkab->jkla", ur1, ur2))
        # N^6: O^4V^2 / N^4: O^2V^2
        - 0.25 * einsum("ijkl,jkla->ia",
                        einsum("ijbc,klbc->ijkl", ur2, t2_1), einsum("jd,klad->jkla", ur1, t2_1))
        # first order pphh-pphh
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("ijkb,jkab->ia", einsum("jkbc,ic->ijkb", ur2, t1_1), ur2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("ik,ka->ia", einsum("ijbc,jkbc->ik", ur2, ur2), t1_1)
    )  # fmt: skip
    dm.vv += 2 * (
        # Apply (1 + P_ab) to:
        # third order ph-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kb,ka->ab",
                     einsum("jd,jkbd->kb", ur1, t2_2), einsum("ic,ikac->ka", ur1, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ib,ia->ab",
                       einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_2), t2_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ib,ia->ab",
                       einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_1), t2_2), ur1)
        # N^3: O^1V^2 / N^2: V^2
        - 1 * einsum("ja,jb->ab",
                     einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, ur1), t1_1), t1_2)
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum("jkad,jkbd->ab",
                     einsum("ij,ikad->jkad", einsum("ic,jc->ij", ur1, ur1), t2_1), t2_2)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("ja,jb->ab",
                       einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, t1_2), ur1), t1_1)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("ja,jb->ab",
                       einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, t1_1), ur1), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ijkb,ijka->ab",
                       einsum("id,jkbd->ijkb", ur1, t2_2), einsum("ic,jkac->ijka", ur1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ib,ia->ab",
                        einsum("ijkd,jkbd->ib", einsum("ic,jkcd->ijkd", ur1, t2_2), t2_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ib,ia->ab",
                        einsum("ijkd,jkbd->ib", einsum("ic,jkcd->ijkd", ur1, t2_1), t2_2), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ia,ib->ab",
                       einsum("kd,kd->", einsum("jc,jkcd->kd", ur1, t2_1), t1_1) * ur1, t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ja->ab",
                       einsum("kd,jkbd->jb", t1_1, t2_1), einsum("ic,ic->", ur1, t1_1) * ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kb,ka->ab",
                     einsum("jd,jkbd->kb", einsum("ic,ic->", ur1, t1_1) * ur1, t2_1), t1_1)
        # second order ph-pphh/pphh-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("ja,jb->ab", einsum("ic,ijac->ja", ur1, ur2), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ib,ia->ab",
                     einsum("ijkd,jkbd->ib", einsum("ic,jkcd->ijkd", ur1, ur2), t2_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijkb,ijka->ab",
                     einsum("ic,jkbc->ijkb", t1_1, t2_1), einsum("id,jkad->ijka", ur1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("jb,ja->ab",
                     einsum("kc,jkbc->jb", t1_1, t2_1), einsum("id,ijad->ja", ur1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("kb,ka->ab",
                     einsum("id,ikbd->kb", einsum("jc,ijcd->id", ur1, ur2), t2_1), t1_1)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 2 * einsum("jkac,jkbc->ab",
                     einsum("ik,ijac->jkac", einsum("id,kd->ik", ur1, t1_1), ur2), t2_1)
    ).symmetrise()  # fmt: skip
    return dm


def diffdm_isr3(ground_state, amplitude, intermediates):
    dm = diffdm_isr3d(ground_state, amplitude, intermediates)
    try:
        check_triples_amplitudes([b.o, b.o, b.o, b.v, b.v, b.v], amplitude)
    except ValueError:
        return dm

    ur1, ur2, ur3 = amplitude.ph, amplitude.pphh, amplitude.ppphhh

    t2_1 = ground_state.td1(b.oovv)

    dm.oo += 2 * (
        # Apply (1 + P_ij) to:
        # first order ph-ppphhh/ppphhh-ph
        # N^6: O^3V^3 / N^6: O^3V^3
        + 3 * einsum("ilbc,jlbc->ij", einsum("ka,iklabc->ilbc", ur1, ur3), t2_1)
    ).symmetrise()  # fmt: skip
    dm.ov += (
        # zeroth order pphh-ppphhh/ppphhh-pphh
        + 3 * einsum("jkbc,ijkabc->ia", ur2, ur3)  # N^6: O^3V^3 / N^6: O^3V^3
    )  # fmt: skip
    dm.vv += 2 * (
        # Apply (1 + P_ab) to:
        # first order ph-ppphhh/ppphhh-ph
        # N^6: O^3V^3 / N^6: O^3V^3
        + 3 * einsum("jkac,jkbc->ab", einsum("id,ijkacd->jkac", ur1, ur3), t2_1)
    ).symmetrise()  # fmt: skip
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "isr0": diffdm_isr0,
    "isr1s": diffdm_isr1s,
    "isr1": diffdm_isr1,
    "isr2d": diffdm_isr2,  # Identical to ISR(2)
    "isr2": diffdm_isr2,
    "isr3d": diffdm_isr3d,
    "isr3": diffdm_isr3,
}


def rohf_dods_state_diffdm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle difference density matrix of an excited state
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
        raise NotImplementedError(f"state_diffdm is not implemented for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
