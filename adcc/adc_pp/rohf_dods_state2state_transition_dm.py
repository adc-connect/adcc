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


def s2s_tdm_isr0(ground_state, amplitude_l, amplitude_r, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)

    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    dm = OneParticleDensity(ground_state, symmetry=OperatorSymmetry.NOSYMMETRY)

    # zeroth order ph-ph
    dm.vv = +1 * einsum("ia,ib->ab", ul1, ur1)  # N^3: O^1V^2 / N^2: V^2
    dm.oo = -1 * einsum("ja,ia->ij", ul1, ur1)  # N^3: O^2V^1 / N^2: O^1V^1
    return dm


def s2s_tdm_isr1s(ground_state, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_isr0(ground_state, amplitude_l, amplitude_r, intermediates)

    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    t1_1 = ground_state.ts1(b.ov)

    # first order ph-ph
    dm.ov += (
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ul1, t1_1), ur1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ul1, ur1), t1_1)  # N^3: O^2V^1 / N^2: O^1V^1
    )  # fmt: skip
    dm.vo += (
        - 1 * einsum("ij,ja->ai", einsum("jb,ib->ij", ur1, t1_1), ul1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ai", einsum("ib,jb->ij", ul1, ur1), t1_1)  # N^3: O^2V^1 / N^2: O^1V^1
    )  # fmt: skip
    return dm


def s2s_tdm_isr1(ground_state, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_isr1s(ground_state, amplitude_l, amplitude_r, intermediates)

    try:
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    except ValueError:
        return dm

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    # zeroth order ph-pphh/pphh-ph
    dm.ov += -2 * einsum("jb,ijab->ia", ul1, ur2)  # N^4: O^2V^2 / N^4: O^2V^2
    dm.vo += -2 * einsum("ijab,jb->ai", ul2, ur1)  # N^4: O^2V^2 / N^4: O^2V^2
    return dm


def s2s_tdm_isr2(ground_state, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_isr1(ground_state, amplitude_l, amplitude_r, intermediates)
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)
    t1_2 = ground_state.ts2(b.ov)

    dm.oo += (
        # second order ph-ph
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,ik->ij", einsum("kb,jb->jk", ur1, t1_1), einsum("ka,ia->ik", ul1, t1_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jklc,iklc->ij",
                     einsum("lb,jkbc->jklc", ur1, t2_1), einsum("la,ikac->iklc", ul1, t2_1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("ik,jk->ij", einsum("ib,kb->ik", t1_1, t1_1), einsum("ja,ka->jk", ul1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("jk,ik->ij", einsum("jb,kb->jk", t1_1, t1_1), einsum("ka,ia->ik", ul1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ilbc,jlbc->ij",
                       einsum("kl,ikbc->ilbc", einsum("ka,la->kl", ul1, ur1), t2_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ib->ij",
                       einsum("kc,jkbc->jb", einsum("la,klac->kc", ul1, t2_1), t2_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jc,ic->ij",
                     einsum("lb,jlbc->jc", ur1, t2_1), einsum("ka,ikac->ic", ul1, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ia,ja->ij",
                       einsum("lc,ilac->ia", einsum("kb,klbc->lc", ur1, t2_1), t2_1), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("jl,il->ij",
                        einsum("jkbc,klbc->jl", t2_1, t2_1), einsum("la,ia->il", ul1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ik,jk->ij",
                        einsum("ilbc,klbc->ik", t2_1, t2_1), einsum("ja,ka->jk", ul1, ur1))
        # first order ph-pphh
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ul1, ur2), t1_1)
        # first order pphh-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("jb,ib->ij", einsum("jkab,ka->jb", ul2, ur1), t1_1)
        # zeroth order pphh-pphh
        - 2 * einsum("jkab,ikab->ij", ul2, ur2)  # N^5: O^3V^2 / N^4: O^2V^2
    )  # fmt: skip
    dm.ov += (
        # second order ph-ph
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ul1, t1_2), ur1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ul1, ur1), t1_2)  # N^3: O^2V^1 / N^2: O^1V^1
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("jc,ijbc->ib", t1_1, t2_1), ul1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kc,ikac->ia", einsum("jk,jc->kc", einsum("kb,jb->jk", ul1, ur1), t1_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kc,ikac->ia", einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, t1_1), ul1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ia", einsum("kb,jkab->ja", t1_1, t2_1), einsum("jc,ic->ij", ul1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ja->ia", einsum("ic,jc->ij", einsum("kb,ikbc->ic", ul1, t2_1), t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", ur1, t1_1), einsum("jc,jkac->ka", ul1, t2_1))
        # first order pphh-ph
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ja->ia", einsum("jkbc,ikbc->ij", ul2, t2_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijkc,jkac->ia", einsum("jkbc,ib->ijkc", ul2, ur1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("kb,ikab->ia", einsum("jkbc,jc->kb", ul2, ur1), t2_1)
    )  # fmt: skip
    dm.vo += (
        # second order ph-ph
        - 1 * einsum("ij,ja->ai", einsum("jb,ib->ij", ur1, t1_2), ul1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ai", einsum("ib,jb->ij", ul1, ur1), t1_2)  # N^3: O^2V^1 / N^2: O^1V^1
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ai", einsum("ib,kb->ik", einsum("jc,ijbc->ib", t1_1, t2_1), ur1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ai", einsum("kb,jkab->ja", t1_1, t2_1), einsum("ic,jc->ij", ul1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ai", einsum("jk,kb->jb", einsum("kc,jc->jk", ul1, t1_1), ur1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ai", einsum("jk,kb->jb", einsum("kc,jc->jk", ul1, ur1), t1_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ai", einsum("ic,kc->ik", einsum("jb,ijbc->ic", ur1, t2_1), t1_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ka,ik->ai", einsum("jb,jkab->ka", ur1, t2_1), einsum("ic,kc->ik", ul1, t1_1))
        # first order ph-pphh
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ai", einsum("jkbc,ijbc->ik", ur2, t2_1), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ijkb,jkab->ai", einsum("ic,jkbc->ijkb", ul1, ur2), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,ijab->ai", einsum("kc,jkbc->jb", ul1, ur2), t2_1)
    )  # fmt: skip
    dm.vv += (
        # second order ph-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ka,kb->ab",
                     einsum("jc,jkac->ka", ur1, t2_1), einsum("id,ikbd->kb", ul1, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ib,ia->ab",
                       einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_1), t2_1), ul1)
        # N^3: O^1V^2 / N^2: V^2
        - 1 * einsum("ia,ib->ab", einsum("ij,ja->ia", einsum("ic,jc->ij", ul1, ur1), t1_1), t1_1)
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum("ikac,ikbc->ab",
                     einsum("ij,jkac->ikac", einsum("id,jd->ij", ul1, ur1), t2_1), t2_1)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("ja,jb->ab", einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, t1_1), ul1), t1_1)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("jb,ja->ab", einsum("ij,ib->jb", einsum("ic,jc->ij", ul1, t1_1), ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ja,jb->ab",
                       einsum("kc,jkac->ja", einsum("id,ikcd->kc", ul1, t2_1), t2_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ijka,ijkb->ab",
                       einsum("ic,jkac->ijka", ur1, t2_1), einsum("id,jkbd->ijkb", ul1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ib,ia->ab",
                        einsum("ijkd,jkbd->ib", einsum("ic,jkcd->ijkd", ur1, t2_1), t2_1), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ia,ib->ab",
                        einsum("ijkc,jkac->ia", einsum("id,jkcd->ijkc", ul1, t2_1), t2_1), ur1)
        # first order ph-pphh
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,ja->ab", einsum("ic,ijbc->jb", ul1, ur2), t1_1)
        # first order pphh-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ia,ib->ab", einsum("ijac,jc->ia", ul2, ur1), t1_1)
        # zeroth order pphh-pphh
        + 2 * einsum("ijac,ijbc->ab", ul2, ur2)  # N^5: O^2V^3 / N^4: O^2V^2
    )  # fmt: skip
    return dm


def s2s_tdm_isr3d(ground_state, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_isr2(ground_state, amplitude_l, amplitude_r, intermediates)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)
    t1_2 = ground_state.ts2(b.ov)
    t2_2 = ground_state.td2(b.oovv)
    t3_2 = ground_state.tt2(b.ooovvv)
    t1_3 = ground_state.ts3(b.ov)

    dm.oo += (
        # third order ph-ph
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,ik->ij", einsum("kb,jb->jk", ur1, t1_2), einsum("ka,ia->ik", ul1, t1_1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,ik->ij", einsum("kb,jb->jk", ur1, t1_1), einsum("ka,ia->ik", ul1, t1_2))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jklc,iklc->ij",
                     einsum("lb,jkbc->jklc", ur1, t2_2), einsum("la,ikac->iklc", ul1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jklc,iklc->ij",
                     einsum("lb,jkbc->jklc", ur1, t2_1), einsum("la,ikac->iklc", ul1, t2_2))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("ik,jk->ij", einsum("ib,kb->ik", t1_1, t1_2), einsum("ja,ka->jk", ul1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("ik,jk->ij", einsum("kb,ib->ik", t1_1, t1_2), einsum("ja,ka->jk", ul1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("jk,ik->ij", einsum("jb,kb->jk", t1_1, t1_2), einsum("ka,ia->ik", ul1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 0.5 * einsum("jk,ik->ij", einsum("kb,jb->jk", t1_1, t1_2), einsum("ka,ia->ik", ul1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ilbc,jlbc->ij",
                       einsum("kl,ikbc->ilbc", einsum("ka,la->kl", ul1, ur1), t2_1), t2_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("jkbc,ikbc->ij",
                       einsum("kl,jlbc->jkbc", einsum("ka,la->kl", ul1, ur1), t2_1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ib->ij",
                       einsum("kc,jkbc->jb", einsum("la,klac->kc", ul1, t2_1), t2_2), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ib->ij",
                       einsum("kc,jkbc->jb", einsum("la,klac->kc", ul1, t2_2), t2_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jc,ic->ij",
                     einsum("lb,jlbc->jc", ur1, t2_2), einsum("ka,ikac->ic", ul1, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jc,ic->ij",
                     einsum("lb,jlbc->jc", ur1, t2_1), einsum("ka,ikac->ic", ul1, t2_2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ia,ja->ij",
                       einsum("lc,ilac->ia", einsum("kb,klbc->lc", ur1, t2_2), t2_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ia,ja->ij",
                       einsum("lc,ilac->ia", einsum("kb,klbc->lc", ur1, t2_1), t2_2), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("jl,il->ij",
                        einsum("jkbc,klbc->jl", t2_1, t2_2), einsum("la,ia->il", ul1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("jl,il->ij",
                        einsum("klbc,jkbc->jl", t2_1, t2_2), einsum("la,ia->il", ul1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ik,jk->ij",
                        einsum("ilbc,klbc->ik", t2_1, t2_2), einsum("ja,ka->jk", ul1, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ik,jk->ij",
                        einsum("klbc,ilbc->ik", t2_1, t2_2), einsum("ja,ka->jk", ul1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ka,ka->", ul1, t1_1) * einsum("jc,ic->ij",
                                                    einsum("lb,jlbc->jc", ur1, t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("lb,lb->", ur1, t1_1) * einsum("ic,jc->ij",
                                                    einsum("ka,ikac->ic", ul1, t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("kc,kc->",
                       einsum("la,klac->kc", ul1, t2_1), t1_1) * einsum("ib,jb->ij", ur1, t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("lc,lc->",
                       einsum("kb,klbc->lc", ur1, t2_1), t1_1) * einsum("ja,ia->ij", ul1, t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("kb,kb->", ur1, t1_1) * einsum("ia,ja->ij",
                                                      einsum("lc,ilac->ia", t1_1, t2_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ka,ka->", ul1, t1_1) * einsum("jb,ib->ij",
                                                      einsum("lc,jlbc->jb", t1_1, t2_1), ur1)
        # second order ph-pphh
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ul1, ur2), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jk,ik->ij",
                     einsum("klbc,jlbc->jk", ur2, t2_1), einsum("ka,ia->ik", ul1, t1_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ikbc,jkbc->ij",
                     einsum("kl,ilbc->ikbc", einsum("la,ka->kl", ul1, t1_1), ur2), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 2 * einsum("jklb,iklb->ij",
                     einsum("kc,jlbc->jklb", t1_1, t2_1), einsum("ka,ilab->iklb", ul1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jc,ic->ij",
                     einsum("lb,jlbc->jc", einsum("ka,klab->lb", ul1, ur2), t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,ib->ij",
                     einsum("kc,jkbc->jb", t1_1, t2_1), einsum("la,ilab->ib", ul1, ur2))
        # second order pphh-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("jb,ib->ij", einsum("jkab,ka->jb", ul2, ur1), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jlab,ilab->ij",
                     einsum("kl,jkab->jlab", einsum("kc,lc->kl", ur1, t1_1), ul2), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("jk,ik->ij",
                     einsum("kc,jc->jk", ur1, t1_1), einsum("klab,ilab->ik", ul2, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 2 * einsum("iklb,jklb->ij",
                     einsum("kc,ilbc->iklb", t1_1, t2_1), einsum("jlab,ka->jklb", ul2, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("ib,jb->ij",
                     einsum("lc,ilbc->ib", t1_1, t2_1), einsum("jkab,ka->jb", ul2, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("ic,jc->ij",
                     einsum("lb,ilbc->ic", einsum("klab,ka->lb", ul2, ur1), t2_1), t1_1)
    )  # fmt: skip
    dm.ov += (
        # third order ph-ph
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ul1, t1_3), ur1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ul1, ur1), t1_3)  # N^3: O^2V^1 / N^2: O^1V^1
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ia",
                     einsum("kc,kc->", t1_1, t1_1) * einsum("jb,ib->ij", ul1, t1_1), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ia",
                     einsum("kc,kc->", t1_1, t1_1) * einsum("jb,ib->ij", ul1, ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("jc,ijbc->ib", t1_1, t2_2), ul1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("ijbc,jc->ib", t2_1, t1_2), ul1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kc,ikac->ia", einsum("jk,jc->kc", einsum("kb,jb->jk", ul1, ur1), t1_1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kc,ikac->ia", einsum("jk,jc->kc", einsum("kb,jb->jk", ul1, ur1), t1_2), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kc,ikac->ia", einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, t1_1), ul1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kc,ikac->ia", einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, t1_2), ul1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ia", einsum("kb,jkab->ja", t1_1, t2_2), einsum("jc,ic->ij", ul1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ia", einsum("jkab,kb->ja", t2_1, t1_2), einsum("jc,ic->ij", ul1, ur1))
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("ij,ja->ia",
                       einsum("ikcd,jkcd->ij", einsum("lb,iklbcd->ikcd", ul1, t3_2), t2_1), ur1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("iklc,klac->ia",
                       einsum("ib,klbc->iklc", ur1, t2_1), einsum("jd,jklacd->klac", ul1, t3_2))
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("klbc,iklabc->ia",
                       einsum("jk,jlbc->klbc", einsum("kd,jd->jk", ul1, ur1), t2_1), t3_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ja->ia", einsum("ic,jc->ij", einsum("kb,ikbc->ic", ul1, t2_2), t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ja->ia", einsum("ic,jc->ij", einsum("kb,ikbc->ic", ul1, t2_1), t1_2), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", ur1, t1_1), einsum("jc,jkac->ka", ul1, t2_2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", ur1, t1_2), einsum("jc,jkac->ka", ul1, t2_1))
        # N^6: O^3V^3 / N^6: O^3V^3
        - 1 * einsum("kc,ikac->ia",
                     einsum("jb,jkbc->kc", ur1, t2_1), einsum("ld,iklacd->ikac", ul1, t3_2))
        # N^6: O^3V^3 / N^6: O^3V^3
        - 0.5 * einsum("jlcd,ijlacd->ia",
                       einsum("jklc,kd->jlcd", einsum("kb,jlbc->jklc", ur1, t2_1), ul1), t3_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.25 * einsum("ij,ja->ia",
                        einsum("klcd,klcd->", t2_1, t2_1) * einsum("jb,ib->ij", ul1, t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.25 * einsum("ij,ja->ia",
                        einsum("klcd,klcd->", t2_1, t2_1) * einsum("jb,ib->ij", ul1, ur1), t1_1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.25 * einsum("ik,ka->ia",
                        einsum("ib,kb->ik", einsum("jlcd,ijlbcd->ib", t2_1, t3_2), ul1), ur1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.25 * einsum("ja,ij->ia",
                        einsum("klbd,jklabd->ja", t2_1, t3_2), einsum("jc,ic->ij", ul1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ia",
                     einsum("kb,kb->", ul1, t1_1) * einsum("ic,jc->ij", t1_1, t1_1), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,jk->",
                     einsum("jc,kc->jk", t1_1, t1_1), einsum("kb,jb->jk", ul1, ur1)) * t1_1
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ik,ka->ia",
                     einsum("jc,jc->", ul1, t1_1) * einsum("ib,kb->ik", ur1, t1_1), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,jk->",
                     einsum("jb,kb->jk", ur1, t1_1), einsum("jc,kc->jk", ul1, t1_1)) * t1_1
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("jc,ic->ij",
                              einsum("ld,jlcd->jc",
                                     einsum("kb,klbd->ld", ul1, t2_1), t2_1), t1_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("jl,il->ij",
                              einsum("jkcd,klcd->jl", t2_1, t2_1),
                              einsum("lb,ib->il", ul1, t1_1)), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("lb,lb->", ul1, t1_1) * einsum("ikcd,jkcd->ij", t2_1, t2_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ik,ka->ia",
                       einsum("kb,ib->ik",
                              einsum("ld,klbd->kb",
                                     einsum("jc,jlcd->ld", ul1, t2_1), t2_1), ur1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("jb,ib->ij",
                              einsum("jklc,klbc->jb",
                                     einsum("jd,klcd->jklc", ul1, t2_1), t2_1), ur1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("iklc,klac->ia",
                       einsum("ib,klbc->iklc", einsum("jd,jd->", ul1, t1_1) * ur1, t2_1), t2_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia",
                     einsum("jk,ik->ij",
                            einsum("jc,kc->jk", t1_1, t1_1), einsum("kb,ib->ik", ul1, t1_1)), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ia",
                     einsum("ik,jk->ij",
                            einsum("ib,kb->ik", ur1, t1_1),
                            einsum("jc,kc->jk", ul1, t1_1)), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jb,jb->",
                     einsum("kc,jkbc->jb", einsum("ld,klcd->kc", ul1, t2_1), t2_1), ur1) * t1_1
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kc,kc->",
                     einsum("jb,jkbc->kc", ur1, t2_1), t1_1) * einsum("ld,ilad->ia", ul1, t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kc,ikac->ia",
                     einsum("jb,jkbc->kc", einsum("ld,ld->", ul1, t1_1) * ur1, t2_1), t2_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ia",
                       einsum("ik,jk->ij",
                              einsum("ic,kc->ik", t1_1, t1_1), einsum("jb,kb->jk", ul1, t1_1)), ur1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ik,ka->ia",
                       einsum("jk,ij->ik",
                              einsum("jc,kc->jk", t1_1, t1_1), einsum("jb,ib->ij", ul1, ur1)), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ia",
                       einsum("jk,ik->ij",
                              einsum("kb,jb->jk", ur1, t1_1), einsum("kc,ic->ik", ul1, t1_1)), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ik,ka->ia",
                       einsum("ij,jk->ik",
                              einsum("ib,jb->ij", t1_1, t1_1), einsum("kc,jc->jk", ul1, ur1)), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("la,il->ia",
                       einsum("kd,klad->la", einsum("jb,jkbd->kd", ur1, t2_1), t2_1),
                       einsum("lc,ic->il", ul1, t1_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ik,ka->ia",
                       einsum("id,kd->ik",
                              einsum("lc,ilcd->id",
                                     einsum("jb,jlbc->lc", ur1, t2_1), t2_1), ul1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("lb,lb->",
                       einsum("jklc,jkbc->lb",
                              einsum("ld,jkcd->jklc", ul1, t2_1), t2_1), ur1) * t1_1
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("jl,jl->",
                       einsum("jkbc,klbc->jl", t2_1, t2_1), einsum("ld,jd->jl", ul1, ur1)) * t1_1
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ij,ja->ia",
                        einsum("jc,ic->ij",
                               einsum("jkld,klcd->jc",
                                      einsum("jb,klbd->jkld", ul1, t2_1), t2_1), t1_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ik,ka->ia",
                        einsum("jk,ij->ik",
                               einsum("jlcd,klcd->jk", t2_1, t2_1),
                               einsum("jb,ib->ij", ul1, ur1)), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ijkd,jkad->ia",
                        einsum("ib,jkbd->ijkd",
                               einsum("il,lb->ib",
                                      einsum("lc,ic->il", ul1, t1_1), ur1), t2_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ik,ka->ia",
                        einsum("ij,jk->ik",
                               einsum("ilbc,jlbc->ij", t2_1, t2_1),
                               einsum("kd,jd->jk", ul1, ur1)), t1_1)
        # second order ph-pphh
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ij,ja->ia", einsum("ic,jc->ij", t1_1, t1_1), einsum("kb,jkab->ja", ul1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ia", einsum("ib,kb->ik", einsum("jc,ijbc->ib", ul1, ur2), t1_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("ikcd,jkcd->ij", t2_1, t2_1), einsum("lb,jlab->ja", ul1, ur2))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijkb,jkab->ia",
                       einsum("id,jkbd->ijkb", einsum("lc,ilcd->id", ul1, t2_1), t2_1), ur2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("il,la->ia",
                       einsum("ikbc,klbc->il", ur2, t2_1), einsum("jd,jlad->la", ul1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("iklc,klac->ia",
                       einsum("ib,klbc->iklc", einsum("jd,ijbd->ib", ul1, ur2), t2_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("kd,ikad->ia",
                       einsum("kl,ld->kd", einsum("jlbc,jkbc->kl", ur2, t2_1), ul1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("lc,ilac->ia",
                       einsum("jklb,jkbc->lc", einsum("ld,jkbd->jklb", ul1, ur2), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijkc,jkac->ia",
                     einsum("ikld,jlcd->ijkc", einsum("kb,ilbd->ikld", ul1, t2_1), t2_1), ur2)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijlc,jlac->ia",
                     einsum("ijkb,klbc->ijlc", einsum("jd,ikbd->ijkb", ul1, ur2), t2_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kc,ikac->ia",
                     einsum("jb,jkbc->kc", einsum("ld,jlbd->jb", ul1, ur2), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 0.25 * einsum("ijkl,jkla->ia",
                        einsum("ilcd,jkcd->ijkl", t2_1, t2_1), einsum("lb,jkab->jkla", ul1, ur2))
        # N^6: O^4V^2 / N^4: O^2V^2
        - 0.25 * einsum("ijkl,jkla->ia",
                        einsum("ijbc,klbc->ijkl", ur2, t2_1), einsum("jd,klad->jkla", ul1, t2_1))
        # second order pphh-ph
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ja->ia", einsum("jkbc,ikbc->ij", ul2, t2_2), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijkc,jkac->ia", einsum("jkbc,ib->ijkc", ul2, ur1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("kb,ikab->ia", einsum("jkbc,jc->kb", ul2, ur1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("ik,ka->ia", einsum("kb,ib->ik", einsum("jkbc,jc->kb", ul2, t1_1), t1_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("ij,ja->ia", einsum("jc,ic->ij", einsum("jkbc,kb->jc", ul2, t1_1), ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,jb->", einsum("jkbc,kc->jb", ul2, ur1), t1_1) * t1_1
        # first order pphh-pphh
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("ijkb,jkab->ia", einsum("jkbc,ic->ijkb", ul2, t1_1), ur2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("ik,ka->ia", einsum("jkbc,ijbc->ik", ul2, ur2), t1_1)
    )  # fmt: skip
    dm.vo += (
        # third order ph-ph
        - 1 * einsum("ij,ja->ai", einsum("jb,ib->ij", ur1, t1_3), ul1)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ai", einsum("ib,jb->ij", ul1, ur1), t1_3)  # N^3: O^2V^1 / N^2: O^1V^1
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ai",
                     einsum("kc,kc->", t1_1, t1_1) * einsum("jb,ib->ij", ur1, t1_1), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ij,ja->ai",
                     einsum("kc,kc->", t1_1, t1_1) * einsum("ib,jb->ij", ul1, ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ai", einsum("ib,kb->ik", einsum("jc,ijbc->ib", t1_1, t2_2), ur1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ai", einsum("ib,kb->ik", einsum("ijbc,jc->ib", t2_1, t1_2), ur1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ai", einsum("kb,jkab->ja", t1_1, t2_2), einsum("ic,jc->ij", ul1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ja,ij->ai", einsum("jkab,kb->ja", t2_1, t1_2), einsum("ic,jc->ij", ul1, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ai", einsum("jk,kb->jb", einsum("kc,jc->jk", ul1, t1_1), ur1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ai", einsum("jk,kb->jb", einsum("kc,jc->jk", ul1, t1_2), ur1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ai", einsum("jk,kb->jb", einsum("kc,jc->jk", ul1, ur1), t1_1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ai", einsum("jk,kb->jb", einsum("kc,jc->jk", ul1, ur1), t1_2), t2_1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("il,la->ai",
                       einsum("ikcd,klcd->il", einsum("jb,ijkbcd->ikcd", ur1, t3_2), t2_1), ul1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("klac,iklc->ai",
                       einsum("jb,jklabc->klac", ur1, t3_2), einsum("id,klcd->iklc", ul1, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ai", einsum("ic,kc->ik", einsum("jb,ijbc->ic", ur1, t2_2), t1_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ik,ka->ai", einsum("ic,kc->ik", einsum("jb,ijbc->ic", ur1, t2_1), t1_2), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ka,ik->ai", einsum("jb,jkab->ka", ur1, t2_2), einsum("ic,kc->ik", ul1, t1_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ka,ik->ai", einsum("jb,jkab->ka", ur1, t2_1), einsum("ic,kc->ik", ul1, t1_2))
        # N^6: O^3V^3 / N^6: O^3V^3
        - 1 * einsum("jkbc,ijkabc->ai",
                     einsum("kc,jb->jkbc", einsum("ld,klcd->kc", ul1, t2_1), ur1), t3_2)
        # N^6: O^3V^3 / N^6: O^3V^3
        - 0.5 * einsum("jkbc,ijkabc->ai",
                       einsum("jklc,lb->jkbc", einsum("ld,jkcd->jklc", ul1, t2_1), ur1), t3_2)
        # N^6: O^3V^3 / N^6: O^3V^3
        - 0.5 * einsum("jkbc,ijkabc->ai",
                       einsum("jl,klbc->jkbc", einsum("ld,jd->jl", ul1, ur1), t2_1), t3_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.25 * einsum("ij,ja->ai",
                        einsum("klcd,klcd->", t2_1, t2_1) * einsum("jb,ib->ij", ur1, t1_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.25 * einsum("ij,ja->ai",
                        einsum("klcd,klcd->", t2_1, t2_1) * einsum("ib,jb->ij", ul1, ur1), t1_1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.25 * einsum("il,la->ai",
                        einsum("ib,lb->il", einsum("jkcd,ijkbcd->ib", t2_1, t3_2), ur1), ul1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.25 * einsum("ja,ij->ai",
                        einsum("klbc,jklabc->ja", t2_1, t3_2), einsum("id,jd->ij", ul1, ur1))
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ik,ka->ai",
                     einsum("jb,jb->", ur1, t1_1) * einsum("ic,kc->ik", t1_1, t1_1), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,jk->",
                     einsum("jc,kc->jk", t1_1, t1_1),
                     einsum("kb,jb->jk", ul1, ur1)) * einsum("ia->ai", t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ik,ka->ai",
                     einsum("jb,jb->", ur1, t1_1) * einsum("ic,kc->ik", ul1, t1_1), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("jk,jk->",
                     einsum("jb,kb->jk", ur1, t1_1),
                     einsum("jc,kc->jk", ul1, t1_1)) * einsum("ia->ai", t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("il,la->ai",
                       einsum("jl,ij->il",
                              einsum("jkcd,klcd->jl", t2_1, t2_1),
                              einsum("jb,ib->ij", ur1, t1_1)), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("il,la->ai",
                       einsum("ld,id->il",
                              einsum("kc,klcd->ld",
                                     einsum("jb,jkbc->kc", ur1, t2_1), t2_1), t1_1), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ai",
                       einsum("ib,jb->ij",
                              einsum("iklc,klbc->ib",
                                     einsum("id,klcd->iklc", ul1, t2_1), t2_1), ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("il,la->ai",
                       einsum("ld,id->il",
                              einsum("kc,klcd->ld",
                                     einsum("jb,jkbc->kc", ur1, t2_1), t2_1), ul1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ai",
                       einsum("ib,jb->ij",
                              einsum("kc,ikbc->ib",
                                     einsum("ld,klcd->kc", ul1, t2_1), t2_1), ur1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ai",
                       einsum("jb,ib->ij", ur1, t1_1),
                       einsum("kc,jkac->ja", einsum("ld,klcd->kc", ul1, t2_1), t2_1))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ik,ka->ai",
                     einsum("jk,ij->ik",
                            einsum("jc,kc->jk", t1_1, t1_1), einsum("jb,ib->ij", ur1, t1_1)), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ij,ja->ai",
                     einsum("jk,ik->ij",
                            einsum("jb,kb->jk", ur1, t1_1), einsum("ic,kc->ik", ul1, t1_1)), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jb,jb->",
                     einsum("kc,jkbc->jb",
                            einsum("ld,klcd->kc", ul1, t2_1), t2_1), ur1) * einsum("ia->ai", t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jb,jb->", ur1, t1_1) * einsum("kc,ikac->ai",
                                                    einsum("ld,klcd->kc", ul1, t2_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jb,ijab->ai",
                     einsum("kc,kc->", einsum("ld,klcd->kc", ul1, t2_1), t1_1) * ur1, t2_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ik,ka->ai",
                       einsum("ij,jk->ik",
                              einsum("ic,jc->ij", t1_1, t1_1), einsum("kb,jb->jk", ur1, t1_1)), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("il,la->ai",
                       einsum("jb,jb->", ur1, t1_1) * einsum("ikcd,klcd->il", t2_1, t2_1), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ik,ka->ai",
                       einsum("jk,ij->ik",
                              einsum("jb,kb->jk", t1_1, t1_1), einsum("ic,jc->ij", ul1, ur1)), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai",
                       einsum("ik,jk->ij",
                              einsum("kb,ib->ik", ur1, t1_1), einsum("kc,jc->jk", ul1, t1_1)), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai",
                       einsum("ik,jk->ij",
                              einsum("ib,kb->ik", t1_1, t1_1), einsum("kc,jc->jk", ul1, ur1)), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("jb,jb->", ur1, t1_1) * einsum("iklc,klac->ai",
                                                      einsum("id,klcd->iklc", ul1, t2_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("lb,lb->",
                       einsum("jklc,jkbc->lb",
                              einsum("ld,jkcd->jklc", ul1, t2_1), t2_1), ur1)
                                      * einsum("ia->ai", t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("jl,jl->",
                       einsum("jkbc,klbc->jl", t2_1, t2_1),
                       einsum("ld,jd->jl", ul1, ur1)) * einsum("ia->ai", t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("il,la->ai",
                        einsum("ld,id->il",
                               einsum("jklc,jkcd->ld",
                                      einsum("lb,jkbc->jklc", ur1, t2_1), t2_1), t1_1), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("il,la->ai",
                        einsum("jl,ij->il",
                               einsum("jkbc,klbc->jl", t2_1, t2_1),
                               einsum("id,jd->ij", ul1, ur1)), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("il,la->ai",
                        einsum("lb,ib->il", ur1, t1_1),
                        einsum("jklc,jkac->la", einsum("ld,jkcd->jklc", ul1, t2_1), t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ij,ja->ai",
                        einsum("il,jl->ij",
                               einsum("ikbc,klbc->il", t2_1, t2_1),
                               einsum("ld,jd->jl", ul1, ur1)), t1_1)
        # second order ph-pphh
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ai", einsum("jkbc,ijbc->ik", ur2, t2_2), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ijkb,jkab->ai", einsum("ic,jkbc->ijkb", ul1, ur2), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,ijab->ai", einsum("kc,jkbc->jb", ul1, ur2), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ik,ka->ai", einsum("kc,ic->ik", einsum("jkbc,jb->kc", ur2, t1_1), t1_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ik,ka->ai", einsum("kc,ic->ik", einsum("jkbc,jb->kc", ur2, t1_1), ul1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,jb->", einsum("kc,jkbc->jb", ul1, ur2), t1_1) * einsum("ia->ai", t1_1)
        # second order pphh-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ij,ja->ai", einsum("ic,jc->ij", t1_1, t1_1), einsum("jkab,kb->ja", ul2, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ik,ka->ai", einsum("ib,kb->ik", einsum("ijbc,jc->ib", ul2, ur1), t1_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ai",
                       einsum("ikcd,jkcd->ij", t2_1, t2_1), einsum("jlab,lb->ja", ul2, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijlb,jlab->ai",
                       einsum("id,jlbd->ijlb", einsum("kc,ikcd->id", ur1, t2_1), t2_1), ul2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("la,il->ai",
                       einsum("kc,klac->la", ur1, t2_1), einsum("ijbd,jlbd->il", ul2, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("iklc,klac->ai",
                       einsum("ib,klbc->iklc", einsum("ijbd,jd->ib", ul2, ur1), t2_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("kc,ikac->ai",
                       einsum("kl,lc->kc", einsum("jlbd,jkbd->kl", ul2, t2_1), ur1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("kc,ikac->ai",
                       einsum("jklb,jlbc->kc", einsum("jlbd,kd->jklb", ul2, ur1), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijlb,jlab->ai",
                     einsum("ikld,jkbd->ijlb", einsum("lc,ikcd->ikld", ur1, t2_1), t2_1), ul2)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("iklc,klac->ai",
                     einsum("ijkb,jlbc->iklc", einsum("ijbd,kd->ijkb", ul2, ur1), t2_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kc,ikac->ai",
                     einsum("lb,klbc->kc", einsum("jlbd,jd->lb", ul2, ur1), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 0.25 * einsum("ijkl,jkla->ai",
                        einsum("ikcd,jlcd->ijkl", t2_1, t2_1), einsum("jlab,kb->jkla", ul2, ur1))
        # N^6: O^4V^2 / N^4: O^2V^2
        - 0.25 * einsum("iklc,klac->ai",
                        einsum("ijkl,jc->iklc", einsum("ijbd,klbd->ijkl", ul2, t2_1), ur1), t2_1)
        # first order pphh-pphh
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("ijkb,jkab->ai", einsum("jkbc,ic->ijkb", ur2, t1_1), ul2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("ik,ka->ai", einsum("ijbc,jkbc->ik", ul2, ur2), t1_1)
    )  # fmt: skip
    dm.vv += (
        # third order ph-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ka,kb->ab",
                     einsum("jc,jkac->ka", ur1, t2_1), einsum("id,ikbd->kb", ul1, t2_2))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ka,kb->ab",
                     einsum("jc,jkac->ka", ur1, t2_2), einsum("id,ikbd->kb", ul1, t2_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ib,ia->ab",
                       einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_2), t2_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ib,ia->ab",
                       einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_1), t2_2), ul1)
        # N^3: O^1V^2 / N^2: V^2
        - 1 * einsum("ia,ib->ab", einsum("ij,ja->ia", einsum("ic,jc->ij", ul1, ur1), t1_1), t1_2)
        # N^3: O^1V^2 / N^2: V^2
        - 1 * einsum("jb,ja->ab", einsum("ij,ib->jb", einsum("ic,jc->ij", ul1, ur1), t1_1), t1_2)
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum("ikac,ikbc->ab",
                     einsum("ij,jkac->ikac", einsum("id,jd->ij", ul1, ur1), t2_1), t2_2)
        # N^5: O^2V^3 / N^4: O^2V^2
        - 1 * einsum("jkbc,jkac->ab",
                     einsum("ij,ikbc->jkbc", einsum("id,jd->ij", ul1, ur1), t2_1), t2_2)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("ja,jb->ab", einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, t1_2), ul1), t1_1)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("ja,jb->ab", einsum("ij,ia->ja", einsum("ic,jc->ij", ur1, t1_1), ul1), t1_2)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("jb,ja->ab", einsum("ij,ib->jb", einsum("ic,jc->ij", ul1, t1_2), ur1), t1_1)
        # N^3: O^1V^2 / N^2: V^2
        - 0.5 * einsum("jb,ja->ab", einsum("ij,ib->jb", einsum("ic,jc->ij", ul1, t1_1), ur1), t1_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ja,jb->ab",
                       einsum("kc,jkac->ja", einsum("id,ikcd->kc", ul1, t2_2), t2_1), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ja,jb->ab",
                       einsum("kc,jkac->ja", einsum("id,ikcd->kc", ul1, t2_1), t2_2), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ijka,ijkb->ab",
                       einsum("ic,jkac->ijka", ur1, t2_1), einsum("id,jkbd->ijkb", ul1, t2_2))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ijka,ijkb->ab",
                       einsum("ic,jkac->ijka", ur1, t2_2), einsum("id,jkbd->ijkb", ul1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ib,ia->ab",
                        einsum("ijkd,jkbd->ib", einsum("ic,jkcd->ijkd", ur1, t2_2), t2_1), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ib,ia->ab",
                        einsum("ijkd,jkbd->ib", einsum("ic,jkcd->ijkd", ur1, t2_1), t2_2), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ia,ib->ab",
                        einsum("ijkc,jkac->ia", einsum("id,jkcd->ijkc", ul1, t2_2), t2_1), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ia,ib->ab",
                        einsum("ijkc,jkac->ia", einsum("id,jkcd->ijkc", ul1, t2_1), t2_2), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ia,ib->ab",
                       einsum("kd,kd->", einsum("jc,jkcd->kd", ur1, t2_1), t1_1) * ul1, t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ib,ia->ab",
                       einsum("kd,ikbd->ib", t1_1, t2_1), einsum("jc,jc->", ur1, t1_1) * ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ja,jb->ab",
                       einsum("kd,jkad->ja", t1_1, t2_1), einsum("ic,ic->", ul1, t1_1) * ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("kb,ka->ab",
                     einsum("jc,jc->", ur1, t1_1) * einsum("id,ikbd->kb", ul1, t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ka,kb->ab",
                     einsum("jc,jkac->ka", einsum("id,id->", ul1, t1_1) * ur1, t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("jb,ja->ab",
                       einsum("kc,kc->", einsum("id,ikcd->kc", ul1, t2_1), t1_1) * ur1, t1_1)
        # second order ph-pphh
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,ja->ab", einsum("ic,ijbc->jb", ul1, ur2), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ia,ib->ab",
                     einsum("ijkc,jkac->ia", einsum("id,jkcd->ijkc", ul1, ur2), t2_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijka,ijkb->ab",
                     einsum("ic,jkac->ijka", t1_1, t2_1), einsum("id,jkbd->ijkb", ul1, ur2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ja,jb->ab",
                     einsum("kc,jkac->ja", t1_1, t2_1), einsum("id,ijbd->jb", ul1, ur2))
        # N^5: O^2V^3 / N^4: O^2V^2
        + 2 * einsum("jkbc,jkac->ab",
                     einsum("ik,ijbc->jkbc", einsum("id,kd->ik", ul1, t1_1), ur2), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("ka,kb->ab",
                     einsum("jc,jkac->ka", einsum("id,ijcd->jc", ul1, ur2), t2_1), t1_1)
        # second order pphh-ph
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ia,ib->ab", einsum("ijac,jc->ia", ul2, ur1), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijkb,ijka->ab",
                     einsum("kc,ijbc->ijkb", t1_1, t2_1), einsum("ijad,kd->ijka", ul2, ur1))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("kb,ka->ab",
                     einsum("ijkd,ijbd->kb", einsum("ijcd,kc->ijkd", ul2, ur1), t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("jb,ja->ab",
                     einsum("kc,jkbc->jb", t1_1, t2_1), einsum("ijad,id->ja", ul2, ur1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("kb,ka->ab",
                     einsum("ic,ikbc->kb", einsum("ijcd,jd->ic", ul2, ur1), t2_1), t1_1)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 2 * einsum("jkad,jkbd->ab",
                     einsum("ik,ijad->jkad", einsum("ic,kc->ik", ur1, t1_1), ul2), t2_1)
    )  # fmt: skip
    return dm


def s2s_tdm_isr3(ground_state, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_isr3d(ground_state, amplitude_l, amplitude_r, intermediates)

    try:
        check_triples_amplitudes([b.o, b.o, b.o, b.v, b.v, b.v], amplitude_l, amplitude_r)
    except ValueError:
        return dm

    ul1, ul2, ul3 = amplitude_l.ph, amplitude_l.pphh, amplitude_l.ppphhh
    ur1, ur2, ur3 = amplitude_r.ph, amplitude_r.pphh, amplitude_r.ppphhh

    t2_1 = ground_state.td1(b.oovv)

    dm.oo += (
        # first order ph-ppphhh
        # N^6: O^3V^3 / N^6: O^3V^3
        + 3 * einsum("ilbc,jlbc->ij", einsum("ka,iklabc->ilbc", ul1, ur3), t2_1)
        # first order ppphhh-ph
        # N^6: O^3V^3 / N^6: O^3V^3
        - 3 * einsum("jkbc,ikbc->ij", einsum("jklabc,la->jkbc", ul3, ur1), t2_1)
    )  # fmt: skip
    dm.ov += (
        # zeroth order pphh-ppphhh
        + 3 * einsum("jkbc,ijkabc->ia", ul2, ur3)  # N^6: O^3V^3 / N^6: O^3V^3
    )  # fmt: skip
    dm.vo += (
        # zeroth order ppphhh-pphh
        + 3 * einsum("ijkabc,jkbc->ai", ul3, ur2)  # N^6: O^3V^3 / N^6: O^3V^3
    )  # fmt: skip
    dm.vv += (
        # first order ph-ppphhh
        # N^6: O^3V^3 / N^6: O^3V^3
        + 3 * einsum("jkbc,jkac->ab", einsum("id,ijkbcd->jkbc", ul1, ur3), t2_1)
        # first order ppphhh-ph
        # N^6: O^3V^3 / N^6: O^3V^3
        + 3 * einsum("jkac,jkbc->ab", einsum("ijkacd,id->jkac", ul3, ur1), t2_1)
    )  # fmt: skip
    return dm


# Ref: https://doi.org/10.1080/00268976.2013.859313
DISPATCH = {
    "isr0": s2s_tdm_isr0,
    "isr1s": s2s_tdm_isr1s,
    "isr1": s2s_tdm_isr1,
    "isr2": s2s_tdm_isr2,
    "isr3d": s2s_tdm_isr3d,
    "isr3": s2s_tdm_isr3,
}


def rohf_dods_state2state_transition_dm(
    method, ground_state, amplitude_from, amplitude_to, intermediates=None
):
    """
    Compute the state to state transition density matrix
    state in the MO basis using the intermediate-states representation.

    Parameters
    ----------
    method : str, IsrMethod
        The method to use for the computation (e.g. "isr2")
    ground_state : LazyRoDodsMp
        The ground state upon which the excitation was based
    amplitude_from : AmplitudeVector
        The amplitude vector of the state to start from
    amplitude_to : AmplitudeVector
        The amplitude vector of the state to excite to
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, IsrMethod):
        method = IsrMethod(method)
    if not isinstance(ground_state, LazyRoDodsMp):
        raise TypeError("ground_state should be a LazyRoDodsMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError(
            f"state2state_transition_dm is not implemented for {method.name}."
        )
    else:
        # final state is on the bra side/left (complex conjugate)
        # see ref https://doi.org/10.1080/00268976.2013.859313, appendix A2
        ret = DISPATCH[method.name](ground_state, amplitude_to, amplitude_from, intermediates)
        return ret.evaluate()
