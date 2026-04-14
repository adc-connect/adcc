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


def s2s_tdm_isr0_2p(mp, amplitude_l, amplitude_r, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    dm = TwoParticleDensity(mp, symmetry=OperatorSymmetry.NOSYMMETRY)
    dm.oooo = 4 * (
        # N^4: O^4 / N^4: O^4
        + 1 * einsum("jk,il->ijkl", einsum("ka,ja->jk", ul1, ur1), d_oo)
    ).antisymmetrise(0, 1).antisymmetrise(2, 3)

    dm.ovov = (
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ja,ib->iajb", ul1, ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ab,ij->iajb", einsum("ka,kb->ab", ul1, ur1), d_oo)
    )
    return dm


def s2s_tdm_isr1_2p(mp, amplitude_l, amplitude_r, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_isr0_2p(mp, amplitude_l, amplitude_r, intermediates)

    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)

    dm.oovv += (
        4 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ib,ja->ijab", einsum("kc,ikbc->ib", ul1, t2), ur1)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijka,kb->ijab", einsum("kc,ijac->ijka", ul1, t2), ur1)
        ).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ikab->ijab", einsum("kc,jc->jk", ul1, ur1), t2)
        ).antisymmetrise(0, 1)
    )

    dm.vvoo += (
        4 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ib,ja->abij", einsum("kc,ikbc->ib", ur1, t2), ul1)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ikab->abij", einsum("jc,kc->jk", ul1, ur1), t2)
        ).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijka,kb->abij", einsum("kc,ijac->ijka", ur1, t2), ul1)
        ).antisymmetrise(0, 1)
    )

    try:
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
        ul2, ur2 = amplitude_l.pphh, amplitude_r.pphh
        dm.ooov += (
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("kb,ijab->ijka", ul1, ur2)
        )
        dm.ovoo += (
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("jkab,ib->iajk", ul2, ur1)
        )
        dm.ovvv += (
            # N^5: O^2V^3 / N^4: O^1V^3
            - 2 * einsum("ja,ijbc->iabc", ul1, ur2)
        )
        dm.vvov += (
            # N^5: O^2V^3 / N^4: O^1V^3
            - 2 * einsum("ijab,jc->abic", ul2, ur1)
        )

    except ValueError:
        pass

    return dm


def s2s_tdm_isr2_2p(mp, amplitude_l, amplitude_r, intermediates):
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_isr1_2p(mp, amplitude_l, amplitude_r, intermediates)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    dm.oooo += (
        # N^6: O^4V^2 / N^4: O^2V^2
        + 2 * einsum("klab,ijab->ijkl", ul2, ur2)
        # N^6: O^5V^1 / N^4: O^2V^2
        - 1 * einsum("klmc,ijmc->ijkl",
                     einsum("mb,klbc->klmc", ur1, t2),
                     einsum("ma,ijac->ijmc", ul1, t2))
        + 4 * (
            # N^4: O^4 / N^4: O^4
            + 1 * einsum("jk,il->ijkl", einsum("ka,ja->jk", ul1, ur1), p0.oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            - 2 * einsum("ik,jl->ijkl", einsum("kmab,imab->ik", ul2, ur2), d_oo)
            # N^6: O^5V^1 / N^4: O^2V^2
            + 1 * einsum("jlmc,ikmc->ijkl", einsum("jb,lmbc->jlmc", ur1, t2),
                         einsum("ka,imac->ikmc", ul1, t2))
            # N^4: O^4 / N^4: O^4
            + 0.5 * einsum("jk,il->ijkl",
                           einsum("km,jm->jk",
                                  einsum("ka,ma->km", ul1, ur1), p0.oo), d_oo)
            # N^4: O^4 / N^4: O^4
            + 0.5 * einsum("il,jk->ijkl",
                           einsum("im,lm->il",
                                  einsum("ma,ia->im", ul1, ur1), p0.oo), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("il,jk->ijkl",
                         einsum("lc,ic->il", einsum("nb,lnbc->lc", ur1, t2),
                                einsum("ma,imac->ic", ul1, t2)), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ik,jl->ijkl",
                         einsum("kmnc,imnc->ik", einsum("nb,kmbc->kmnc", ur1, t2),
                                einsum("na,imac->imnc", ul1, t2)), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jl->ijkl",
                           einsum("ia,ka->ik",
                                  einsum("mc,imac->ia",
                                         einsum("nb,mnbc->mc", ur1, t2), t2), ul1),
                           d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jl->ijkl",
                           einsum("inbc,knbc->ik",
                                  einsum("mn,imbc->inbc",
                                         einsum("ma,na->mn", ul1, ur1), t2), t2),
                           d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jl->ijkl",
                           einsum("kb,ib->ik",
                                  einsum("mc,kmbc->kb",
                                         einsum("na,mnac->mc", ul1, t2), t2), ur1),
                           d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("kc,ijlc->ijkl",
                         einsum("mb,kmbc->kc", ur1, t2),
                         einsum("la,ijac->ijlc", ul1, t2))
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijlm,km->ijkl",
                           einsum("ijbc,lmbc->ijlm", t2, t2),
                           einsum("ka,ma->km", ul1, ur1))
        ).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("iklb,jb->ijkl",
                         einsum("ic,klbc->iklb",
                                einsum("ma,imac->ic", ul1, t2), t2), ur1)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.5 * einsum("jklm,im->ijkl",
                           einsum("jmbc,klbc->jklm", t2, t2),
                           einsum("ma,ia->im", ul1, ur1))
        ).antisymmetrise(0, 1)
    )

    dm.ooov += (
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1 * einsum("ijkl,la->ijka", einsum("klbc,ijbc->ijkl", ul2, t2), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("kb,ijab->ijka", einsum("klbc,lc->kb", ul2, ur1), t2)
        + 2 * (
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("jk,ia->ijka", einsum("kb,jb->jk", ul1, p0.ov), ur1)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("jk,ia->ijka", einsum("kb,jb->jk", ul1, ur1), p0.ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ia->ijka", einsum("klbc,jlbc->jk", ul2, t2), ur1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ja,ik->ijka", einsum("lb,jlab->ja", ul1, ur2), d_oo)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 2 * einsum("jklb,ilab->ijka", einsum("klbc,jc->jklb", ul2, ur1), t2)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ia,jk->ijka",
                         einsum("il,la->ia",
                                einsum("lb,ib->il", ul1, p0.ov), ur1), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ia,jk->ijka",
                         einsum("il,la->ia",
                                einsum("lb,ib->il", ul1, ur1), p0.ov), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ja,ik->ijka",
                         einsum("jm,ma->ja",
                                einsum("lmbc,jlbc->jm", ul2, t2), ur1), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ia,jk->ijka",
                         einsum("ilmc,lmac->ia",
                                einsum("lmbc,ib->ilmc", ul2, ur1), t2), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ia,jk->ijka",
                         einsum("lb,ilab->ia",
                                einsum("lmbc,mc->lb", ul2, ur1), t2), d_oo)
        ).antisymmetrise(0, 1)
    )

    dm.oovv += (
        4 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ib,ja->ijab", einsum("kc,ikbc->ib", ul1, td2), ur1)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijka,kb->ijab", einsum("kc,ijac->ijka", ul1, td2), ur1)
        ).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ikab->ijab", einsum("kc,jc->jk", ul1, ur1), td2)
        ).antisymmetrise(0, 1)
    )

    dm.ovoo += (
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1 * einsum("ijkl,la->iajk", einsum("ilbc,jkbc->ijkl", ur2, t2), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 2 * einsum("ib,jkab->iajk", einsum("lc,ilbc->ib", ul1, ur2), t2)
        + 2 * (
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ik,ja->iajk", einsum("ib,kb->ik", ur1, p0.ov), ul1)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ik,ja->iajk", einsum("ilbc,klbc->ik", ur2, t2), ul1)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ik,ja->iajk", einsum("kb,ib->ik", ul1, ur1), p0.ov)
            # N^6: O^4V^2 / N^4: O^2V^2
            - 2 * einsum("iklb,jlab->iajk", einsum("kc,ilbc->iklb", ul1, ur2), t2)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ka,ij->iajk", einsum("klab,lb->ka", ul2, ur1), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ja,ik->iajk",
                         einsum("jl,la->ja",
                                einsum("lb,jb->jl", ur1, p0.ov), ul1), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ka,ij->iajk",
                         einsum("km,ma->ka",
                                einsum("lmbc,klbc->km", ur2, t2), ul1), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ja,ik->iajk",
                         einsum("jl,la->ja",
                                einsum("jb,lb->jl", ul1, ur1), p0.ov), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ka,ij->iajk",
                         einsum("klmb,lmab->ka",
                                einsum("kc,lmbc->klmb", ul1, ur2), t2), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2 * einsum("ja,ik->iajk",
                         einsum("lb,jlab->ja",
                                einsum("mc,lmbc->lb", ul1, ur2), t2), d_oo)
        ).antisymmetrise(2, 3)
    )

    dm.ovov += (
        # N^6: O^3V^3 / N^4: O^2V^2
        - 4 * einsum("jkac,ikbc->iajb", ul2, ur2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ab,ij->iajb", einsum("ka,kb->ab", ul1, ur1), p0.oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ib,ja->iajb", einsum("ic,bc->ib", ur1, p0.vv), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ja,ib->iajb", einsum("jc,ac->ja", ul1, p0.vv), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ij,ab->iajb", einsum("jc,ic->ij", ul1, ur1), p0.vv)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 2 * einsum("ab,ij->iajb", einsum("klac,klbc->ab", ul2, ur2), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ib,ja->iajb", einsum("kb,ik->ib", ur1, p0.oo), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ja,ib->iajb", einsum("ka,jk->ja", ul1, p0.oo), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ijla,lb->iajb",
                     einsum("ic,jlac->ijla",
                            einsum("kd,ikcd->ic", ul1, t2), t2), ur1)
        # N^6: O^3V^3 / N^4: O^2V^2
        + 1 * einsum("jkac,ikbc->iajb",
                     einsum("kl,jlac->jkac",
                            einsum("kd,ld->kl", ul1, ur1), t2), t2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1 * einsum("ijka,kb->iajb",
                     einsum("ic,jkac->ijka", ur1, t2),
                     einsum("ld,klbd->kb", ul1, t2))
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1 * einsum("jkla,iklb->iajb",
                     einsum("lc,jkac->jkla", ur1, t2),
                     einsum("ld,ikbd->iklb", ul1, t2))
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijla,lb->iajb",
                       einsum("ijkl,ka->ijla",
                              einsum("ikcd,jlcd->ijkl", t2, t2), ul1), ur1)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("ikla,jklb->iajb",
                       einsum("ic,klac->ikla", ur1, t2),
                       einsum("jd,klbd->jklb", ul1, t2))
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("ijkb,ka->iajb",
                     einsum("jd,ikbd->ijkb",
                            einsum("lc,jlcd->jd", ur1, t2), t2), ul1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijkb,ka->iajb",
                     einsum("jklc,ilbc->ijkb",
                            einsum("kd,jlcd->jklc", ur1, t2), t2), ul1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijlb,la->iajb",
                     einsum("ijkc,klbc->ijlb",
                            einsum("id,jkcd->ijkc", ur1, t2), t2), ul1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijla,lb->iajb",
                     einsum("ijkc,klac->ijla",
                            einsum("jd,ikcd->ijkc", ul1, t2), t2), ur1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1 * einsum("la,ijlb->iajb",
                     einsum("kc,klac->la", ur1, t2),
                     einsum("jd,ilbd->ijlb", ul1, t2))
        # N^6: O^3V^3 / N^4: O^2V^2
        - 1 * einsum("jkac,ikbc->iajb",
                     einsum("jl,klac->jkac",
                            einsum("jd,ld->jl", ul1, ur1), t2), t2)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijka,kb->iajb",
                     einsum("iklc,jlac->ijka",
                            einsum("kd,ilcd->iklc", ul1, t2), t2), ur1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ja,ib->iajb",
                     einsum("lc,jlac->ja", ur1, t2), einsum("kd,ikbd->ib", ul1, t2))
        # N^6: O^3V^3 / N^4: O^2V^2
        - 1 * einsum("ikbc,jkac->iajb",
                     einsum("il,klbc->ikbc", einsum("ld,id->il", ul1, ur1), t2), t2)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ib,ja->iajb",
                       einsum("ld,ilbd->ib",
                              einsum("kc,klcd->ld", ur1, t2), t2), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb",
                       einsum("kb,ka->ab",
                              einsum("kc,bc->kb", ur1, p0.vv), ul1), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb",
                       einsum("ka,kb->ab",
                              einsum("kc,ac->ka", ul1, p0.vv), ur1), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ja,ib->iajb",
                       einsum("kc,jkac->ja",
                              einsum("ld,klcd->kc", ul1, t2), t2), ur1)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 1 * einsum("ab,ij->iajb",
                     einsum("lmac,lmbc->ab",
                            einsum("km,klac->lmac",
                                   einsum("md,kd->km", ul1, ur1), t2), t2), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ab,ij->iajb",
                       einsum("ka,kb->ab",
                              einsum("lc,klac->ka",
                                     einsum("md,lmcd->lc", ul1, t2), t2),
                              ur1), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ab,ij->iajb",
                     einsum("la,lb->ab",
                            einsum("kc,klac->la", ur1, t2),
                            einsum("md,lmbd->lb", ul1, t2)), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb",
                       einsum("mb,ma->ab",
                              einsum("ld,lmbd->mb",
                                     einsum("kc,klcd->ld", ur1, t2), t2),
                              ul1), d_oo)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb",
                       einsum("klma,klmb->ab",
                              einsum("mc,klac->klma", ur1, t2),
                              einsum("md,klbd->klmb", ul1, t2)), d_oo)
    )

    dm.ovvv += (
        # N^6: O^3V^3 / N^4: O^1V^3
        + 1 * einsum("ijka,jkbc->iabc", einsum("jkad,id->ijka", ul2, ur1), t2)
        # N^5: O^2V^3 / N^4: O^1V^3
        + 2 * einsum("ja,ijbc->iabc", einsum("jkad,kd->ja", ul2, ur1), t2)
        + 2 * (
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ac,ib->iabc", einsum("ja,jc->ac", ul1, p0.ov), ur1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("ac,ib->iabc", einsum("ja,jc->ac", ul1, ur1), p0.ov)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("ac,ib->iabc", einsum("jkad,jkcd->ac", ul2, t2), ur1)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 2 * einsum("ikab,kc->iabc", einsum("jkad,ijbd->ikab", ul2, t2), ur1)
        ).antisymmetrise(2, 3)
    )

    dm.vvoo += (
        4 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ib,ja->abij", einsum("kc,ikbc->ib", ur1, td2), ul1)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("ijka,kb->abij", einsum("kc,ijac->ijka", ur1, td2), ul1)
        ).antisymmetrise(0, 1)
        + 2 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1 * einsum("jk,ikab->abij", einsum("jc,kc->jk", ul1, ur1), td2)
        ).antisymmetrise(2, 3)
    )

    dm.vvov += (
        # N^6: O^3V^3 / N^4: O^1V^3
        + 1 * einsum("ijkc,jkab->abic", einsum("id,jkcd->ijkc", ul1, ur2), t2)
        # N^5: O^2V^3 / N^4: O^1V^3
        + 2 * einsum("jc,ijab->abic", einsum("kd,jkcd->jc", ul1, ur2), t2)
        + 2 * (
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("bc,ia->abic", einsum("jc,jb->bc", ur1, p0.ov), ul1)
            # N^5: O^2V^3 / N^4: O^1V^3
            + 1 * einsum("bc,ia->abic", einsum("jkcd,jkbd->bc", ur2, t2), ul1)
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1 * einsum("bc,ia->abic", einsum("jb,jc->bc", ul1, ur1), p0.ov)
            # N^6: O^3V^3 / N^4: O^1V^3
            - 2 * einsum("ikac,kb->abic", einsum("jkcd,ijad->ikac", ur2, t2), ul1)
        ).antisymmetrise(0, 1)
    )

    dm.vvvv += (
        # N^6: O^2V^4 / N^4: V^4
        + 2 * einsum("ijab,ijcd->abcd", ul2, ur2)
        # N^6: O^2V^4 / N^4: V^4
        - 1 * einsum("ikab,ikcd->abcd",
                     einsum("ij,jkab->ikab", einsum("ie,je->ij", ul1, ur1), t2), t2)
        + 4 * (
            # N^4: V^4 / N^4: V^4
            + 1 * einsum("ac,bd->abcd", einsum("ia,ic->ac", ul1, ur1), p0.vv)
            # N^6: O^3V^3 / N^4: V^4
            + 1 * einsum("jabc,jd->abcd",
                         einsum("ijbc,ia->jabc",
                                einsum("jkbe,ikce->ijbc", t2, t2), ul1), ur1)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        + 2 * (
            # N^5: O^1V^4 / N^4: V^4
            + 1 * einsum("ka,kbcd->abcd",
                         einsum("je,jkae->ka", ur1, t2),
                         einsum("ib,ikcd->kbcd", ul1, t2))
            # N^6: O^3V^3 / N^4: V^4
            + 0.5 * einsum("ibcd,ia->abcd",
                           einsum("ijkb,jkcd->ibcd",
                                  einsum("ie,jkbe->ijkb", ur1, t2), t2), ul1)
        ).antisymmetrise(0, 1)
        + 2 * (
            # N^5: O^1V^4 / N^4: V^4
            + 1 * einsum("jabc,jd->abcd",
                         einsum("kc,jkab->jabc",
                                einsum("ie,ikce->kc", ul1, t2), t2), ur1)
            # N^6: O^3V^3 / N^4: V^4
            + 0.5 * einsum("iabd,ic->abcd",
                           einsum("ijkd,jkab->iabd",
                                  einsum("ie,jkde->ijkd", ul1, t2), t2), ur1)
        ).antisymmetrise(2, 3)
    )
    return dm


DISPATCH = {"isr0": s2s_tdm_isr0_2p,
            "isr1": s2s_tdm_isr1_2p,
            "isr2": s2s_tdm_isr2_2p,
            "isr2x": s2s_tdm_isr2_2p,      # same as ISR(2)
            }


def state2state_transition_dm_2p(method, ground_state, amplitude_from,
                                 amplitude_to, intermediates=None):
    """
    Compute the two-particle state to state transition density matrix
    state in the MO basis using the intermediate-states representation.

    Parameters
    ----------
    method : str, IsrMethod
        The method to use for the computation (e.g. "isr2")
    ground_state : LazyMp
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
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state2state_transition_dm_2p is not implemented "
                                  f"for {method.name}.")
    else:
        # final state is on the bra side/left (complex conjugate) (analogous to
        # 1p densities)
        # see ref https://doi.org/10.1080/00268976.2013.859313, appendix A2
        ret = DISPATCH[method.name](ground_state, amplitude_to, amplitude_from,
                                    intermediates)
        return ret.evaluate()
