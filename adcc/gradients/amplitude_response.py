#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2021 by the adcc authors
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
from math import sqrt

from adcc.functions import einsum, direct_sum, evaluate

from .TwoParticleDensityMatrix import TwoParticleDensityMatrix
from adcc.OneParticleOperator import OneParticleOperator
from adcc.LazyMp import LazyMp
from adcc.Excitation import Excitation
import adcc.block as b


def t2bar_oovv_adc2(exci, g1a_adc0):
    mp = exci.ground_state
    hf = mp.reference_state
    u = exci.excitation_vector
    df_ia = mp.df(b.ov)
    t2bar = 0.5 * (
        hf.oovv
        - 2.0 * einsum(
            "ijcb,ac->ijab", hf.oovv, g1a_adc0.vv
        ).antisymmetrise(2, 3)
        + 2.0 * einsum(
            "kjab,ik->ijab", hf.oovv, g1a_adc0.oo
        ).antisymmetrise(0, 1)
        + 4.0 * einsum(
            "ia,jkbc,kc->ijab", u.ph, hf.oovv, u.ph
        ).antisymmetrise(2, 3).antisymmetrise(0, 1)
    ) / (
        2.0 * direct_sum("ia+jb->ijab", df_ia, df_ia).symmetrise(0, 1)
    )
    return t2bar


def tbarD_oovv_adc3(exci, g1a_adc0):
    mp = exci.ground_state
    hf = mp.reference_state
    u = exci.excitation_vector
    df_ia = mp.df(b.ov)
    # Table XI (10.1063/1.5085117)
    tbarD = 0.5 * (
        hf.oovv
        - 2.0 * einsum(
            "ijcb,ac->ijab", hf.oovv, g1a_adc0.vv
        ).antisymmetrise(2, 3)
        + 2.0 * einsum(
            "kjab,ik->ijab", hf.oovv, g1a_adc0.oo
        ).antisymmetrise(0, 1)
        + 4.0 * einsum(
            "ia,jkbc,kc->ijab", u.ph, hf.oovv, u.ph
        ).antisymmetrise(2, 3).antisymmetrise(0, 1)
    ) / (
        2.0 * direct_sum("ia+jb->ijab", df_ia, df_ia).symmetrise(0, 1)
    )
    return tbarD


def tbar_TD_oovv_adc3(exci, g1a_adc0):
    mp = exci.ground_state
    hf = mp.reference_state
    # Table XI (10.1063/1.5085117)
    tbarD = tbarD_oovv_adc3(exci, g1a_adc0)
    tbarD.evaluate()
    ret = 2.0 * 4.0 * (
        + 2.0 * einsum("ikac,jckb->ijab", tbarD, hf.ovov)
        - 1.0 * einsum("ijcd,abcd->ijab", tbarD, hf.vvvv)
        - 1.0 * einsum("klab,ijkl->ijab", tbarD, hf.oooo)
    ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    return ret


def rho_bar_adc3(exci, g1a_adc0):
    mp = exci.ground_state
    hf = mp.reference_state
    u = exci.excitation_vector
    df_ia = mp.df(b.ov)
    rho_bar = 2.0 * (
        + 1.0 * einsum("ijka,jk->ia", hf.ooov, g1a_adc0.oo)
        + 1.0 * einsum("ijkb,jb,ka->ia", hf.ooov, u.ph, u.ph)
        - 1.0 * einsum("icab,bc->ia", hf.ovvv, g1a_adc0.vv)
        + 1.0 * einsum("jcab,jb,ic->ia", hf.ovvv, u.ph, u.ph)
    ) / df_ia
    return rho_bar


def tbar_rho_oovv_adc3(exci, g1a_adc0):
    mp = exci.ground_state
    hf = mp.reference_state
    # Table XI (10.1063/1.5085117)
    rho_bar = rho_bar_adc3(exci, g1a_adc0)
    ret = (
        - 1.0 * 2.0 * einsum("jcab,ic->ijab", hf.ovvv, rho_bar).antisymmetrise(0, 1)
        - 1.0 * 2.0 * einsum("ijkb,ka->ijab", hf.ooov, rho_bar).antisymmetrise(2, 3)
    )
    return ret


def t2bar_oovv_adc3(exci, g1a_adc0, g2a_adc1):
    mp = exci.ground_state
    hf = mp.reference_state
    u = exci.excitation_vector
    tbar_TD = tbar_TD_oovv_adc3(exci, g1a_adc0)
    tbar_TD.evaluate()
    tbar_rho = tbar_rho_oovv_adc3(exci, g1a_adc0)
    tbar_rho.evaluate()
    t2 = mp.t2(b.oovv).evaluate()
    df_ia = mp.df(b.ov)
    rx = einsum("jb,ijab->ia", u.ph, t2).evaluate()

    ttilde1 = (
        - 1.0 * einsum("klab,ijkm,lm->ijab", t2, hf.oooo, g1a_adc0.oo)
        - 1.0 * 2.0 * einsum("ka,ijkl,lb->ijab", u.ph, hf.oooo, rx).antisymmetrise(2, 3)
        + 1.0 * 2.0 * (
            + 0.5 * einsum("ik,jklm,lmab->ijab", g1a_adc0.oo, hf.oooo, t2)
            - 2.0 * einsum("jkab,lm,ilkm->ijab", t2, g1a_adc0.oo, hf.oooo)
        ).antisymmetrise(0, 1)
        - 1.0 * 4.0 * (
            + 0.5 * einsum("ia,kc,jklm,lmbc->ijab", u.ph, u.ph, hf.oooo, t2)
            - 2.0 * einsum("ka,likm,jmbc,lc->ijab", u.ph, hf.oooo, t2, u.ph)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    ttilde1.evaluate()

    ttilde2 = 4.0 * (
        - 1.0 * einsum("ijkc,lc,klab->ijab", hf.ooov, u.ph, u.pphh)
        + 1.0 * 2.0 * einsum("ikab,jlkc,lc->ijab", u.pphh, hf.ooov, u.ph).antisymmetrise(0, 1)
        - 1.0 * 4.0 * einsum("jklb,kc,ilac->ijab", hf.ooov, u.ph, u.pphh).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    ttilde2.evaluate()

    ttilde3 = (
        + 1.0 * 2.0 * einsum("ijbc,ac->ijab", hf.oovv, g1a_adc0.vv).antisymmetrise(2, 3)
        - 1.0 * 2.0 * einsum("jkab,ik->ijab", hf.oovv, g1a_adc0.oo).antisymmetrise(0, 1)
        + 1.0 * 4.0 * einsum("ia,jkbc,kc->ijab", u.ph, hf.oovv, u.ph).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    ttilde3.evaluate()

    # TODO: intermediate x_ka <ic||kd>?
    # TODO: intermediate x_lc t_jlbd?
    ttilde4 = (
        + 1.0 * 2.0 * (
            - 2.0 * einsum("jkab,ickd,cd->ijab", t2, hf.ovov, g1a_adc0.vv) #1 k
            - 2.0 * einsum("klab,ickd,jcld->ijab", t2, g2a_adc1.ovov, hf.ovov)
            + 1.0 * einsum("ic,jkab,ld,lckd->ijab", u.ph, t2, u.ph, hf.ovov)
            + 1.0 * einsum("jkab,kc,ld,lcid->ijab", t2, u.ph, u.ph, hf.ovov)
        ).antisymmetrise(0, 1)
        + 1.0 * 2.0 * (
            - 2.0 * einsum("ijcb,kalc,kl->ijab", t2, hf.ovov, g1a_adc0.oo) #2 k
            - 2.0 * einsum("ijcd,kalc,kbld->ijab", t2, g2a_adc1.ovov, hf.ovov)
            + 1.0 * einsum("ka,ijbc,ld,kdlc->ijab", u.ph, t2, u.ph, hf.ovov)
            + 1.0 * einsum("ijbc,kc,ld,kdla->ijab", t2, u.ph, u.ph, hf.ovov)
        ).antisymmetrise(2, 3)
        + 1.0 * 4.0 * (
            + 1.0 * einsum("ac,jdkc,ikbd->ijab", g1a_adc0.vv, hf.ovov, t2) #3 k
            - 1.0 * einsum("jckb,ikad,cd->ijab", hf.ovov, t2, g1a_adc0.vv) #4 k
            - 1.0 * einsum("ik,kclb,jlac->ijab", g1a_adc0.oo, hf.ovov, t2) #5 k
            + 1.0 * einsum("jckb,ilac,lk->ijab", hf.ovov, t2, g1a_adc0.oo) #6 k
            - 1.0 * einsum("ic,jckb,ka->ijab", u.ph, hf.ovov, rx)
            - 1.0 * einsum("jb,kc,lcid,klad->ijab", u.ph, u.ph, hf.ovov, t2)
            - 1.0 * einsum("ka,jckb,ic->ijab", u.ph, hf.ovov, rx)
            - 1.0 * einsum("jb,kc,lakd,ilcd->ijab", u.ph, u.ph, hf.ovov, t2)
            + 2.0 * einsum("ka,ickd,ld,jlbc->ijab", u.ph, hf.ovov, u.ph, t2)
            + 2.0 * einsum("ic,kcla,kd,jlbd->ijab", u.ph, hf.ovov, u.ph, t2)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    ttilde4.evaluate()

    ttilde5 = - 4.0 * (
        + 1.0 * einsum("kcab,kd,ijcd->ijab", hf.ovvv, u.ph, u.pphh)
        + 1.0 * 2.0 * einsum("ijbc,kcad,kd->ijab", u.pphh, hf.ovvv, u.ph).antisymmetrise(2, 3)
        + 1.0 * 4.0 * einsum("jcbd,kd,ikac->ijab", hf.ovvv, u.ph, u.pphh).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    ttilde5.evaluate()

    ttilde6 = (
        - 1.0 * einsum("ijcd,abde,ce->ijab", t2, hf.vvvv, g1a_adc0.vv)
        - 1.0 * 2.0 * einsum("ic,abcd,jkde,ke->ijab", u.ph, hf.vvvv, t2, u.ph).antisymmetrise(0, 1)
        - 1.0 * 2.0 * (
            + 0.5 * einsum("ac,bcde,ijde->ijab", g1a_adc0.vv, hf.vvvv, t2)
            - 2.0 * einsum("ijbc,de,adce->ijab", t2, g1a_adc0.vv, hf.vvvv)
        ).antisymmetrise(2, 3)
        - 1.0 * 4.0 * (
            + 0.5 * einsum("ia,kc,bcde,jkde->ijab", u.ph, u.ph, hf.vvvv, t2)
            - 2.0 * einsum("ic,adce,jkeb,kd->ijab", u.ph, hf.vvvv, t2, u.ph)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    ttilde6.evaluate()
    # TODO: prefactors etc maybe wrong...
    ret = 0.5 * (
        hf.oovv
        + ttilde1 + ttilde2 + ttilde3 + ttilde4 + ttilde5 + ttilde6
        + tbar_TD + tbar_rho 
    ) / (2.0 * direct_sum("ia+jb->ijab", df_ia, df_ia).symmetrise(0, 1))
    return ret


def t2bar_oovv_cvs_adc2(exci, g1a_adc0):
    mp = exci.ground_state
    hf = mp.reference_state
    df_ia = mp.df(b.ov)
    t2bar = 0.5 * (
        - einsum("ijcb,ac->ijab", hf.oovv, g1a_adc0.vv).antisymmetrise(2, 3)
    ) / direct_sum("ia+jb->ijab", df_ia, df_ia).symmetrise(0, 1)
    return t2bar


def ampl_relaxed_dms_mp2(mp):
    hf = mp.reference_state
    t2 = mp.t2(b.oovv)
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    g1a.oo = -0.5 * einsum('ikab,jkab->ij', t2, t2)
    g1a.vv = 0.5 * einsum('ijac,ijbc->ab', t2, t2)
    g2a.oovv = -1.0 * mp.t2(b.oovv)
    return g1a, g2a


def ampl_relaxed_dms_adc0(exci):
    hf = exci.reference_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    # g2a is not required for the adc0 gradient,
    # but expected by amplitude_relaxed_densities
    g1a.oo = - 1.0 * einsum("ia,ja->ij", u.ph, u.ph)
    g1a.vv = + 1.0 * einsum("ia,ib->ab", u.ph, u.ph)
    return g1a, g2a


def ampl_relaxed_dms_adc1(exci):
    hf = exci.reference_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    g1a.oo = - 1.0 * einsum("ia,ja->ij", u.ph, u.ph)
    g1a.vv = + 1.0 * einsum("ia,ib->ab", u.ph, u.ph)
    g2a.ovov = - 1.0 * einsum("ja,ib->iajb", u.ph, u.ph)
    return g1a, g2a


def ampl_relaxed_dms_adc2(exci):
    u = exci.excitation_vector
    mp = exci.ground_state
    g1a_adc1, g2a_adc1 = ampl_relaxed_dms_adc1(exci)
    t2 = mp.t2(b.oovv)
    t2bar = t2bar_oovv_adc2(exci, g1a_adc1).evaluate()

    g1a = g1a_adc1.copy()
    g1a.oo += (
        - 2.0 * einsum('jkab,ikab->ij', u.pphh, u.pphh)
        - 2.0 * einsum('jkab,ikab->ij', t2bar, t2).symmetrise((0, 1))
    )
    g1a.vv += (
        + 2.0 * einsum("ijac,ijbc->ab", u.pphh, u.pphh)
        + 2.0 * einsum("ijac,ijbc->ab", t2bar, t2).symmetrise((0, 1))
    )

    g2a = g2a_adc1.copy()
    ru_ov = einsum("ijab,jb->ia", t2, u.ph)
    g2a.oovv = (
        0.5 * (
            - 1.0 * t2
            + 2.0 * einsum("ijcb,ca->ijab", t2, g1a_adc1.vv).antisymmetrise((2, 3))
            - 2.0 * einsum("kjab,ki->ijab", t2, g1a_adc1.oo).antisymmetrise((0, 1))
            - 4.0 * einsum(
                "ia,jb->ijab", u.ph, ru_ov
            ).antisymmetrise((0, 1)).antisymmetrise((2, 3))
        )
        - 2.0 * t2bar
    )
    g2a.ooov = -2.0 * einsum("kb,ijab->ijka", u.ph, u.pphh)
    g2a.ovvv = -2.0 * einsum("ja,ijbc->iabc", u.ph, u.pphh)
    return g1a, g2a


def ampl_relaxed_dms_adc2x(exci):
    u = exci.excitation_vector
    g1a, g2a = ampl_relaxed_dms_adc2(exci)

    g2a.ovov += -4.0 * einsum("ikbc,jkac->iajb", u.pphh, u.pphh)
    g2a.oooo = 2.0 * einsum('ijab,klab->ijkl', u.pphh, u.pphh)
    g2a.vvvv = 2.0 * einsum('ijcd,ijab->abcd', u.pphh, u.pphh)

    return g1a, g2a


def ampl_relaxed_dms_adc3(exci):
    u = exci.excitation_vector
    mp = exci.ground_state
    hf = mp.reference_state
    g1a_adc1, g2a_adc1 = ampl_relaxed_dms_adc1(exci)
    t2bar = t2bar_oovv_adc3(exci, g1a_adc1, g2a_adc1).evaluate()
    tbarD = tbarD_oovv_adc3(exci, g1a_adc1).evaluate()
    rho_bar = rho_bar_adc3(exci, g1a_adc1).evaluate()
    t2 = mp.t2(b.oovv)
    tD = mp.td2(b.oovv)
    rho = mp.mp2_diffdm

    print("bar", 0.25 * t2bar.dot(hf.oovv))

    # Table IX (10.1063/1.5085117) 
    g1a = g1a_adc1.copy()
    g1a.oo += (
        - 2.0 * einsum("jkab,ikab->ij", u.pphh, u.pphh)
        - 1.0 * 2.0 * (
            + 1.0 * einsum("ikab,jkab->ij", t2, t2bar)
            + 1.0 * einsum("ikab,jkab->ij", tD, tbarD)
            + 0.5 * einsum("ia,ja->ij", rho_bar, rho.ov)
        ).symmetrise(0, 1)
    )
    g1a.vv += (
        + 2.0 * einsum("ijac,ijbc->ab", u.pphh, u.pphh)
        + 1.0 * 2.0 *(
            + 1.0 * einsum("ijac,ijbc->ab", t2bar, t2)
            + 1.0 * einsum("ijac,ijbc->ab", tbarD, tD)
            + 0.5 * einsum("ia,ib->ab", rho_bar, rho.ov)
        ).symmetrise(0, 1)
    )

    tsq_ovov = einsum("ikac,jkbc->iajb", t2, t2).evaluate()
    tsq_vvvv = einsum("klab,klcd->abcd", t2, t2).evaluate()
    tsq_oooo = einsum("ijcd,klcd->ijkl", t2, t2).evaluate()
    rx = einsum("ijab,jb->ia", t2, u.ph)

    g2a = TwoParticleDensityMatrix(hf)
    g2a.oooo = (
        + 2.0 * einsum("ijab,klab->ijkl", u.pphh, u.pphh)
        + 0.5 * 2.0 * (
            + 2.0 * einsum("ijab,klab->ijkl", tbarD, t2)
            + 0.5 * 2.0 * einsum("jm,imkl->ijkl", g1a_adc1.oo, tsq_oooo).antisymmetrise(0, 1)
            + 1.0 * 2.0 * einsum("kc,ijbc,ma,lmab->ijkl", u.ph, t2, u.ph, t2).antisymmetrise(2, 3)
            + 1.0 * 4.0 * (
                + 1.0 * einsum("ik,jl->ijkl", rho.oo, g1a_adc1.oo)
                - 1.0 * einsum("lajb,ia,kb->ijkl", tsq_ovov, u.ph, u.ph)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ).symmetrise([(0, 1), (2, 3)])
    )
    g2a.ooov = (
        - 2.0 * einsum("kb,ijab->ijka", u.ph, u.pphh)
        + 1.0 * einsum("la,ijbc,klbc->ijka", u.ph, t2, u.pphh)
        + 1.0 * 2.0 * (
            + 2.0 * einsum("ic,jlab,klbc->ijka", u.ph, t2, u.pphh)
            - 1.0 * einsum("ja,ilbc,klbc->ijka", u.ph, t2, u.pphh)
            + 1.0 * einsum("ja,ik->ijka", rho.ov, g1a_adc1.oo)
            + 1.0 * einsum("ia,jb,kb->ijka", u.ph, rho.ov, u.ph)
        ).antisymmetrise(0, 1)
        - 0.5 * einsum("ijab,kb->ijka", t2, rho_bar)
    )
    g2a.oovv = 0.5 * (
        - 1.0 * t2 - 1.0 * tD - 4.0 * t2bar
        - 1.0 * 2.0 * (
            + 1.0 * einsum("ijbc,ac->ijab", tD + t2, g1a_adc1.vv)
        ).antisymmetrise(2, 3)
        + 1.0 * 2.0 * (
            + 1.0 * einsum("jkab,ik->ijab", tD + t2, g1a_adc1.oo)
        ).antisymmetrise(0, 1)
        - 1.0 * 4.0 * (
            + 1.0 * einsum("ia,jb->ijab", u.ph, einsum("jkbc,kc->jb", tD, u.ph) + rx)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    g2a.ovov = (
        + 1.0 * g2a_adc1.ovov
        - 4.0 * einsum("ikbc,jkac->iajb", u.pphh, u.pphh)
        + 1.0 * einsum("ij,ab->iajb", rho.oo, g1a_adc1.vv)
        + 1.0 * einsum("ab,ij->iajb", rho.vv, g1a_adc1.oo)
        + 0.5 * 2.0 * (
            - 4.0 * einsum("ikbc,jkac->iajb", t2, tbarD)
            + 1.0 * einsum("ibjc,ac->iajb", tsq_ovov, g1a_adc1.vv)
            - 1.0 * einsum("ibka,jk->iajb", tsq_ovov, g1a_adc1.oo)
            + 1.0 * einsum("ka,ikbc,jc->iajb", u.ph, t2, rx)
            + 1.0 * einsum("jc,ikbc,ka->iajb", u.ph, t2, rx)
            + 1.0 * einsum("ja,cb,ic->iajb", u.ph, rho.vv, u.ph)
            - 1.0 * einsum("ib,kj,ka->iajb", u.ph, rho.oo, u.ph)
            - 2.0 * einsum("jckb,ic,ka->iajb", tsq_ovov, u.ph, u.ph)
            + 0.5 * einsum("acbd,ic,jd->iajb", tsq_vvvv, u.ph, u.ph)
            + 0.5 * einsum("ikjl,ka,lb->iajb", tsq_oooo, u.ph, u.ph)
        ).symmetrise([(0, 2), (1, 3)])  # TODO: symmetrise correct?
    )
    g2a.ovvv = (
        - 2.0 * einsum("ja,ijbc->iabc", u.ph, u.pphh)
        + 1.0 * einsum("id,jkbc,jkad->iabc", u.ph, t2, u.pphh)
        + 1.0 * 2.0 * (
            - 2.0 * einsum("jb,ikcd,jkad->iabc", u.ph, t2, u.pphh)
            + 1.0 * einsum("ib,jkcd,jkad->iabc", u.ph, t2, u.pphh)
            - 1.0 * einsum("ic,ab->iabc", rho.ov, g1a_adc1.vv)
            + 1.0 * einsum("ib,jc,ja->iabc", u.ph, rho.ov, u.ph)
        ).antisymmetrise(2, 3)
        - 0.5 * einsum("ijbc,ja->iabc", t2, rho_bar)
    )
    g2a.vvvv = (
        + 2.0 * einsum("ijcd,ijab->abcd", u.pphh, u.pphh)
        + 0.5 * 2.0 * (
            + 2.0 * einsum("ijab,ijcd->abcd", tbarD, t2)
            + 0.5 * 2.0 * (
                - 1.0 * einsum("be,aecd->abcd", g1a_adc1.vv, tsq_vvvv)
            ).antisymmetrise(0, 1)
            + 1.0 * 2.0 * (
                + 1.0 * einsum("ia,ijcd,jkbe,ke->abcd", u.ph, t2, t2, u.ph)  # TODO: not sure...
            ).antisymmetrise(0, 1)
            + 1.0 * 4.0 * (
                + 1.0 * einsum("bd,ac->abcd", rho.vv, g1a_adc1.vv)
                - 1.0 * einsum("idjb,ia,jc->abcd", tsq_ovov, u.ph, u.ph)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ).symmetrise([(0, 1), (2, 3)])
    )
    return g1a, g2a 


### CVS ###

def ampl_relaxed_dms_cvs_adc0(exci):
    hf = exci.reference_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    # g2a is not required for cvs-adc0 gradient,
    # but expected by amplitude_relaxed_densities
    g1a.cc = - 1.0 * einsum("Ia,Ja->IJ", u.ph, u.ph)
    g1a.vv = + 1.0 * einsum("Ia,Ib->ab", u.ph, u.ph)
    return g1a, g2a


def ampl_relaxed_dms_cvs_adc1(exci):
    hf = exci.reference_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    g2a.cvcv = - 1.0 * einsum("Ja,Ib->IaJb", u.ph, u.ph)
    g1a.cc = - 1.0 * einsum("Ia,Ja->IJ", u.ph, u.ph)
    g1a.vv = + 1.0 * einsum("Ia,Ib->ab", u.ph, u.ph)

    # Prerequisites for the OC block of the
    # orbital response Lagrange multipliers:
    fc = hf.fock(b.cc).diagonal()
    fo = hf.fock(b.oo).diagonal()
    fco = direct_sum("-j+I->jI", fc, fo).evaluate()
    g1a.co = - 1.0 * einsum('JbKc,ibKc->Ji', g2a.cvcv, hf.ovcv) / fco
    return g1a, g2a


def ampl_relaxed_dms_cvs_adc2(exci):
    hf = exci.reference_state
    mp = exci.ground_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)

    # Determine the t-amplitudes and multipliers:
    t2oovv = mp.t2(b.oovv)
    t2ccvv = mp.t2(b.ccvv)
    t2ocvv = mp.t2(b.ocvv)
    g1a_cvs0, g2a_cvs0 = ampl_relaxed_dms_cvs_adc0(exci)
    t2bar = t2bar_oovv_cvs_adc2(exci, g1a_cvs0).evaluate()

    g1a.cc = (
        - 1.0 * einsum("Ia,Ja->IJ", u.ph, u.ph)
        - 1.0 * einsum("kJba,kIba->IJ", u.pphh, u.pphh)
        - 0.5 * einsum('IKab,JKab->IJ', t2ccvv, t2ccvv)
        - 0.5 * einsum('kIab,kJab->IJ', t2ocvv, t2ocvv)
    )

    g1a.oo = (
        - 1.0 * einsum("jKba,iKba->ij", u.pphh, u.pphh)
        - 2.0 * einsum("ikab,jkab->ij", t2bar, t2oovv).symmetrise((0, 1))
        - 0.5 * einsum('iKab,jKab->ij', t2ocvv, t2ocvv)
        - 0.5 * einsum('ikab,jkab->ij', t2oovv, t2oovv)
    )

    # Pre-requisites for the OC block of the
    # orbital response Lagrange multipliers:
    fc = hf.fock(b.cc).diagonal()
    fo = hf.fock(b.oo).diagonal()
    fco = direct_sum("-j+I->jI", fc, fo).evaluate()

    g1a.vv = (
        + 1.0 * einsum("Ia,Ib->ab", u.ph, u.ph)
        + 2.0 * einsum('jIcb,jIca->ab', u.pphh, u.pphh)
        + 2.0 * einsum('ijac,ijbc->ab', t2bar, t2oovv).symmetrise((0, 1))
        + 0.5 * einsum('IJac,IJbc->ab', t2ccvv, t2ccvv)
        + 0.5 * einsum('ijac,ijbc->ab', t2oovv, t2oovv)
        + 1.0 * einsum('iJac,iJbc->ab', t2ocvv, t2ocvv)
    )

    g2a.cvcv = (
        - einsum("Ja,Ib->IaJb", u.ph, u.ph)
    )

    # The factor 1/sqrt(2) is needed because of the scaling used in adcc
    # for the ph-pphh blocks.
    g2a.occv = (1 / sqrt(2)) * (
        2.0 * einsum('Ib,kJba->kJIa', u.ph, u.pphh)
    )

    g2a.oovv = (
        + 1.0 * einsum('ijcb,ca->ijab', t2oovv, g1a_cvs0.vv).antisymmetrise((2, 3))
        - 1.0 * t2oovv
        - 2.0 * t2bar
    )

    # The factor 2/sqrt(2) is necessary because of the way
    # that the ph-pphh is scaled.
    g2a.ovvv = (2 / sqrt(2)) * (
        einsum('Ja,iJcb->iabc', u.ph, u.pphh)
    )

    g2a.ccvv = - 1.0 * t2ccvv
    g2a.ocvv = - 1.0 * t2ocvv

    # This is the OC block of the orbital response
    # Lagrange multipliers (lambda):
    g1a.co = (
        - 1.0 * einsum('JbKc,ibKc->Ji', g2a.cvcv, hf.ovcv)
        - 0.5 * einsum('JKab,iKab->Ji', g2a.ccvv, hf.ocvv)
        + 1.0 * einsum('kJLa,ikLa->Ji', g2a.occv, hf.oocv)
        + 0.5 * einsum('kJab,ikab->Ji', g2a.ocvv, hf.oovv)
        - 1.0 * einsum('kLJa,kLia->Ji', g2a.occv, hf.ocov)
        + 1.0 * einsum('iKLa,JKLa->Ji', g2a.occv, hf.cccv)
        + 0.5 * einsum('iKab,JKab->Ji', g2a.ocvv, hf.ccvv)
        - 0.5 * einsum('ikab,kJab->Ji', g2a.oovv, hf.ocvv)
        + 0.5 * einsum('iabc,Jabc->Ji', g2a.ovvv, hf.cvvv)
    ) / fco

    return g1a, g2a


def ampl_relaxed_dms_cvs_adc2x(exci):
    hf = exci.reference_state
    mp = exci.ground_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)

    # Determine the t-amplitudes and multipliers:
    t2oovv = mp.t2(b.oovv)
    t2ccvv = mp.t2(b.ccvv)
    t2ocvv = mp.t2(b.ocvv)
    g1a_cvs0, _ = ampl_relaxed_dms_cvs_adc0(exci)
    t2bar = t2bar_oovv_cvs_adc2(exci, g1a_cvs0).evaluate()

    g1a.cc = (
        - 1.0 * einsum("Ia,Ja->IJ", u.ph, u.ph)
        - 1.0 * einsum("kJba,kIba->IJ", u.pphh, u.pphh)
        - 0.5 * einsum('IKab,JKab->IJ', t2ccvv, t2ccvv)
        - 0.5 * einsum('kIab,kJab->IJ', t2ocvv, t2ocvv)
    )

    g1a.oo = (
        - 1.0 * einsum("jKba,iKba->ij", u.pphh, u.pphh)
        - 2.0 * einsum("ikab,jkab->ij", t2bar, t2oovv).symmetrise((0, 1))
        - 0.5 * einsum('iKab,jKab->ij', t2ocvv, t2ocvv)
        - 0.5 * einsum('ikab,jkab->ij', t2oovv, t2oovv)
    )

    # Pre-requisites for the OC block of the
    # orbital response Lagrange multipliers:
    fc = hf.fock(b.cc).diagonal()
    fo = hf.fock(b.oo).diagonal()
    fco = direct_sum("-j+I->jI", fc, fo).evaluate()

    g1a.vv = (
        + 1.0 * einsum("Ia,Ib->ab", u.ph, u.ph)
        + 2.0 * einsum('jIcb,jIca->ab', u.pphh, u.pphh)
        + 2.0 * einsum('ijac,ijbc->ab', t2bar, t2oovv).symmetrise((0, 1))
        + 0.5 * einsum('IJac,IJbc->ab', t2ccvv, t2ccvv)
        + 0.5 * einsum('ijac,ijbc->ab', t2oovv, t2oovv)
        + 1.0 * einsum('iJac,iJbc->ab', t2ocvv, t2ocvv)
    )

    g2a.cvcv = (
        - 1.0 * einsum("Ja,Ib->IaJb", u.ph, u.ph)
        - 1.0 * einsum('kIbc,kJac->IaJb', u.pphh, u.pphh)
        + 1.0 * einsum('kIcb,kJac->IaJb', u.pphh, u.pphh)
    )

    # The factor 1/sqrt(2) is needed because of the scaling used in adcc
    # for the ph-pphh blocks.
    g2a.occv = (1 / sqrt(2)) * (
        2.0 * einsum('Ib,kJba->kJIa', u.ph, u.pphh)
    )

    g2a.oovv = (
        + 1.0 * einsum('ijcb,ca->ijab', t2oovv, g1a_cvs0.vv).antisymmetrise((2, 3))
        - 1.0 * t2oovv
        - 2.0 * t2bar
    )

    # The factor 2/sqrt(2) is necessary because of
    # the way that the ph-pphh is scaled
    g2a.ovvv = (2 / sqrt(2)) * (
        einsum('Ja,iJcb->iabc', u.ph, u.pphh)
    )

    g2a.ovov = 1.0 * (
        - einsum("iKbc,jKac->iajb", u.pphh, u.pphh)
        + einsum("iKcb,jKac->iajb", u.pphh, u.pphh)
    )

    g2a.ccvv = - 1.0 * t2ccvv
    g2a.ocvv = - 1.0 * t2ocvv
    g2a.ococ = 1.0 * einsum("iJab,kLab->iJkL", u.pphh, u.pphh)
    g2a.vvvv = 1.0 * einsum("iJcd,iJab->abcd", u.pphh, u.pphh)

    # TODO: remove
    # g2a.ococ *= 0.0
    # g2a.vvvv *= 0.0

    g1a.co = (
        - 1.0 * einsum('JbKc,ibKc->Ji', g2a.cvcv, hf.ovcv)
        - 0.5 * einsum('JKab,iKab->Ji', g2a.ccvv, hf.ocvv)
        + 1.0 * einsum('kJLa,ikLa->Ji', g2a.occv, hf.oocv)
        + 0.5 * einsum('kJab,ikab->Ji', g2a.ocvv, hf.oovv)
        - 1.0 * einsum('kLJa,kLia->Ji', g2a.occv, hf.ocov)
        + 1.0 * einsum('iKLa,JKLa->Ji', g2a.occv, hf.cccv)
        + 0.5 * einsum('iKab,JKab->Ji', g2a.ocvv, hf.ccvv)
        - 0.5 * einsum('ikab,kJab->Ji', g2a.oovv, hf.ocvv)
        + 0.5 * einsum('iabc,Jabc->Ji', g2a.ovvv, hf.cvvv)
        + 1.0 * einsum('kJmL,ikmL->Ji', g2a.ococ, hf.oooc)
        - 1.0 * einsum('iKlM,JKMl->Ji', g2a.ococ, hf.ccco)
        + 1.0 * einsum('iakb,kbJa->Ji', g2a.ovov, hf.ovcv)
    ) / fco

    return g1a, g2a


DISPATCH = {
    "mp2":  ampl_relaxed_dms_mp2,
    "adc0": ampl_relaxed_dms_adc0,
    "adc1": ampl_relaxed_dms_adc1,
    "adc2": ampl_relaxed_dms_adc2,
    "adc2x": ampl_relaxed_dms_adc2x,
    "adc3": ampl_relaxed_dms_adc3,
    "cvs-adc0": ampl_relaxed_dms_cvs_adc0,
    "cvs-adc1": ampl_relaxed_dms_cvs_adc1,
    "cvs-adc2": ampl_relaxed_dms_cvs_adc2,
    "cvs-adc2x": ampl_relaxed_dms_cvs_adc2x,
}


def amplitude_relaxed_densities(excitation_or_mp):
    """Computation of amplitude-relaxed one- and two-particle density matrices

    Parameters
    ----------
    excitation_or_mp : LazyMp, Excitation
        Data for which the densities are requested, either LazyMp for ground
        state densities or Excitation for excited state densities

    Returns
    -------
    (OneParticleOperator, TwoParticleDensityMatrix)
        Tuple of amplitude-relaxed one- and two-particle density matrices

    Raises
    ------
    NotImplementedError
        if density matrices are not implemented for a given method
    """
    if isinstance(excitation_or_mp, LazyMp):
        method_name = "mp2"
    elif isinstance(excitation_or_mp, Excitation):
        method_name = excitation_or_mp.method.name
    if method_name not in DISPATCH:
        raise NotImplementedError("Amplitude response is not "
                                  f"implemented for {method_name}.")
    g1a, g2a = DISPATCH[method_name](excitation_or_mp)
    return evaluate(g1a), evaluate(g2a)
