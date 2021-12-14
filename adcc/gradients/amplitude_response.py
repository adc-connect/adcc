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
    t2bar = (
        0.5 * (
            hf.oovv
            - 2.0 * einsum(
                "ijcb,ac->ijab", hf.oovv, g1a_adc0.vv
            ).antisymmetrise((2, 3))
            + 2.0 * einsum(
                "kjab,ik->ijab", hf.oovv, g1a_adc0.oo
            ).antisymmetrise((0, 1))
            + 4.0 * einsum(
                "ia,jkbc,kc->ijab", u.ph, hf.oovv, u.ph
            ).antisymmetrise((2, 3)).antisymmetrise((0, 1))
        ) / (
            2.0 * direct_sum("ia+jb->ijab", df_ia, df_ia).symmetrise((0, 1))
        )
    )
    return t2bar


def t2bar_oovv_cvs_adc2(exci, g1a_adc0):
    mp = exci.ground_state
    hf = mp.reference_state
    df_ia = mp.df(b.ov)
    t2bar = 0.5 * (
        - einsum("ijcb,ac->ijab", hf.oovv, g1a_adc0.vv).antisymmetrise((2, 3))
    ) / direct_sum("ia+jb->ijab", df_ia, df_ia).symmetrise((0, 1))
    return t2bar


def ampl_relaxed_dms_adc1(exci):
    hf = exci.reference_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    g1a.oo = - 1.0 * einsum("ia,ja->ij", u.ph, u.ph)
    g1a.vv = + 1.0 * einsum("ia,ib->ab", u.ph, u.ph)
    g2a.ovov = - 1.0 * einsum("ja,ib->iajb", u.ph, u.ph)
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
    # These are the multipliers:
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
    g2a.vvvv = 2.0 * einsum("iJcd,iJab->abcd", u.pphh, u.pphh)

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


def ampl_relaxed_dms_mp2(mp):
    hf = mp.reference_state
    t2 = mp.t2(b.oovv)
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    g1a.oo = -0.5 * einsum('ikab,jkab->ij', t2, t2)
    g1a.vv = 0.5 * einsum('ijac,ijbc->ab', t2, t2)
    g2a.oovv = -1.0 * mp.t2(b.oovv)
    return g1a, g2a


DISPATCH = {
    "mp2":  ampl_relaxed_dms_mp2,
    "adc0": ampl_relaxed_dms_adc0,
    "adc1": ampl_relaxed_dms_adc1,
    "adc2": ampl_relaxed_dms_adc2,
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
