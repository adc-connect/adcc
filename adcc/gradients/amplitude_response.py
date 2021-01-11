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


def ampl_relaxed_dms_adc1(exci):
    hf = exci.reference_state
    u = exci.excitation_vector
    g1a = OneParticleOperator(hf)
    g2a = TwoParticleDensityMatrix(hf)
    g1a.oo = -1.0 * einsum("ia,ja->ij", u.ph, u.ph)
    g1a.vv = +1.0 * einsum("ia,ib->ab", u.ph, u.ph)
    g2a.ovov = -1.0 * einsum("ja,ib->iajb", u.ph, u.ph)
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
    "adc1": ampl_relaxed_dms_adc1,
    "adc2": ampl_relaxed_dms_adc2,
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
