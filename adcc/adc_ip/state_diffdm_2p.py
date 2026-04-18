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
from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum, zeros_like
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.TwoParticleDensity import TwoParticleDensity
from adcc.NParticleOperator import OperatorSymmetry

from .util import check_doubles_amplitudes, check_singles_amplitudes
from math import sqrt


def diffdm_ip_adc0_2p(mp, amplitude, intermediates):
    check_singles_amplitudes([b.o], amplitude)
    u1 = amplitude.h

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    dm = TwoParticleDensity(mp, symmetry=OperatorSymmetry.HERMITIAN)

    dm.oooo = (
        + 4.0 * (
            # N^4: O^4 / N^4: O^4
            + 1 * einsum("il,jk->ijkl", einsum("i,l->il", u1, u1), d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    return dm


def diffdm_ip_adc1_2p(mp, amplitude, intermediates):
    dm = diffdm_ip_adc0_2p(mp, amplitude, intermediates)  # Get ADC(0) result
    u1 = amplitude.h

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)

    dm.oovv += (
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2.0 * einsum("iab,j->ijab", einsum("k,ikab->iab", u1, t2), u1).antisymmetrise(0, 1)
    )
    return dm


def diffdm_ip_adc2_2p(mp, amplitude, intermediates):
    dm = diffdm_ip_adc1_2p(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v], amplitude)
    u1, u2 = amplitude.h, amplitude.phh
    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    dm.oooo += (
        # N^5: O^4V^1 / N^4: O^4
        + 2.0 * einsum("ija,kla->ijkl", u2, u2)
        + 4.0 * (
            # N^4: O^4 / N^4: O^4
            + 1.0 * einsum("il,jk->ijkl", einsum("i,l->il", u1, u1), p0.oo)
            # N^4: O^3V^1 / N^4: O^4
            - 2.0 * einsum("ik,jl->ijkl", einsum("ima,kma->ik", u2, u2), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ik,jl->ijkl", einsum("kab,iab->ik", einsum("n,knab->kab", u1, t2), einsum("m,imab->iab", u1, t2)), d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1.0 * (
            2.0 * einsum("jkl,i->ijkl", einsum("jab,klab->jkl", einsum("m,jmab->jab", u1, t2), t2), u1)
        ).antisymmetrise(0, 1).symmetrise([(0, 2), (1, 3)])
        + 1.0 * (
            # N^4: O^4 / N^4: O^4
            + 4.0 * einsum("il,jk->ijkl", einsum("l,i->il", einsum("m,lm->l", u1, p0.oo), u1), d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3).symmetrise([(0, 2), (1, 3)])
    )
    dm.ooov += (
        # N^4: O^3V^1 / N^4: O^3V^1
        + sqrt(2) * einsum("k,ija->ijka", u1, u2)
        # N^5: O^3V^2 / N^4: O^2V^2
        - sqrt(2) * einsum("kb,ijab->ijka", einsum("l,klb->kb", u1, u2), t2)
        + 2.0 * (
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1.0 * einsum("jk,ia->ijka", einsum("j,k->jk", u1, u1), p0.ov)
            # N^5: O^3V^2 / N^4: O^2V^2
            + sqrt(2) * einsum("ika,j->ijka", einsum("klb,ilab->ika", u2, t2), u1)
            # N^4: O^3V^1 / N^4: O^3V^1
            + sqrt(2) * einsum("ja,ik->ijka", einsum("l,jla->ja", u1, u2), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1.0 * einsum("ia,jk->ijka", einsum("a,i->ia", einsum("l,la->a", u1, p0.ov), u1), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + sqrt(2) * einsum("ja,ik->ijka", einsum("mb,jmab->ja", einsum("l,lmb->mb", u1, u2), t2), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * sqrt(2) * einsum("ia,jk->ijka", einsum("a,i->ia", einsum("lmb,lmab->a", u2, t2), u1), d_oo)
        ).antisymmetrise(0, 1)
    )
    dm.oovv += (
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2.0 * einsum("iab,j->ijab", einsum("k,ikab->iab", u1, td2), u1).antisymmetrise(0, 1)
    )
    dm.ovov += (
        # N^5: O^3V^2 / N^4: O^2V^2
        - 2.0 * einsum("jka,ikb->iajb", u2, u2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", einsum("kla,klb->ab", u2, u2), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1.0 * einsum("ij,ab->iajb", einsum("i,j->ij", u1, u1), p0.vv)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 1.0 * einsum("ibc,jac->iajb", einsum("l,ilbc->ibc", u1, t2), einsum("k,jkac->jac", u1, t2))
        # N^4: O^1V^3 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", einsum("lbc,lac->ab", einsum("m,lmbc->lbc", u1, t2), einsum("k,klac->lac", u1, t2)), d_oo)
        # N^5: O^2V^3 / N^4: O^2V^2
        - 2.0 * einsum("jab,i->iajb", einsum("kbc,jkac->jab", einsum("l,klbc->kbc", u1, t2), t2), u1).symmetrise([(0, 2), (1, 3)])
    )
    dm.ovvv += (
        # N^5: O^2V^3 / N^4: O^1V^3
        - sqrt(2) * einsum("ja,ijbc->iabc", einsum("k,jka->ja", u1, u2), t2)
        # N^5: O^2V^3 / N^4: O^1V^3
        - 0.5 * sqrt(2) * einsum("abc,i->iabc", einsum("jka,jkbc->abc", u2, t2), u1)
    )
    dm.vvvv += (
        # N^5: O^1V^4 / N^4: V^4
        - 1.0 * einsum("kcd,kab->abcd", einsum("j,jkcd->kcd", u1, t2), einsum("i,ikab->kab", u1, t2))
    )
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "ip-adc0": diffdm_ip_adc0_2p,
    "ip-adc1": diffdm_ip_adc1_2p,
    "ip-adc2": diffdm_ip_adc2_2p,
    "ip-adc2x": diffdm_ip_adc2_2p,      # same as ADC(2)
}


def state_diffdm_2p(method, ground_state, amplitude, intermediates=None):
    """
    Compute the two-particle difference density matrix of an excited state
    in the MO basis.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude : AmplitudeVector
        The amplitude vector
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
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
