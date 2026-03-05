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


def diffdm_ea_adc0_2p(mp, amplitude, intermediates):
    check_singles_amplitudes([b.v], amplitude)
    u1 = amplitude.p

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    dm = TwoParticleDensity(mp, symmetry=OperatorSymmetry.HERMITIAN)

    # TODO: Store intermediate?
    # p1_vv = einsum("a,b->ab", u1, u1).evaluate()

    dm.ovov = (
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", einsum("a,b->ab", u1, u1), d_oo)
    )
    return dm


def diffdm_ea_adc1_2p(mp, amplitude, intermediates):
    check_singles_amplitudes([b.v], amplitude)
    u1 = amplitude.p

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    dm = TwoParticleDensity(mp, symmetry=OperatorSymmetry.HERMITIAN)

    # ADC(1) diffdm
    t2 = mp.t2(b.oovv)

    dm.oovv = (
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2.0 * einsum("ija,b->ijab", einsum("c,ijac->ija", u1, t2), u1).antisymmetrise(2, 3)
    )
    return dm


def diffdm_ea_adc2_2p(mp, amplitude, intermediates):
    dm = diffdm_ea_adc1_2p(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.p, amplitude.pph
    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    dm.oooo += (
        4.0 * (
        # N^4: O^2V^2 / N^4: O^4
        + 1.0 * einsum("il,jk->ijkl", einsum("iab,lab->il", u2, u2), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 * einsum("ik,jl->ijkl", einsum("kmc,imc->ik", einsum("b,kmbc->kmc", u1, t2), einsum("a,imac->imc", u1, t2)), d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        # N^5: O^4V^1 / N^4: O^2V^2
        - 1.0 * einsum("klc,ijc->ijkl", einsum("b,klbc->klc", u1, t2), einsum("a,ijac->ijc", u1, t2))
    )
    dm.ooov += (
        2.0 * (
        # N^4: O^3V^1 / N^4: O^3V^1
        + sqrt(2) * einsum("ia,jk->ijka", einsum("b,iab->ia", u1, u2), d_oo)
        # N^4: O^3V^1 / N^4: O^3V^1
        + 1 * einsum("ia,jk->ijka", einsum("i,a->ia", einsum("b,ib->i", u1, p0.ov), u1), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        + sqrt(2) * einsum("ja,ik->ijka", einsum("lb,jlab->ja", einsum("c,lbc->lb", u1, u2), t2), d_oo)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 / sqrt(2) * einsum("ja,ik->ijka", einsum("j,a->ja", einsum("lbc,jlbc->j", u2, t2), u1), d_oo)
        ).antisymmetrise(0, 1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + sqrt(2) * einsum("kb,ijab->ijka", einsum("c,kbc->kb", u1, u2), t2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1.0 / sqrt(2) * einsum("ijk,a->ijka", einsum("kbc,ijbc->ijk", u2, t2), u1)
        
    )
    dm.oovv += (
        2.0 * einsum("ija,b->ijab", einsum("c,ijac->ija", u1, td2), u1).antisymmetrise(2, 3)
    )
    dm.ovov += (
        # N^5: O^2V^3 / N^4: O^2V^2
        - 2.0 * einsum("jac,ibc->iajb", u2, u2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", einsum("a,b->ab", u1, u1), p0.oo)
        # N^4: O^1V^3 / N^4: O^2V^2
        + 2.0 * einsum("ab,ij->iajb", einsum("kac,kbc->ab", u2, u2), d_oo)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1.0 * einsum("ikb,jka->iajb", einsum("d,ikbd->ikb", u1, t2), einsum("c,jkac->jka", u1, t2))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("ab,ij->iajb", einsum("klb,kla->ab", einsum("d,klbd->klb", u1, t2), einsum("c,klac->kla", u1, t2)), d_oo)
        + 2.0 * (
            # N^5: O^3V^2 / N^4: O^2V^2
            - 1 * einsum("ijb,a->iajb", einsum("jkc,ikbc->ijb", einsum("d,jkcd->jkc", u1, t2), t2), u1)
            # N^4: O^2V^2 / N^4: O^2V^2
            - 0.5 * einsum("ab,ij->iajb", einsum("b,a->ab", einsum("c,bc->b", u1, p0.vv), u1), d_oo)
        ).symmetrise([(0, 2), (1, 3)])
    )
    dm.ovvv += (
        (
        # N^4: O^1V^3 / N^4: O^1V^3
        + 1 * einsum("ac,ib->iabc", einsum("a,c->ac", u1, u1), p0.ov)
        # N^5: O^2V^3 / N^4: O^1V^3
        + sqrt(2) * einsum("iac,b->iabc", einsum("jad,ijcd->iac", u2, t2), u1)
        ).antisymmetrise(2, 3)
        # N^4: O^1V^3 / N^4: O^1V^3
        - sqrt(2) * einsum("a,ibc->iabc", u1, u2)
        # N^5: O^2V^3 / N^4: O^1V^3
        + sqrt(2) * einsum("ja,ijbc->iabc", einsum("d,jad->ja", u1, u2), t2)
    )
    dm.vvvv += (
        # N^5: O^1V^4 / N^4: V^4
        + 2.0 * einsum("iab,icd->abcd", u2, u2)
        + 1.0 * (
            # N^4: V^4 / N^4: V^4
            + 4.0 * einsum("ac,bd->abcd", einsum("a,c->ac", u1, u1), p0.vv)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
        # N^5: O^2V^3 / N^4: V^4
        + 1.0 * (
            2.0 * einsum("bcd,a->abcd", einsum("ijb,ijcd->bcd", einsum("e,ijbe->ijb", u1, t2), t2), u1)
        ).antisymmetrise(0, 1).symmetrise([(0, 2), (1, 3)])
    )
    return dm


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "ea-adc0": diffdm_ea_adc0_2p,
    "ea-adc1": diffdm_ea_adc1_2p,
    "ea-adc2": diffdm_ea_adc2_2p,
    "ea-adc2x": diffdm_ea_adc2_2p,      # same as ADC(2)
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
