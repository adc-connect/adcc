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


def tdm_isr0_2p(mp, amplitude, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude)
    u1 = amplitude.ph

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    dm = TwoParticleDensity(mp, symmetry=OperatorSymmetry.NOSYMMETRY)

    # N^4: O^3V^1 / N^4: O^3V^1
    dm.ovoo = 2.0 * einsum("ka,ij->iajk", u1, d_oo).antisymmetrise(2, 3)
    return dm


def tdm_isr1s_2p(mp, amplitude, intermediates):
    dm = tdm_isr0_2p(mp, amplitude, intermediates)  # Get ISR(0) result
    u1 = amplitude.ph

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)

    dm.ooov += (
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1.0 * einsum("kb,ijab->ijka", u1, t2)
        + 2.0 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            einsum("ia,jk->ijka", einsum("lb,ilab->ia", u1, t2), d_oo)
        ).antisymmetrise(0, 1)
    )
    dm.ovvv += (
        # N^5: O^2V^3 / N^4: O^1V^3
        - 1.0 * einsum("ja,ijbc->iabc", u1, t2)
    )

    return dm


def tdm_isr1_2p(mp, amplitude, intermediates):
    dm = tdm_isr1s_2p(mp, amplitude, intermediates)  # Get ISR(1)-s result

    try:
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude, amplitude)
        u2 = amplitude.pphh
        dm.vvoo += (
            # N^4: O^2V^2 / N^4: O^2V^2
            - 2.0 * einsum("ijab->abij", u2)
        )
    except ValueError:
        pass
    return dm


def tdm_isr2_2p(mp, amplitude, intermediates):
    dm = tdm_isr1_2p(mp, amplitude, intermediates)  # Get ADC(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1 = amplitude.ph
    u2 = amplitude.pphh

    hf = mp.reference_state
    d_oo = zeros_like(hf.foo)
    d_oo.set_mask("ii", 1)

    t2 = mp.t2(b.oovv)
    td2 = mp.td2(b.oovv)
    tt2 = mp.tt2(b.ooovvv)
    p0 = mp.mp2_diffdm

    dm.oooo += (
        # N^6: O^4V^2 / N^4: O^2V^2
        + 1.0 * einsum("klab,ijab->ijkl", u2, t2)
        + 4.0 * (
            # N^4: O^4 / N^4: O^4
            + 1.0 * einsum("jk,il->ijkl", einsum("ka,ja->jk", u1, p0.ov), d_oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 1.0 * einsum("jk,il->ijkl", einsum("kmab,jmab->jk", u2, t2), d_oo)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    dm.ooov += (
        # N^5: O^3V^2 / N^4: O^2V^2
        - 1.0 * einsum("kb,ijab->ijka", u1, td2)
        + 2.0 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1.0 * einsum("ia,jk->ijka", einsum("lb,ilab->ia", u1, td2), d_oo)
        ).antisymmetrise(0, 1)
    )
    dm.oovv += (
        # N^6: O^3V^3 / N^6: O^3V^3
        + 1.0 * einsum("kc,ijkabc->ijab", u1, tt2)
    )
    dm.ovoo += (
        2.0 * (
            # N^4: O^3V^1 / N^4: O^3V^1
            + 1 * einsum("ka,ij->iajk", u1, p0.oo)
            # N^6: O^4V^2 / N^4: O^2V^2
            + 1 * einsum("ijlb,klab->iajk", einsum("jc,ilbc->ijlb", u1, t2), t2)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 0.5 * einsum("ka,ij->iajk", einsum("la,kl->ka", u1, p0.oo), d_oo)
            # N^4: O^3V^1 / N^4: O^3V^1
            + 0.5 * einsum("ja,ik->iajk", einsum("jb,ab->ja", u1, p0.vv), d_oo)
            # N^4: O^2V^2 / N^4: O^2V^2
            + 0.5 * einsum("ka,ij->iajk",
                           einsum("lb,klab->ka",
                                  einsum("mc,lmbc->lb", u1, t2), t2), d_oo)
        ).antisymmetrise(2, 3)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 1.0 * einsum("ib,jkab->iajk", einsum("lc,ilbc->ib", u1, t2), t2)
        # N^6: O^4V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijkl,la->iajk", einsum("ilbc,jkbc->ijkl", t2, t2), u1)
    )
    dm.ovov += (
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1.0 * einsum("ja,ib->iajb", u1, p0.ov)
        # N^6: O^3V^3 / N^4: O^2V^2
        - 2.0 * einsum("jkac,ikbc->iajb", u2, t2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", einsum("ka,kb->ab", u1, p0.ov), d_oo)
        # N^5: O^2V^3 / N^4: O^2V^2
        + 1.0 * einsum("ab,ij->iajb", einsum("klac,klbc->ab", u2, t2), d_oo)
    )
    dm.ovvv += (
        # N^5: O^2V^3 / N^4: O^1V^3
        - 1.0 * einsum("ja,ijbc->iabc", u1, td2)
    )
    dm.vvoo += (
        + 4.0 * (
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1.0 * einsum("ia,jb->abij", u1, p0.ov)
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)
    )
    dm.vvov += (
        2.0 * (
            # N^4: O^1V^3 / N^4: O^1V^3
            + 1.0 * einsum("ia,bc->abic", u1, p0.vv)
            # N^6: O^3V^3 / N^4: O^1V^3
            + 1.0 * einsum("ikbc,ka->abic", einsum("ijbd,jkcd->ikbc", t2, t2), u1)
        ).antisymmetrise(0, 1)
        # N^5: O^2V^3 / N^4: O^1V^3
        + 1.0 * einsum("jc,ijab->abic", einsum("kd,jkcd->jc", u1, t2), t2)
        # N^6: O^3V^3 / N^4: O^1V^3
        + 0.5 * einsum("ijkc,jkab->abic", einsum("id,jkcd->ijkc", u1, t2), t2)
    )
    dm.vvvv += (
        # N^6: O^2V^4 / N^4: V^4
        + 1.0 * einsum("ijab,ijcd->abcd", u2, t2)
    )
    return dm


DISPATCH = {
    "isr0": tdm_isr0_2p,
    "isr1s": tdm_isr1s_2p,
    "isr1": tdm_isr1_2p,
    "isr2": tdm_isr2_2p,
    "isr2x": tdm_isr2_2p,
}


def transition_dm_2p(method, ground_state, amplitude, intermediates=None):
    """
    Compute the two-particle transition density matrix from ground to excited
    state in the MO basis.

    Parameters
    ----------
    method : str, AdcMethod
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
        raise NotImplementedError("transition_dm_2p is not implemented "
                                  f"for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
