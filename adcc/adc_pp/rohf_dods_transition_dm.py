## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from .util import check_doubles_amplitudes, check_singles_amplitudes


def tdm_isr0(ground_state, amplitude, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude)

    ul1 = amplitude.ph

    dm = OneParticleDensity(ground_state, symmetry=OperatorSymmetry.NOSYMMETRY)
    dm.vo = +1 * einsum("ia->ai", ul1)  # N^2: O^1V^1 / N^2: O^1V^1
    return dm


def tdm_isr1(ground_state, amplitude, intermediates):
    dm = tdm_isr0(ground_state, amplitude, intermediates)

    ul1 = amplitude.ph

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)

    # only contributions from first order ph -> isr1s = isr1
    dm.oo += -1 * einsum("ja,ia->ij", ul1, t1_1)  # N^3: O^2V^1 / N^2: O^1V^1
    dm.ov += -1 * einsum("jb,ijab->ia", ul1, t2_1)  # N^4: O^2V^2 / N^4: O^2V^2
    dm.vv += +1 * einsum("ia,ib->ab", ul1, t1_1)  # N^3: O^1V^2 / N^2: V^2
    return dm


def tdm_isr2(ground_state, amplitude, intermediates):
    dm = tdm_isr1(ground_state, amplitude, intermediates)
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)

    ul1, ul2 = amplitude.ph, amplitude.pphh

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)
    t1_2 = ground_state.ts2(b.ov)
    t2_2 = ground_state.td2(b.oovv)

    dm.oo += (
        # second order ph
        - 1 * einsum("ja,ia->ij", ul1, t1_2)  # N^3: O^2V^1 / N^2: O^1V^1
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ia,ja->ij", einsum("kb,ikab->ia", t1_1, t2_1), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ul1, t2_1), t1_1)
        # first order pphh
        - 1 * einsum("jkab,ikab->ij", ul2, t2_1)  # N^5: O^3V^2 / N^4: O^2V^2
    )  # fmt: skip
    dm.ov += (
        # second order ph
        - 1 * einsum("jb,ijab->ia", ul1, t2_2)  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("jb,jb->", ul1, t1_1) * t1_1  # N^2: O^1V^1 / N^2: O^1V^1
    )  # fmt: skip
    dm.vo += (
        # second order ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ijab->ai", einsum("kc,jkbc->jb", ul1, t2_1), t2_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai", einsum("ib,jb->ij", t1_1, t1_1), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai", einsum("ib,jb->ij", ul1, t1_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ik,ka->ai", einsum("ijbc,jkbc->ik", t2_1, t2_1), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ijkb,jkab->ai", einsum("ic,jkbc->ijkb", ul1, t2_1), t2_1)
    )  # fmt: skip
    dm.vv += (
        # second order ph
        + 1 * einsum("ia,ib->ab", ul1, t1_2)  # N^3: O^1V^2 / N^2: V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ja->ab", einsum("ic,ijbc->jb", ul1, t2_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ib,ia->ab", einsum("jc,ijbc->ib", t1_1, t2_1), ul1)
        # first order pphh
        + 1 * einsum("ijac,ijbc->ab", ul2, t2_1)  # N^5: O^2V^3 / N^4: O^2V^2
    )  # fmt: skip
    return dm


def tdm_isr3(ground_state, amplitude, intermediates):
    dm = tdm_isr2(ground_state, amplitude, intermediates)

    ul1, ul2 = amplitude.ph, amplitude.pphh

    t1_1 = ground_state.ts1(b.ov)
    t2_1 = ground_state.td1(b.oovv)
    t1_2 = ground_state.ts2(b.ov)
    t2_2 = ground_state.td2(b.oovv)
    t3_2 = ground_state.tt2(b.ooovvv)
    t1_3 = ground_state.ts3(b.ov)
    t2_3 = ground_state.td3(b.oovv)

    # no triples and quadruples contributions -> isr3d = isr3t = isr3
    dm.oo += (
        # third order ph
        - 1 * einsum("ja,ia->ij", ul1, t1_3)  # N^3: O^2V^1 / N^2: O^1V^1
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("kb,kb->", t1_1, t1_1) * einsum("ja,ia->ij", ul1, t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ia,ja->ij", einsum("kb,ikab->ia", t1_1, t2_2), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("ia,ja->ij", einsum("ikab,kb->ia", t2_1, t1_2), ul1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.5 * einsum("ikbc,jkbc->ij", einsum("la,iklabc->ikbc", ul1, t3_2), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ul1, t2_2), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ul1, t2_1), t1_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.25 * einsum("klbc,klbc->", t2_1, t2_1) * einsum("ja,ia->ij", ul1, t1_1)
        # N^6: O^3V^3 / N^6: O^3V^3
        + 0.25 * einsum("ia,ja->ij", einsum("klbc,iklabc->ia", t2_1, t3_2), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        + 1 * einsum("ka,ka->", ul1, t1_1) * einsum("ib,jb->ij", t1_1, t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("la,la->", ul1, t1_1) * einsum("ikbc,jkbc->ij", t2_1, t2_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ik,jk->ij", einsum("ib,kb->ik", t1_1, t1_1), einsum("ja,ka->jk", ul1, t1_1))
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("jk,ik->ij", einsum("jb,kb->jk", t1_1, t1_1), einsum("ka,ia->ik", ul1, t1_1))
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("jb,ib->ij",
                       einsum("kc,jkbc->jb", einsum("la,klac->kc", ul1, t2_1), t2_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("jb,ib->ij",
                        einsum("jklc,klbc->jb", einsum("ja,klac->jklc", ul1, t2_1), t2_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("jl,il->ij",
                        einsum("jkbc,klbc->jl", t2_1, t2_1), einsum("la,ia->il", ul1, t1_1))
        # second order pphh
        - 1 * einsum("jkab,ikab->ij", ul2, t2_2)  # N^5: O^3V^2 / N^4: O^2V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        + 2 * einsum("jb,ib->ij", einsum("jkab,ka->jb", ul2, t1_1), t1_1)
    )  # fmt: skip
    dm.ov += (
        # third order ph
        - 1 * einsum("jb,ijab->ia", ul1, t2_3)  # N^4: O^2V^2 / N^4: O^2V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ijab->ia", einsum("kc,kc->", t1_1, t1_1) * ul1, t2_1)
        - 1 * einsum("jb,jb->", ul1, t1_2) * t1_1  # N^2: O^1V^1 / N^2: O^1V^1
        - 1 * einsum("jb,jb->", ul1, t1_1) * t1_2  # N^2: O^1V^1 / N^2: O^1V^1
        # N^6: O^3V^3 / N^6: O^3V^3
        - 1 * einsum("jkbc,ijkabc->ia", einsum("jc,kb->jkbc", ul1, t1_1), t3_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kc,kc->", einsum("jb,jkbc->kc", ul1, t2_1), t1_1) * t1_1
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("kb,ikab->ia", einsum("jc,jc->", ul1, t1_1) * t1_1, t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ijkc,jkac->ia",
                       einsum("ib,jkbc->ijkc", einsum("ld,ilbd->ib", ul1, t2_1), t2_1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum("ij,ja->ia",
                       einsum("ikbc,jkbc->ij", t2_1, t2_1), einsum("ld,jlad->ja", ul1, t2_1))
        # N^6: O^4V^2 / N^4: O^2V^2
        - 1 * einsum("ijlc,jlac->ia",
                     einsum("iklb,jkbc->ijlc", einsum("ld,ikbd->iklb", ul1, t2_1), t2_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("jb,ijab->ia",
                       einsum("jk,kb->jb", einsum("jc,kc->jk", t1_1, t1_1), ul1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("jb,ijab->ia",
                       einsum("jk,kb->jb", einsum("jc,kc->jk", ul1, t1_1), t1_1), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum("kb,ikab->ia",
                       einsum("ld,klbd->kb", einsum("jc,jlcd->ld", ul1, t2_1), t2_1), t2_1)
        # N^6: O^4V^2 / N^4: O^2V^2
        - 0.25 * einsum("ijkl,jkla->ia",
                        einsum("ilbc,jkbc->ijkl", t2_1, t2_1), einsum("ld,jkad->jkla", ul1, t2_1))
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("jb,ijab->ia",
                        einsum("jk,kb->jb", einsum("jlcd,klcd->jk", t2_1, t2_1), ul1), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("lc,ilac->ia",
                        einsum("jklb,jkbc->lc", einsum("ld,jkbd->jklb", ul1, t2_1), t2_1), t2_1)
        # second order pphh
        - 0.5 * einsum("jkbc,ijkabc->ia", ul2, t3_2)  # N^6: O^3V^3 / N^6: O^3V^3
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("jb,ijab->ia", einsum("jkbc,kc->jb", ul2, t1_1), t2_1)
        - 0.5 * einsum("jkbc,jkbc->", ul2, t2_1) * t1_1  # N^4: O^2V^2 / N^4: O^2V^2
    )  # fmt: skip
    dm.vo += (
        # third order ph
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ijab->ai", einsum("kc,jkbc->jb", ul1, t2_2), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,ijab->ai", einsum("kc,jkbc->jb", ul1, t2_1), t2_2)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai", einsum("ib,jb->ij", t1_1, t1_2), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai", einsum("jb,ib->ij", t1_1, t1_2), ul1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai", einsum("ib,jb->ij", ul1, t1_2), t1_1)
        # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("ij,ja->ai", einsum("ib,jb->ij", ul1, t1_1), t1_2)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ik,ka->ai", einsum("ijbc,jkbc->ik", t2_1, t2_2), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ik,ka->ai", einsum("jkbc,ijbc->ik", t2_1, t2_2), ul1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ijkb,jkab->ai", einsum("ic,jkbc->ijkb", ul1, t2_2), t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ijkb,jkab->ai", einsum("ic,jkbc->ijkb", ul1, t2_1), t2_2)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("kc,ikac->ai", einsum("jb,jb->", ul1, t1_1) * t1_1, t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("jb,jb->", einsum("kc,jkbc->jb", ul1, t2_1), t1_1) * einsum("ia->ai", t1_1)
    )  # fmt: skip
    dm.vv += (
        # third order ph
        + 1 * einsum("ia,ib->ab", ul1, t1_3)  # N^3: O^1V^2 / N^2: V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ja->ab", einsum("ic,ijbc->jb", ul1, t2_2), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum("jb,ja->ab", einsum("ic,ijbc->jb", ul1, t2_1), t1_2)
        # N^3: O^1V^2 / N^2: V^2
        - 1 * einsum("ia,ib->ab", einsum("jc,jc->", t1_1, t1_1) * ul1, t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ib,ia->ab", einsum("jc,ijbc->ib", t1_1, t2_2), ul1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ib,ia->ab", einsum("ijbc,jc->ib", t2_1, t1_2), ul1)
        # N^6: O^3V^3 / N^6: O^3V^3
        - 0.5 * einsum("jkbc,jkac->ab", einsum("id,ijkbcd->jkbc", ul1, t3_2), t2_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        - 0.25 * einsum("ia,ib->ab", einsum("jkcd,jkcd->", t2_1, t2_1) * ul1, t1_1)
        # N^6: O^3V^3 / N^6: O^3V^3
        - 0.25 * einsum("ib,ia->ab", einsum("jkcd,ijkbcd->ib", t2_1, t3_2), ul1)
        # N^3: O^1V^2 / N^2: V^2
        + 0.5 * einsum("ja,jb->ab", einsum("ij,ia->ja", einsum("ic,jc->ij", t1_1, t1_1), ul1), t1_1)
        # N^3: O^1V^2 / N^2: V^2
        + 0.5 * einsum("ia,ib->ab", einsum("ij,ja->ia", einsum("ic,jc->ij", ul1, t1_1), t1_1), t1_1)
        # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum("ja,jb->ab",
                       einsum("kc,jkac->ja", einsum("id,ikcd->kc", ul1, t2_1), t2_1), t1_1)
        # N^3: O^1V^2 / N^2: V^2
        - 1 * einsum("ja,jb->ab", einsum("ic,ic->", ul1, t1_1) * t1_1, t1_1)
        # N^5: O^2V^3 / N^4: O^2V^2
        - 0.5 * einsum("id,id->", ul1, t1_1) * einsum("jkac,jkbc->ab", t2_1, t2_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        - 0.25 * einsum("ia,ib->ab",
                        einsum("ijkc,jkac->ia", einsum("id,jkcd->ijkc", ul1, t2_1), t2_1), t1_1)
        # N^5: O^3V^2 / N^4: O^2V^2
        + 0.25 * einsum("ja,jb->ab",
                        einsum("ij,ia->ja", einsum("ikcd,jkcd->ij", t2_1, t2_1), ul1), t1_1)
        # second order pphh
        + 1 * einsum("ijac,ijbc->ab", ul2, t2_2)  # N^5: O^2V^3 / N^4: O^2V^2
        # N^4: O^2V^2 / N^4: O^2V^2
        - 2 * einsum("ja,jb->ab", einsum("ijac,ic->ja", ul2, t1_1), t1_1)
    )  # fmt: skip
    return dm


DISPATCH = {
    "isr0": tdm_isr0,
    "isr1s": tdm_isr1,  # Identical to ISR(1)
    "isr1": tdm_isr1,
    "isr2d": tdm_isr2,  # Identical to ISR(2)
    "isr2": tdm_isr2,
    "isr3d": tdm_isr3,  # Identical to ISR(3)
    "isr3": tdm_isr3,
}


def rohf_dods_transition_dm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle transition density matrix from ground to excited
    state in the MO basis.

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
        raise NotImplementedError(f"transition_dm is not implemented for {method.name}.")
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
