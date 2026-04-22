#!/usr/bin/env python3
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
from math import sqrt

from adcc import block as b
from adcc.AdcMethod import IsrMethod, AdcMethod
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.LazyMp import LazyMp
from adcc.NParticleOperator import OperatorSymmetry
from adcc.OneParticleDensity import OneParticleDensity


from .util import check_doubles_amplitudes, check_singles_amplitudes


def diffdm_isr0(mp, amplitude, intermediates):
    # C is either c(ore) or o(ccupied)
    C = b.c if mp.has_core_occupied_space else b.o
    check_singles_amplitudes([C, b.v], amplitude)
    u1 = amplitude.ph

    dm = OneParticleDensity(mp, symmetry=OperatorSymmetry.HERMITIAN)
    dm[C + C] = -einsum("ia,ja->ij", u1, u1)
    dm.vv = einsum("ia,ib->ab", u1, u1)
    return dm


def diffdm_isr1(mp, amplitude, intermediates):
    dm = diffdm_isr0(mp, amplitude, intermediates)  # Get ISR(0) result

    try:
        # ISR(1)-d
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
        u1, u2 = amplitude.ph, amplitude.pphh
        dm.ov += -2 * einsum("jb,ijab->ia", u1, u2)
    except ValueError:
        # no doubles contribution
        pass
    return dm


def diffdm_isr2(mp, amplitude, intermediates):
    dm = diffdm_isr1(mp, amplitude, intermediates)  # Get ISR(1) result
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    u1, u2 = amplitude.ph, amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_oo = dm.oo.evaluate()  # ISR(1) diffdm
    p1_vv = dm.vv.evaluate()  # ISR(1) diffdm

    # Zeroth order doubles contributions
    p2_oo = -einsum("ikab,jkab->ij", u2, u2)
    p2_vv = einsum("ijac,ijbc->ab", u2, u2)
    p2_ov = -2 * einsum("jb,ijab->ia", u1, u2).evaluate()

    # ISR(2) intermediate (TODO Move to intermediates)
    ru1 = einsum("ijab,jb->ia", t2, u1).evaluate()

    # Compute second-order contributions to the density matrix
    dm.oo += (  # adc2_p_oo
        2 * p2_oo - einsum("ia,ja->ij", ru1, ru1) + (
            + einsum("ik,kj->ij", p1_oo, p0.oo)
            - einsum("ikcd,jkcd->ij", t2,
                     + 0.5 * einsum("lk,jlcd->jkcd", p1_oo, t2)
                     - einsum("jkcb,db->jkcd", t2, p1_vv))
            - einsum("ia,jkac,kc->ij", u1, t2, ru1)
        ).symmetrise()
    )

    dm.vv += (  # adc2_p_vv
        2 * p2_vv + einsum("ia,ib->ab", ru1, ru1) - (
            + einsum("ac,cb->ab", p1_vv, p0.vv)
            + einsum("klbc,klac->ab", t2,
                     + 0.5 * einsum("klad,cd->klac", t2, p1_vv)
                     - einsum("jk,jlac->klac", p1_oo, t2))
            - einsum("ikac,kc,ib->ab", t2, ru1, u1)
        ).symmetrise()
    )

    dm.ov += (  # adc2_p_ov
        - einsum("ijab,jb->ia", t2, p2_ov)
        - einsum("ib,ba->ia", p0.ov, p1_vv)
        + einsum("ij,ja->ia", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ia", u1, t2, u2)
        - einsum("ikcd,jkcd,ja->ia", t2, u2, u1)
    )
    return dm


def diffdm_cvs_isr1(mp, amplitude, intermediates):
    dm = diffdm_isr0(mp, amplitude, intermediates)  # Get ISR(0) result

    try:
        # ISR(1)-d
        check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
        u1, u2 = amplitude.ph, amplitude.pphh
        p2_ov = -sqrt(2) * einsum("jb,ijab->ia", u1, u2)
        dm.ov += p2_ov
    except ValueError:
        # no doubles contribution
        pass
    return dm


def diffdm_cvs_isr2(mp, amplitude, intermediates):
    dm = diffdm_cvs_isr1(mp, amplitude, intermediates)  # Get cvs-ISR(1) result
    check_doubles_amplitudes([b.o, b.c, b.v, b.v], amplitude)
    u1, u2 = amplitude.ph, amplitude.pphh

    t2 = mp.t2(b.oovv)
    p0 = intermediates.cvs_p0
    p1_vv = dm.vv.evaluate()  # ISR(1) diffdm

    # Zeroth order doubles contributions
    p2_vo = -sqrt(2) * einsum("ijab,jb->ai", u2, u1)
    p2_oo = -einsum("ljab,kjab->kl", u2, u2)
    p2_vv = 2 * einsum("ijac,ijbc->ab", u2, u2)

    # Second order contributions
    # cvs_isr2_dp_oo
    dm.oo += p2_oo + einsum("ab,ikac,jkbc->ij", p1_vv, t2, t2)

    dm.ov += (  # cvs_isr2_dp_ov
        - einsum("ka,ab->kb", p0.ov, p1_vv)
        - einsum("lkdb,dl->kb", t2, p2_vo)
        + 1 / sqrt(2) * einsum("ib,klad,liad->kb", u1, t2, u2)
    )

    dm.vv += p2_vv - 0.5 * (  # cvs_isr2_dp_vv
        + einsum("cb,ac->ab", p1_vv, p0.vv)
        + einsum("cb,ac->ab", p0.vv, p1_vv)
        + einsum("ijbc,ijad,cd->ab", t2, t2, p1_vv)
    )

    # Add 2nd order correction to CVS-ISR(1) diffdm
    dm.cc -= einsum("kIab,kJab->IJ", u2, u2)
    return dm


def diffdm_isr3(mp, amplitude, intermediates):
    dm = diffdm_isr2(mp, amplitude, intermediates)  # starts from ADC2 values
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude)
    ur1, ur2 = amplitude.ph, amplitude.pphh  # ADC amplitudes

    t2_1 = mp.t2(b.oovv)  # first order doubles
    t1_2 = mp.diffdm(level=2).ov  # second order singles
    t2_2 = mp.td2(b.oovv)  # second order doubles
    t3_2 = mp.tt2(b.ooovvv)  # second order triples

    p0_3 = mp.mp3_dm_correction  # third order dm correction
    p0_2 = mp.mp2_dm_correction  # second order dm correction

    dm.oo = (  # adc3_p_oo
        -1 * einsum("ia,ja->ij", ur1, ur1)
        - 2 * einsum("ikab,jkab->ij", ur2, ur2)
        + 1
        * einsum(
            "jklc,iklc->ij",
            einsum("lb,jkbc->jklc", ur1, t2_1),
            einsum("la,ikac->iklc", ur1, t2_1),
        )
        + 0.5
        * einsum(
            "ilbc,jlbc->ij",
            einsum(
                "kl,ikbc->ilbc",
                einsum("ka,la->kl", ur1, ur1), t2_1,
            ), t2_1,
        )
        - 1
        * einsum(
            "jc,ic->ij",
            einsum("lb,jlbc->jc", ur1, t2_1),
            einsum("ka,ikac->ic", ur1, t2_1),
        )
        + 2 * (
            -2 * einsum("ib,jb->ij", einsum("ka,ikab->ib", ur1, ur2), t1_2)
            - 0.5 * einsum("ik,jk->ij", einsum("ia,ka->ik", ur1, ur1), p0_2.oo)
            - 0.5 * einsum("ik,jk->ij", einsum("ia,ka->ik", ur1, ur1), p0_3.oo)
            + 1
            * einsum(
                "jklc,iklc->ij",
                einsum("lb,jkbc->jklc", ur1, t2_2),
                einsum("la,ikac->iklc", ur1, t2_1),
            )
            + 0.5
            * einsum(
                "ilbc,jlbc->ij",
                einsum("kl,ikbc->ilbc", einsum("ka,la->kl", ur1, ur1), t2_1), t2_2,
            )
            + 0.5
            * einsum(
                "jb,ib->ij",
                einsum("kc,jkbc->jb", einsum("la,klac->kc", ur1, t2_1), t2_1), ur1,
            )
            + 0.5
            * einsum(
                "jb,ib->ij",
                einsum("kc,jkbc->jb", einsum("la,klac->kc", ur1, t2_1), t2_2), ur1,
            )
            + 0.5
            * einsum(
                "jb,ib->ij",
                einsum("kc,jkbc->jb", einsum("la,klac->kc", ur1, t2_2), t2_1), ur1,
            )
            - 1
            * einsum(
                "jc,ic->ij",
                einsum("lb,jlbc->jc", ur1, t2_2),
                einsum("ka,ikac->ic", ur1, t2_1),
            )
        ).symmetrise()
    )
    dm.ov = (  # adc3_p_ov
        -2 * einsum("jb,ijab->ia", ur1, ur2)
        + 1 * einsum("ik,ka->ia", einsum("jkbc,ijbc->ik", ur2, t2_1), ur1)
        + 1 * einsum("ik,ka->ia", einsum("jkbc,ijbc->ik", ur2, t2_2), ur1)
        + 1 * einsum("ijkb,jkab->ia", einsum("ic,jkbc->ijkb", ur1, ur2), t2_1)
        + 1 * einsum("ijkb,jkab->ia", einsum("ic,jkbc->ijkb", ur1, ur2), t2_2)
        + 1 * einsum("ib,ab->ia", einsum("jc,ijbc->ib", ur1, ur2), p0_2.vv)
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ur1, t1_2), ur1)
        - 1 * einsum("ij,ja->ia", einsum("jb,ib->ij", ur1, p0_3.ov), ur1)
        - 1 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, ur1), t1_2)
        - 1 * einsum("ij,ja->ia", einsum("ib,jb->ij", ur1, ur1), p0_3.ov)
        - 1 * einsum("ja,ij->ia", einsum("kb,jkab->ja", ur1, ur2), p0_2.oo)
        - 2 * einsum("kb,ikab->ia", einsum("jc,jkbc->kb", ur1, ur2), t2_1)
        - 2 * einsum("kb,ikab->ia", einsum("jc,jkbc->kb", ur1, ur2), t2_2)
        + 1
        * einsum(
            "kc,ikac->ia",
            einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, ur1), t1_2), t2_1,
        )
        + 1
        * einsum(
            "kc,ikac->ia",
            einsum("jk,jc->kc", einsum("jb,kb->jk", ur1, t1_2), ur1), t2_1,
        )
        + 0.5
        * einsum(
            "il,la->ia",
            einsum(
                "ikcd,klcd->il", einsum("jb,ijkbcd->ikcd", ur1, t3_2), t2_1), ur1,
        )
        + 0.5
        * einsum(
            "iklc,klac->ia",
            einsum("id,klcd->iklc", ur1, t2_1),
            einsum("jb,jklabc->klac", ur1, t3_2),
        )
        + 0.5
        * einsum(
            "ijkb,jkab->ia",
            einsum("id,jkbd->ijkb", einsum("lc,ilcd->id", ur1, t2_1), t2_1), ur2,
        )
        + 0.5
        * einsum(
            "ld,ilad->ia",
            einsum("jklb,jkbd->ld", einsum("lc,jkbc->jklb", ur1, ur2), t2_1), t2_1,
        )
        + 0.5
        * einsum(
            "il,la->ia",
            einsum("ikbc,klbc->il", ur2, t2_1),
            einsum("jd,jlad->la", ur1, t2_1),
        )
        + 0.5
        * einsum(
            "kd,ikad->ia",
            einsum("kl,ld->kd", einsum("jlbc,jkbc->kl", ur2, t2_1), ur1), t2_1,
        )
        - 1
        * einsum(
            "ij,ja->ia",
            einsum("ic,jc->ij", einsum("kb,ikbc->ic", ur1, t2_1), t1_2), ur1,
        )
        - 1
        * einsum(
            "ka,ik->ia",
            einsum("jc,jkac->ka", ur1, t2_1),
            einsum("ib,kb->ik", ur1, t1_2),
        )
        - 1
        * einsum(
            "kc,ikac->ia",
            einsum("ld,klcd->kc", ur1, t2_1),
            einsum("jb,ijkabc->ikac", ur1, t3_2),
        )
        - 1
        * einsum(
            "ijkc,jkac->ia",
            einsum("ikld,jlcd->ijkc", einsum("kb,ilbd->ikld", ur1, t2_1), t2_1),
            ur2,
        )
        - 1
        * einsum(
            "ijld,jlad->ia",
            einsum("ijkb,klbd->ijld", einsum("jc,ikbc->ijkb", ur1, ur2), t2_1),
            t2_1,
        )
        - 1
        * einsum(
            "kc,ikac->ia",
            einsum("jb,jkbc->kc", einsum("ld,jlbd->jb", ur1, ur2), t2_1), t2_1,
        )
        - 0.5
        * einsum(
            "jkbc,ijkabc->ia",
            einsum("jklc,lb->jkbc", einsum("ld,jkcd->jklc", ur1, t2_1), ur1), t3_2,
        )
        - 0.5
        * einsum(
            "jkbc,ijkabc->ia",
            einsum("jl,klbc->jkbc", einsum("jd,ld->jl", ur1, ur1), t2_1), t3_2,
        )
        - 0.25
        * einsum(
            "ijkl,jkla->ia",
            einsum("ilcd,jkcd->ijkl", t2_1, t2_1),
            einsum("lb,jkab->jkla", ur1, ur2),
        )
        - 0.25
        * einsum(
            "ijkl,jkla->ia",
            einsum("ijbc,klbc->ijkl", ur2, t2_1),
            einsum("jd,klad->jkla", ur1, t2_1),
        )
    )
    dm.vv = (  # adc3_p_vv
        +1 * einsum("ia,ib->ab", ur1, ur1)
        + 2 * einsum("ijac,ijbc->ab", ur2, ur2)
        + 1
        * einsum(
            "kb,ka->ab",
            einsum("jd,jkbd->kb", ur1, t2_1),
            einsum("ic,ikac->ka", ur1, t2_1),
        )
        - 1
        * einsum(
            "jkad,jkbd->ab",
            einsum("ij,ikad->jkad", einsum("ic,jc->ij", ur1, ur1), t2_1), t2_1,
        )
        - 0.5
        * einsum(
            "ijkb,ijka->ab",
            einsum("id,jkbd->ijkb", ur1, t2_1),
            einsum("ic,jkac->ijka", ur1, t2_1),
        )
        + 2
        * (
            +2 * einsum("ja,jb->ab", einsum("ic,ijac->ja", ur1, ur2), t1_2)
            - 0.5 * einsum("ib,ia->ab", einsum("ic,bc->ib", ur1, p0_2.vv), ur1)
            - 0.5 * einsum("ib,ia->ab", einsum("ic,bc->ib", ur1, p0_3.vv), ur1)
            + 1
            * einsum(
                "kb,ka->ab",
                einsum("jd,jkbd->kb", ur1, t2_2),
                einsum("ic,ikac->ka", ur1, t2_1),
            )
            + 0.5
            * einsum(
                "ib,ia->ab",
                einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_1), t2_1), ur1,
            )
            + 0.5
            * einsum(
                "ib,ia->ab",
                einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_2), t2_1), ur1,
            )
            + 0.5
            * einsum(
                "ib,ia->ab",
                einsum("kd,ikbd->ib", einsum("jc,jkcd->kd", ur1, t2_1), t2_2), ur1,
            )
            - 1
            * einsum(
                "jkad,jkbd->ab",
                einsum("ij,ikad->jkad", einsum("ic,jc->ij", ur1, ur1), t2_1), t2_2,
            )
            - 0.5
            * einsum(
                "ijkb,ijka->ab",
                einsum("id,jkbd->ijkb", ur1, t2_2),
                einsum("ic,jkac->ijka", ur1, t2_1),
            )
        ).symmetrise()
    )

    return dm


def diffdm_cvs_adc3(mp, amplitude, intermediates):
    raise NotImplementedError("CVS-ADC(3) is not implemented yet")


# dict controlling the dispatch of the state_diffdm function
DISPATCH = {
    "isr0": diffdm_isr0,
    "isr1s": diffdm_isr0,  # Identical to ISR(0)
    "isr1": diffdm_isr1,
    "isr2d": diffdm_isr2,  # Identical to ISR(2)
    "isr2": diffdm_isr2,
    "isr3": diffdm_isr3,
    "cvs-isr0": diffdm_isr0,
    "cvs-isr1s": diffdm_isr0,  # Identical to ISR(0)
    "cvs-isr1": diffdm_cvs_isr1,
    "cvs-isr2d": diffdm_cvs_isr2,  # Identical to CVS-ISR(2)
    "cvs-isr2": diffdm_cvs_isr2,
}


def state_diffdm(method, ground_state, amplitude, intermediates=None):
    """
    Compute the one-particle difference density matrix of an excited state
    in the MO basis.

    Parameters
    ----------
    method : str, IsrMethod
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
        raise NotImplementedError(
            "state_diffdm is not implemented " f"for {method.name}."
        )
    else:
        ret = DISPATCH[method.name](ground_state, amplitude, intermediates)
        return ret.evaluate()
