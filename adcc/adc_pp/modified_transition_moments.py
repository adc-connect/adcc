#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
from adcc.AdcMethod import IsrMethod
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import einsum, evaluate
from adcc.Intermediates import Intermediates
from adcc.LazyMp import LazyMp


def mtm_isr0(mp, op, intermediates):
    f1 = op.vo.transpose()
    return AmplitudeVector(ph=f1)


def mtm_isr1(mp, op, intermediates):
    ampl = mtm_isr0(mp, op, intermediates)
    f1 = - 1.0 * einsum("ijab,jb->ia", mp.t2(b.oovv), op.ov)
    return ampl + AmplitudeVector(ph=f1)


def mtm_isr2(mp, op, intermediates):
    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm

    ampl = mtm_isr1(mp, op, intermediates)
    f1 = (
        + 0.5 * einsum("ijab,jkbc,ck->ia", t2, t2, op.vo)
        + 0.5 * einsum("ij,aj->ia", p0.oo, op.vo)
        - 0.5 * einsum("bi,ab->ia", op.vo, p0.vv)
        + 1.0 * einsum("ib,ab->ia", p0.ov, op.vv)
        - 1.0 * einsum("ji,ja->ia", op.oo, p0.ov)
        - 1.0 * einsum("ijab,jb->ia", mp.td2(b.oovv), op.ov)
    )
    f2 = (
        + 1.0 * einsum("ijac,bc->ijab", t2, op.vv).antisymmetrise(2, 3)
        + 1.0 * einsum("ki,jkab->ijab", op.oo, t2).antisymmetrise(0, 1)
    )
    return ampl + AmplitudeVector(ph=f1, pphh=f2)


def mtm_cvs_isr0(mp, op, intermediates):
    f1 = op.vc.transpose()
    return AmplitudeVector(ph=f1)


def mtm_cvs_isr2(mp, op, intermediates):

    ampl = mtm_cvs_isr0(mp, op, intermediates)
    f1 = (
        - 0.5 * einsum("bI,ab->Ia", op.vc, intermediates.cvs_p0.vv)
        - 1.0 * einsum("jI,ja->Ia", op.oc, intermediates.cvs_p0.ov)
    )
    f2 = (1 / sqrt(2)) * einsum("kI,kjab->jIab", op.oc, mp.t2(b.oovv))
    return ampl + AmplitudeVector(ph=f1, pphh=f2)


def mtm_isr3(mp, op, intermediates):

    f1 = mtm_isr2(mp, op, intermediates).ph
    f2 = mtm_isr2(mp, op, intermediates).pphh
    # mp amplitudes
    t1_2 = mp.diffdm(level=2).ov
    t2_1 = mp.t2(b.oovv)
    t2_3 = mp.td3(b.oovv)
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()

    d_vv, d_vo, d_ov, d_oo = op.vv, op.vo, op.ov, op.oo  # density operators

    p0_3 = mp.mp3_dm_correction
    p0_2 = mp.mp2_dm_correction
    p0_3_oo, p0_3_vv, p0_3_ov = p0_3.oo, p0_3.vv, p0_3.ov
    p0_2_oo, p0_2_vv = p0_2.oo, p0_2.vv

    f1 += (
        + 1 * einsum("ab,ib->ia", d_vv, p0_3_ov)  # N^3: O^1V^2 / N^2: V^2
        + 0.5 * einsum("aj,ij->ia", d_vo, p0_3_oo)  # N^3: O^2V^1 / N^2: O^1V^1
        - 1 * einsum("ijab,jb->ia", t2_3, d_ov)  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum("ji,ja->ia", d_oo, p0_3_ov)  # N^3: O^2V^1 / N^2: O^1V^1
        - 0.5 * einsum("bi,ab->ia", d_vo, p0_3_vv)  # N^3: O^1V^2 / N^2: V^2
        + 1 * einsum(
            "jb,ijab->ia", einsum("kb,jk->jb", t1_2, d_oo), t2_1
        )  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum(
            "jc,ijac->ia", einsum("jb,bc->jc", d_ov, p0_2_vv), t2_1
        )  # N^4: O^2V^2 / N^4: O^2V^2
        + 1 * einsum(
            "ijkc,jkac->ia", einsum("jb,ibkc->ijkc", d_ov, t2sq), t2_1
        )  # N^5: O^3V^2 / N^4: O^2V^2
        + 0.5 * einsum(
            "ka,ik->ia", einsum("jkab,jb->ka", t2_1, d_ov), p0_2_oo
        )  # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum(
            "kc,ikac->ia", einsum("jkbc,bj->kc", t2_2, d_vo), t2_1
        )  # N^4: O^2V^2 / N^4: O^2V^2
        + 0.5 * einsum(
            "jb,ijab->ia", einsum("jkbc,ck->jb", t2_1, d_vo), t2_2
        )  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum(
            "jb,ijab->ia", einsum("jc,cb->jb", t1_2, d_vv), t2_1
        )  # N^4: O^2V^2 / N^4: O^2V^2
        - 1 * einsum(
            "kb,ikab->ia", einsum("jb,jk->kb", d_ov, p0_2_oo), t2_1
        )  # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum(
            "ic,ac->ia", einsum("ijbc,jb->ic", t2_1, d_ov), p0_2_vv
        )  # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum(
            "kc,iakc->ia", einsum("jkbc,jb->kc", t2_1, d_ov), t2sq
        )  # N^4: O^2V^2 / N^4: O^2V^2
        - 0.5 * einsum(
            "jkbc,ijkabc->ia", einsum("klbc,jl->jkbc", t2_1, d_oo), t3_2
        )  # N^6: O^3V^3 / N^6: O^3V^3
        - 0.5 * einsum(
            "jkbc,ijkabc->ia", einsum("jkbd,dc->jkbc", t2_1, d_vv), t3_2
        )  # N^6: O^3V^3 / N^6: O^3V^3
        - 0.25 * einsum(
            "ijkl,jkla->ia",
            einsum("ijcd,klcd->ijkl", t2_1, t2_1),
            einsum("klab,jb->jkla", t2_1, d_ov),
        )  # N^6: O^4V^2 / N^4: O^2V^2
    )

    f2 += (
        2 * (
            0.5 * einsum("jkab,ki->ijab", t2_2, d_oo)  # N^5: O^3V^2 / N^4: O^2V^2
        ).antisymmetrise(0, 1)
        + 2 * (
            0.5 * einsum("ijac,bc->ijab", t2_2, d_vv)  # N^5: O^2V^3 / N^4: O^2V^2
        ).antisymmetrise(2, 3)
        - 0.5 * einsum("ijkabc,kc->ijab", t3_2, d_ov)  # N^6: O^3V^3 / N^6: O^3V^3
    )
    return AmplitudeVector(ph=f1, pphh=f2)


def mtm_cvs_isr3(mp, op, intermediates):
    raise NotImplementedError("CVS-ADC(3) is not implemented yet")


DISPATCH = {
    "isr0": mtm_isr0,
    "isr1s": mtm_isr1,  # Identical to ISR(1)
    "isr1": mtm_isr1,
    "isr2d": mtm_isr2,  # Identical to ISR(2)
    "isr2": mtm_isr2,
    "isr3": mtm_isr3,
    "isr3d": mtm_isr3,
    "cvs-isr0": mtm_cvs_isr0,
    "cvs-isr1s": mtm_cvs_isr0,  # Identical to CVS-ISR(0)
    "cvs-isr1": mtm_cvs_isr0,  # Identical to CVS-ISR(0)
    "cvs-isr2d": mtm_cvs_isr2,  # Identical to CVS-ISR(2)
    "cvs-isr2": mtm_cvs_isr2,
}


def modified_transition_moments(
    method, ground_state, operator=None, intermediates=None
):
    """Compute the modified transition moments (MTM) for the provided
    ISR method with reference to the passed ground state.

    Parameters
    ----------
    method: adcc.IsrMethod
        Provide a method at which to compute the MTMs
    ground_state : adcc.LazyMp
        The MP ground state
    operator : adcc.OneParticleOperator or list, optional
        Only required if different operators than the standard
        electric dipole operators in the MO basis should be used.
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse

    Returns
    -------
    adcc.AmplitudeVector or list of adcc.AmplitudeVector
    """
    if not isinstance(method, IsrMethod):
        method = IsrMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    unpack = False
    if operator is None:
        operator = ground_state.reference_state.operators.electric_dipole
    elif not isinstance(operator, (list, tuple)):
        unpack = True
        operator = [operator]
    if method.name not in DISPATCH:
        raise NotImplementedError(
            f"modified_transition_moments is not implemented for {method.name}."
        )

    ret = [DISPATCH[method.name](ground_state, op, intermediates)
           for op in operator]
    if unpack:
        assert len(ret) == 1
        ret = ret[0]
    return evaluate(ret)
