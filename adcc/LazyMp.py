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
import libadcc
import numpy as np

from .GroundState import GroundState
from .Intermediates import Intermediates, register_as_intermediate
from .MoSpaces import split_spaces
from .NParticleOperator import OperatorSymmetry
from .OneParticleDensity import OneParticleDensity
from .ReferenceState import ReferenceState
from .functions import direct_sum, einsum
from .misc import cached_member_function, cached_property
from .timings import timed_member_call
from . import block as b


class LazyMp(GroundState):
    @cached_member_function()
    def td1(self, space: str) -> libadcc.Tensor:
        """T2 amplitudes"""
        hf = self.reference_state
        sp = split_spaces(space)
        assert all(s == b.v for s in sp[2:])
        eia = self.df(sp[0] + b.v)
        ejb = self.df(sp[1] + b.v)
        return (
            hf.eri(space) / direct_sum("ia+jb->ijab", eia, ejb).symmetrise((2, 3))
        )

    def ts2(self, space: str) -> libadcc.Tensor:
        split = split_spaces(space)
        assert len(split) == 2 and split[1] == b.v
        if split[0] == b.o:
            return self.ts2_ov
        elif split[0] == b.c:
            return self.ts2_cv
        raise NotImplementedError(
            "Second-order MP singles amplitudes not implemented for space "
            f"'{space}'."
        )

    @cached_property
    @timed_member_call(timer="timer")
    def ts2_ov(self) -> libadcc.Tensor:
        """
        Computes the ov block of the second-order MP singles amplitudes.
        """
        hf = self.reference_state
        t2oo = self.td1(b.oovv)
        denom = -self.df(b.ov)
        res = (
            # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum('jabc,ijbc->ia', hf.ovvv, t2oo)
            # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum('jkib,jkab->ia', hf.ooov, t2oo)
        )
        # additional terms since we don't apply the CVS approximation
        # for the ground state
        # (all of the following terms vanish within the approximation)
        if self.has_core_occupied_space:
            t2oc = self.td1(b.ocvv)
            t2cc = self.td1(b.ccvv)
            res += (
                # N^5: O^2V^2C^1 / N^4: O^1V^2C^1
                + 1.0 * einsum("jKab,jKib->ia", t2oc, hf.ocov)
                # N^5: O^1V^2C^2 / N^4: V^2C^2
                + 0.5 * einsum("JKab,ibJK->ia", t2cc, hf.ovcc)
                # N^5: O^1V^3C^1 / N^4: V^3C^1
                + 0.5 * einsum("iJbc,Jabc->ia", t2oc, hf.cvvv)
            )
        return (res / denom).evaluate()

    @cached_property
    @timed_member_call(timer="timer")
    def ts2_cv(self) -> libadcc.Tensor:
        """
        Computes the cv block of the second-order MP singles amplitudes.
        """
        assert self.has_core_occupied_space
        hf = self.reference_state
        t2oo = self.td1(b.oovv)
        t2oc = self.td1(b.ocvv)
        t2cc = self.td1(b.ccvv)
        denom = -self.df(b.cv)
        res = (
            # N^5: O^1V^2C^2 / N^4: O^1V^2C^1
            + 1.0 * einsum("jKab,jKIb->Ia", t2oc, hf.occv)
            # N^5: V^2C^3 / N^4: V^2C^2
            + 0.5 * einsum("JKab,JKIb->Ia", t2cc, hf.cccv)
            # N^5: O^2V^2C^1 / N^4: O^2V^2
            + 0.5 * einsum("jkab,jkIb->Ia", t2oo, hf.oocv)
            # N^5: V^3C^2 / N^4: V^3C^1
            + 0.5 * einsum("IJbc,Jabc->Ia", t2cc, hf.cvvv)
            # N^5: O^1V^3C^1 / N^4: O^1V^3V
            - 0.5 * einsum("jIbc,jabc->Ia", t2oc, hf.ovvv)
        )
        return (res / denom).evaluate()

    @cached_member_function()
    def td2(self, space: str) -> libadcc.Tensor:
        """Return the T^D_2 term"""
        if space != b.oovv:
            raise NotImplementedError("T^D_2 term not implemented "
                                      f"for space {space}.")
        t2erit = self.t2eri(b.oovv, b.ov).transpose((1, 0, 2, 3))
        denom = direct_sum(
            'ia,jb->ijab', self.df(b.ov), self.df(b.ov)
        ).symmetrise(0, 1)
        return (
            + 4.0 * t2erit.antisymmetrise(2, 3).antisymmetrise(0, 1)
            - 0.5 * self.t2eri(b.oovv, b.vv)
            - 0.5 * self.t2eri(b.oovv, b.oo)
        ) / denom

    @cached_member_function()
    def tt2(self, space: str) -> libadcc.Tensor:
        """
        Return the second order MP triples amplitudes for the given space
        (e.g. o1o1o1v1v1v1).
        """
        if space != b.ooovvv:
            raise NotImplementedError("Second order MP triples amplitudes not "
                                      f"implemented for space {space}.")
        hf = self.reference_state
        df = self.df(b.ov)
        t2_1 = self.t2(b.oovv)
        # denom = a + b + c - i - j - k   //   df = i - a
        denom = - 1 * direct_sum(
            "ia,jkbc->ijkabc", df, direct_sum("jb,kc->jkbc", df, df)
        ).symmetrise(0, 1, 2).symmetrise(3, 4, 5)
        # prefactor of 9, because we have 9 terms each in the expression, while the
        # antisymmetrisation generates 36 terms and introduces a prefactor of 1/36.
        # Each of the 9 terms is therefore generated 4 times.
        # The scaling in the comments is given as: [comp_scaling] / [mem_scaling]
        numerator = (
            # N^7: O^3V^4 / N^6: O^3V^3
            + 9 * einsum('idab,jkcd->ijkabc', hf.ovvv, t2_1)
            # N^7: O^4V^3 / N^6: O^3V^3
            + 9 * einsum('ijla,klbc->ijkabc', hf.ooov, t2_1)
        ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5)
        return numerator / denom

    @cached_member_function()
    def energy_correction(self, level: int = 2) -> float:
        """Obtain the MP energy correction at a particular level"""
        assert level >= 0
        if level < 2:
            return 0.0
        hf = self.reference_state
        is_cvs = self.has_core_occupied_space
        if level == 2 and not is_cvs:
            terms = [(1.0, hf.oovv, self.t2oo)]
        elif level == 2 and is_cvs:
            terms = [(1.0, hf.oovv, self.t2oo),
                     (2.0, hf.ocvv, self.t2oc),
                     (1.0, hf.ccvv, self.t2cc)]
        elif level == 3 and not is_cvs:
            terms = [(1.0, hf.oovv, self.td2(b.oovv))]
        else:
            method = "CVS-MP" if is_cvs else "MP"
            raise NotImplementedError(f"{method} energy correction for level "
                                      f"{level} not implemented.")
        return sum(
            -0.25 * pref * eri.dot(t2)
            for pref, eri, t2 in terms
        )

    def energy(self, level: int = 2) -> float:
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        assert level >= 0
        if level == 0:
            # Sum of orbital energies ...
            return np.einsum(
                "i->", self.reference_state.foo.diagonal().to_ndarray()
            )

        # Accumulator for all energy terms
        energies = [self.reference_state.energy_scf]
        for il in range(2, level + 1):
            energies.append(self.energy_correction(il))
        return sum(energies)

    def to_qcvars(self, properties: bool = False,
                  recurse: bool = False, maxlevel: int = 2) -> dict:
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        return self._to_qcvars(
            gs_type="MP", properties=properties, recurse=recurse,
            maxlevel=maxlevel
        )

    @property
    def mp2_diffdm(self):
        """
        Return the MP2 difference density in the MO basis.
        """
        return self.diffdm(2)

    @property
    def mp2_density(self):
        return self.density(2)

    @property
    def mp2_dipole_moment(self):
        return self.dipole_moment(level=2)


#
# Register cvs_p0 intermediate
#
@register_as_intermediate
def cvs_p0(hf: ReferenceState, mp: LazyMp, intermediates: Intermediates
           ) -> OneParticleDensity:
    # NOTE: equal to mp2_diffdm if the CVS approximation is applied for the density
    # This is also MP specific!
    # TODO: This needs to be solved differenty. Maybe add some parameter
    # apply_cvs to the density and the amplitude methods
    ret = OneParticleDensity(hf.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
    ret.oo = -0.5 * einsum("ikab,jkab->ij", mp.t2oo, mp.t2oo)
    ret.ov = -0.5 * (+ einsum("ijbc,jabc->ia", mp.t2oo, hf.ovvv)
                     + einsum("jkib,jkab->ia", hf.ooov, mp.t2oo)) / mp.df(b.ov)
    ret.vv = 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, mp.t2oo)
    return ret
