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
import libadcc
import numpy as np

from .functions import direct_sum, evaluate, einsum
from .misc import cached_property, cached_member_function
from .ReferenceState import ReferenceState
from .OneParticleOperator import OneParticleOperator, product_trace
from .Intermediates import register_as_intermediate
from .timings import Timer, timed_member_call
from .MoSpaces import split_spaces
from . import block as b


class LazyMp:
    # sigma4+ ranks between third and fourth order
    _special_density_orders = {"sigma4+": 3.5}

    def __init__(self, hf, density_order=None):
        """
        Initialise the class dealing with the Møller-Plesset ground state.

        Parameters
        ----------
        hf : adcc.ReferenceState
            HF reference the MP ground state is build on top.
        density_order : int or str, optional
            MP densities are optionally upgraded (through the strict flag) to the
            level defined by the density order, e.g., a MP2 density can be
            upgraded to a MP3 or sigma4+ density.
        """
        if isinstance(hf, libadcc.HartreeFockSolution_i):
            hf = ReferenceState(hf)
        if not isinstance(hf, ReferenceState):
            raise TypeError("hf needs to be a ReferenceState "
                            "or a HartreeFockSolution_i")
        if density_order is not None and not isinstance(density_order, int) and \
                density_order not in self._special_density_orders:
            raise ValueError(f"Invalid density order {density_order}. Valid are "
                             "numbers and "
                             f"{list(self._special_density_orders.keys())}.")
        self.reference_state = hf
        self.mospaces = hf.mospaces
        self.timer = Timer()
        self.has_core_occupied_space = hf.has_core_occupied_space
        self.density_order = density_order

    def _apply_density_order(self, level):
        """
        Apply the density order to the given level potentially upgrading
        the level according to the density order.
        """
        if self.density_order is None:
            return level
        # ensure that level is valid
        if not isinstance(level, int) and level not in self._special_density_orders:
            raise ValueError(f"Invalid level {level}. Valid are numbers and "
                             f"{list(self._special_density_orders.keys())}.")
        return max(level, self.density_order,
                   key=lambda lev: self._special_density_orders.get(lev, lev))

    def __getattr__(self, attr):
        # Shortcut some quantities, which are needed most often
        if attr.startswith("t2") and len(attr) == 4:  # t2oo, t2oc, t2cc
            xxvv = b.__getattr__(attr[2:4] + "vv")
            return self.t2(xxvv)
        else:
            raise AttributeError

    @cached_member_function
    def df(self, space):
        """Delta Fock matrix"""
        hf = self.reference_state
        s1, s2 = split_spaces(space)
        fC = hf.fock(s1 + s1).diagonal()
        fv = hf.fock(s2 + s2).diagonal()
        return direct_sum("-i+a->ia", fC, fv)

    @cached_member_function
    def t2(self, space):
        """T2 amplitudes"""
        hf = self.reference_state
        sp = split_spaces(space)
        assert all(s == b.v for s in sp[2:])
        eia = self.df(sp[0] + b.v)
        ejb = self.df(sp[1] + b.v)
        return (
            hf.eri(space) / direct_sum("ia+jb->ijab", eia, ejb).symmetrise((2, 3))
        )

    @cached_member_function
    def td2(self, space):
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

    @cached_member_function
    def t2eri(self, space, contraction):
        """
        Return the T2 tensor with ERI tensor contraction intermediates.
        These are called pi1 to pi7 in libadc.
        """
        hf = self.reference_state
        key = space + contraction
        expressions = {
            # space + contraction
            b.ooov + b.vv: ('ijbc,kabc->ijka', b.ovvv),
            b.ooov + b.ov: ('ilab,lkjb->ijka', b.ooov),
            b.oovv + b.oo: ('klab,ijkl->ijab', b.oooo),
            b.oovv + b.ov: ('jkac,kbic->ijab', b.ovov),
            b.oovv + b.vv: ('ijcd,abcd->ijab', b.vvvv),
            b.ovvv + b.oo: ('jkbc,jkia->iabc', b.ooov),
            b.ovvv + b.ov: ('ijbd,jcad->iabc', b.ovvv),
        }
        if key not in expressions:
            raise NotImplementedError("t2eri intermediate not implemented "
                                      f"for space '{space}' and contraction "
                                      f"'{contraction}'.")
        contraction_str, eri_block = expressions[key]
        return einsum(contraction_str, self.t2oo, hf.eri(eri_block))

    @cached_property
    def m_3_plus(self):
        """
        Third order contribution to the ov block of the N+1 part of the dynamic
        self-energy.
        """
        # NOTE: m_3_plus, m_3_minus have to be implemented on LazyMp so we can
        # use them for the evaluation of the MP3 density: We can't add an
        # Intermediate instance to LazyMp (circular reference -> memory leak)!
        return (
            + 1 * einsum("ijbc,jabc->ia", self.t2oo, self.t2eri(b.ovvv, b.ov))
            + 0.5 * einsum(
                "ijbc,jabc->ia", self.td2(b.oovv), self.reference_state.ovvv
            )
            - 0.25 * einsum("ijbc,jabc->ia", self.t2oo, self.t2eri(b.ovvv, b.oo))
        )

    @cached_property
    def m_3_minus(self):
        """
        Third order contribution to the ov block of the N-1 part of the dynamic
        self-energy.
        """
        return (
            + 0.5 * einsum(
                "jkab,jkib->ia", self.td2(b.oovv), self.reference_state.ooov
            )
            - 1 * einsum("jkab,jkib->ia", self.t2oo, self.t2eri(b.ooov, b.ov))
            - 0.25 * einsum("jkab,jkib->ia", self.t2oo, self.t2eri(b.ooov, b.vv))
        )

    @cached_member_function
    def sigma_inf_ov(self, level: int):
        """The ov part of the static self-energy."""
        hf = self.reference_state
        dm = self.diffdm(level - 1, strict=True)
        return (
            - einsum("ijka,jk->ia", hf.ooov, dm.oo)
            + einsum("ijab,jb->ia", hf.oovv, dm.ov)
            - einsum("ibja,jb->ia", hf.ovov, dm.ov)
            + einsum("ibac,bc->ia", hf.ovvv, dm.vv)
        )

    def diffdm(self, level: int = 2, strict: bool = True) -> OneParticleOperator:
        """
        Return the MPn difference density in the MO basis.
        If strict is set, the strict MPn difference density is returned. Otherwise
        the density might be upgraded to a higher order according to the
        density order.
        """
        # deal with the density order: we only upgrade the density!
        if not strict:
            level = self._apply_density_order(level)
        # compute the density
        if level == 2:
            return self.mp2_diffdm
        elif level == 3:
            return self.mp3_diffdm
        elif level == "sigma4+":
            raise NotImplementedError()
        else:
            raise NotImplementedError("Difference density not implemented for level"
                                      f" {level}.")

    @cached_property
    @timed_member_call(timer="timer")
    def mp2_diffdm(self):
        """
        Return the MP2 difference density in the MO basis.
        """
        hf = self.reference_state
        ret = OneParticleOperator(self.mospaces, is_symmetric=True)
        # NOTE: the following 3 blocks are equivalent to the cvs_p0 intermediates
        # defined at the end of this file
        ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        ret.ov = -0.5 * (
            + einsum("ijbc,jabc->ia", self.t2oo, hf.ovvv)
            + einsum("jkib,jkab->ia", hf.ooov, self.t2oo)
        ) / self.df(b.ov)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)

        if self.has_core_occupied_space:
            # additional terms to "revert" CVS for ground state density
            ret.oo += -0.5 * einsum("iLab,jLab->ij", self.t2oc, self.t2oc)
            ret.ov += -0.5 * (
                + einsum("jMib,jMab->ia", hf.ocov, self.t2oc)
                + einsum("iLbc,Labc->ia", self.t2oc, hf.cvvv)
                + einsum("kLib,kLab->ia", hf.ocov, self.t2oc)
                + einsum("iMLb,LMab->ia", hf.occv, self.t2cc)
                - einsum("iLMb,LMab->ia", hf.occv, self.t2cc)
            ) / self.df(b.ov)
            ret.vv += (
                + 0.5 * einsum("IJac,IJbc->ab", self.t2cc, self.t2cc)
                + 1.0 * einsum("kJac,kJbc->ab", self.t2oc, self.t2oc)
            )
            # compute extra CVS blocks
            ret.cc = -0.5 * (
                + einsum("kIab,kJab->IJ", self.t2oc, self.t2oc)
                + einsum('LIab,LJab->IJ', self.t2cc, self.t2cc)
            )
            ret.co = -0.5 * (
                + einsum("kIab,kjab->Ij", self.t2oc, self.t2oo)
                + einsum("ILab,jLab->Ij", self.t2cc, self.t2oc)
            )
            ret.cv = -0.5 * (
                - einsum("jIbc,jabc->Ia", self.t2oc, hf.ovvv)
                + einsum("jkIb,jkab->Ia", hf.oocv, self.t2oo)
                + einsum("jMIb,jMab->Ia", hf.occv, self.t2oc)
                + einsum("ILbc,Labc->Ia", self.t2cc, hf.cvvv)
                + einsum("kLIb,kLab->Ia", hf.occv, self.t2oc)
                + einsum("LMIb,LMab->Ia", hf.cccv, self.t2cc)
            ) / self.df(b.cv)
        ret.reference_state = self.reference_state
        return evaluate(ret)

    @cached_property
    @timed_member_call(timer="timer")
    def mp3_diffdm(self):
        """
        Return the MP3 difference density in the MO basis.
        """
        if self.has_core_occupied_space:
            raise NotImplementedError("MP3 density not implemented for CVS.")
        ret = OneParticleOperator(self.mospaces, is_symmetric=True)

        mp2_dm = self.mp2_diffdm
        ret.oo = (
            mp2_dm.oo  # 2nd order
            # 3rd order
            - einsum("ikab,jkab->ij", self.t2oo, self.td2(b.oovv)).symmetrise(0, 1)
        )
        ret.ov = (
            mp2_dm.ov  # 2nd order
            - (  # 3rd order
                self.sigma_inf_ov(3) + self.m_3_plus + self.m_3_minus
            ) / self.df(b.ov)
        )
        ret.vv = (
            mp2_dm.vv  # 2nd order
            # 3rd order
            + einsum("ijac,ijbc->ab", self.t2oo, self.td2(b.oovv)).symmetrise(0, 1)
        )
        return evaluate(ret)

    def density(self, level: int = 2, strict: bool = True):
        """
        Return the MP density in the MO basis with all corrections
        up to the specified order of perturbation theory.
        If strict is set, the strict MPn density is returned. Otherwise the
        density might be upgraded to a higher order density according to the
        density order.
        """
        if level == 1:
            return self.reference_state.density
        diffdm = self.diffdm(level, strict=strict)
        return self.reference_state.density + diffdm

    def dipole_moment(self, level: int = 2, strict: bool = True):
        """
        Return the MP dipole moment at the specified level of
        perturbation theory.
        If strict is set, the strict MPn dipole moment is computed. Otherwise the
        dipole moment might be computed for a higher order density according to the
        density_order.
        """
        if level == 1:
            return self.reference_state.dipole_moment
        diffdm = self.diffdm(level, strict=strict)
        dipole_integrals = self.reference_state.operators.electric_dipole
        correction = -np.array(
            [product_trace(comp, diffdm) for comp in dipole_integrals]
        )
        return self.reference_state.dipole_moment + correction

    @cached_member_function
    def energy_correction(self, level=2):
        """Obtain the MP energy correction at a particular level"""
        if level > 3:
            raise NotImplementedError(f"MP({level}) energy correction "
                                      "not implemented.")
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
        elif level == 3 and is_cvs:
            raise NotImplementedError("CVS-MP3 energy correction not implemented.")
        return sum(
            -0.25 * pref * eri.dot(t2)
            for pref, eri, t2 in terms
        )

    def energy(self, level=2):
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        if level == 0:
            # Sum of orbital energies ...
            raise NotImplementedError("Total MP(0) energy not implemented.")

        # Accumulator for all energy terms
        energies = [self.reference_state.energy_scf]
        for il in range(2, level + 1):
            energies.append(self.energy_correction(il))
        return sum(energies)

    def to_qcvars(self, properties=False, recurse=False, maxlevel=2):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        qcvars = {}
        for level in range(2, maxlevel + 1):
            try:
                mpcorr = self.energy_correction(level)
                qcvars[f"MP{level} CORRELATION ENERGY"] = mpcorr
                qcvars[f"MP{level} TOTAL ENERGY"] = self.energy(level)
            except NotImplementedError:
                pass
            except ValueError:
                pass

        if properties:
            for level in range(2, maxlevel + 1):
                try:
                    qcvars["MP2 DIPOLE"] = self.dipole_moment(level)
                except NotImplementedError:
                    pass

        if recurse:
            qcvars.update(self.reference_state.to_qcvars(properties, recurse))
        return qcvars

    @property
    def mp2_density(self):
        return self.density(2, strict=True)

    @cached_property
    def mp2_dipole_moment(self):
        return self.dipole_moment(2, strict=True)


#
# Register cvs_p0 intermediate
#
@register_as_intermediate
def cvs_p0(hf: ReferenceState, mp: LazyMp, intermediates):
    # TODO: this is a bit weird with the density order for CVS. But as long as we
    # don't have any other CVS density implemented it should be fine.
    if mp._apply_density_order(2) != 2:
        raise NotImplementedError("The MP2 density with CVS is only implemented "
                                  f"in 2nd order. An upgrade to level "
                                  f"{mp._apply_density_order(2)} is not available.")
    # NOTE: equal to mp2_diffdm if CVS applied for the density
    ret = OneParticleOperator(hf.mospaces, is_symmetric=True)
    ret.oo = -0.5 * einsum("ikab,jkab->ij", mp.t2oo, mp.t2oo)
    ret.ov = -0.5 * (+ einsum("ijbc,jabc->ia", mp.t2oo, hf.ovvv)
                     + einsum("jkib,jkab->ia", hf.ooov, mp.t2oo)) / mp.df(b.ov)
    ret.vv = 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, mp.t2oo)
    return ret
