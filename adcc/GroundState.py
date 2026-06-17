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
from .MoSpaces import MoSpaces, split_spaces
from .NParticleOperator import OperatorSymmetry, product_trace
from .OneParticleDensity import OneParticleDensity
from .ReferenceState import ReferenceState
from .TwoParticleDensity import TwoParticleDensity
from .functions import direct_sum, einsum, zeros_like
from .misc import cached_member_function
from .timings import Timer
from . import block as b

import libadcc

from typing import Union
import numpy as np


class GroundState:
    """
    Base class representing the ground state.
    """
    def __init__(self, hf: Union[ReferenceState, libadcc.HartreeFockSolution_i]):
        if isinstance(hf, libadcc.HartreeFockSolution_i):
            hf = ReferenceState(hf)
        if not isinstance(hf, ReferenceState):
            raise TypeError("hf needs to be a 'ReferenceState' "
                            "or a 'HartreeFockSolution_i'")
        self.reference_state: ReferenceState = hf
        self.mospaces: MoSpaces = hf.mospaces
        self.timer: Timer = Timer()
        self.has_core_occupied_space: bool = hf.has_core_occupied_space

    def td1(self, space: str) -> libadcc.Tensor:
        """First-order ground state doubles amplitudes."""
        raise NotImplementedError(
            "First-order ground state doubles amplitudes not implemented on "
            f"{self.__class__.__name__} class."
        )

    def t2(self, space: str) -> libadcc.Tensor:
        """Alias for td1 for backwards compatiblity."""
        return self.td1(space)

    @property
    def t2oo(self) -> libadcc.Tensor:
        return self.td1(b.oovv)

    @property
    def t2oc(self) -> libadcc.Tensor:
        return self.td1(b.ocvv)

    @property
    def t2cc(self) -> libadcc.Tensor:
        return self.td1(b.ccvv)

    def ts2(self, space: str, apply_cvs: bool = False) -> libadcc.Tensor:
        """Second-order ground state singles amplitudes"""
        # NOTE: In contrast to the other amplitudes, ts2 has the additional
        # apply_cvs argument, since ts2 corresponds to the OV/CV block of the
        # second order 1p density correction.
        raise NotImplementedError(
            "Second-order ground state singles amplitudes not implemented on "
            f"{self.__class__.__name__} class."
        )

    def td2(self, space: str) -> libadcc.Tensor:
        """Second-order ground state doubles amplitudes"""
        raise NotImplementedError(
            "Second-order ground state doubles amplitudes not implemented on "
            f"{self.__class__.__name__} class."
        )

    def tt2(self, space: str) -> libadcc.Tensor:
        """Second-order ground state triples amplitudes"""
        raise NotImplementedError(
            "Second-order ground state triples amplitudes not implemented on "
            f"{self.__class__.__name__} class."
        )

    def energy_correction(self, level: int = 2) -> float:
        """Obtain the ground state energy correction at a particular level"""
        raise NotImplementedError(
            "Ground state energy corrections not implemented on "
            f"{self.__class__.__name__} class."
        )

    def energy(self, level: int = 2) -> float:
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        raise NotImplementedError(
            "Ground state energy not implemented on "
            f"{self.__class__.__name__} class."
        )

    def diffdm(self, level: int = 2, apply_cvs: bool = False) -> OneParticleDensity:
        """
        Return the ground state difference denstiy in the MO basis
        with all corrections up to the specified order of perturbation theory.
        """
        if level >= 0 and level < 2:
            raise ValueError(f"Difference density of order {level} vanishes.")
        elif level == 2:
            return self.second_order_dm_correction(apply_cvs=apply_cvs)
        elif level == 3:
            return (
                self.second_order_dm_correction(apply_cvs=apply_cvs)
                + self.third_order_dm_correction(apply_cvs=apply_cvs)
            )
        else:
            raise NotImplementedError(
                "Only second-order density corection is implemented. "
                f"diffdm of level {level} is not available."
            )

    def density(self, level: int = 2,
                apply_cvs: bool = False) -> OneParticleDensity:
        """
        Return the ground state density in the MO basis with all corrections
        up to the specified order of perturbation theory.
        """
        if level in [0, 1]:
            return self.reference_state.density
        diffdm = self.diffdm(level, apply_cvs=apply_cvs)
        return self.reference_state.density + diffdm

    def diffdm_2p(self, level: int = 2,
                  apply_cvs: bool = False) -> TwoParticleDensity:
        """
        Return the two-particle ground state difference density in the MO basis
        with all corrections up to the specified order of perturbation theory.
        """
        if level == 0:
            raise ValueError("Zeroth-order 2-particle difference density vanishes.")
        elif level == 1:
            return self.first_order_dm_correction_2p(apply_cvs=apply_cvs)
        elif level == 2:
            return (self.first_order_dm_correction_2p(apply_cvs=apply_cvs)
                    + self.second_order_dm_correction_2p(apply_cvs=apply_cvs))
        raise NotImplementedError("Only first and second-order two-particle "
                                  "density corrections are implemented. "
                                  f"2p diffdm of level {level} is not available.")

    def density_2p(self, level: int = 2,
                   apply_cvs: bool = False) -> TwoParticleDensity:
        """
        Return the two-particle ground state density in the MO basis
        with all corrections up to the specified order of perturbation theory.
        """
        if level == 0:
            return self.reference_state.density_2p
        diffdm = self.diffdm_2p(level, apply_cvs=apply_cvs)
        return self.reference_state.density_2p + diffdm

    @cached_member_function()
    def dipole_moment(self, level: int = 2, apply_cvs: bool = False
                      ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """
        Return the ground state dipole moment at the specified level of
        perturbation theory.
        """
        dm = self.density(level, apply_cvs=apply_cvs)
        dipole_integrals = self.reference_state.operators.electric_dipole
        return self.reference_state.nuclear_dipole + np.array([
            product_trace(comp, dm) for comp in dipole_integrals
        ])

    @cached_member_function()
    def ssq(self, level: int = 2, apply_cvs: bool = False) -> float:
        """
        Return <S^2> of the ground state.
        """
        if self.reference_state.restricted:
            raise NotImplementedError(
                "<S^2> is not implemented for restricted HF references."
            )
        ssq_1p_op = self.reference_state.operators.ssq_1p
        ssq_2p_op = self.reference_state.operators.ssq_2p
        # the trace of the second-order (and higher) correction to the RDM1
        # is zero -> no influence on top of HF density for ground state
        ssq_1p = product_trace(ssq_1p_op, self.density(0))
        ssq_2p = product_trace(
            ssq_2p_op, self.density_2p(level, apply_cvs=apply_cvs)
        )
        return ssq_1p + ssq_2p

    @cached_member_function()
    def df(self, space: str) -> libadcc.Tensor:
        """Delta Fock matrix"""
        hf = self.reference_state
        s1, s2 = split_spaces(space)
        fC = hf.fock(s1 + s1).diagonal()
        fv = hf.fock(s2 + s2).diagonal()
        return direct_sum("-i+a->ia", fC, fv)

    @cached_member_function()
    def t2eri(self, space: str, contraction: str) -> libadcc.Tensor:
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

    @property
    def m_3_plus(self) -> libadcc.Tensor:
        """
        Third order contribution to the ov block of the N+1 part of the
        dynamic self-energy.
        """
        raise NotImplementedError(
            "Third order contribution to the ov block of the N+1 part "
            "of the dynamic self-energy not implemented on "
            f"{self.__class__.__name__} class."
        )

    @property
    def m_3_minus(self) -> libadcc.Tensor:
        """
        Third order contribution to the ov block of the N-1 part of the
        dynamic self-energy.
        """
        raise NotImplementedError(
            "Third order contribution to the ov block of the N-1 part "
            "of the dynamic self-energy not implemented on "
            f"{self.__class__.__name__} class."
        )

    @cached_member_function()
    def sigma_inf_ov(self, level: int) -> libadcc.Tensor:
        """The ov part of the static self-energy."""
        hf = self.reference_state
        dm = self.diffdm(level - 1)

        return (
            - einsum("ijka,jk->ia", hf.ooov, dm.oo)
            + einsum("ijab,jb->ia", hf.oovv, dm.ov)
            - einsum("ibja,jb->ia", hf.ovov, dm.ov)
            + einsum("ibac,bc->ia", hf.ovvv, dm.vv)
        )

    @cached_member_function()
    def second_order_dm_correction(self, apply_cvs: bool = False
                                   ) -> OneParticleDensity:
        """
        Return the second-order contribution to the ground state
        difference density in the MO basis.
        """
        if apply_cvs and not self.has_core_occupied_space:
            raise RuntimeError("Cannot apply the CVS approximation to a "
                               "ground state build on top of a HF reference state "
                               "without a core space.")

        ret = OneParticleDensity(
            self.mospaces, symmetry=OperatorSymmetry.HERMITIAN
        )
        ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        ret.ov = self.ts2(b.ov, apply_cvs=apply_cvs)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)

        if self.has_core_occupied_space and not apply_cvs:
            # additional terms since we don't apply the CVS approximation
            # for the GS density. Within the CVS approximation all of the
            # following terms vanish for the MP and RE partitionings.
            # Not sure if this is true for all partitionings!
            ret.oo += -0.5 * einsum("iLab,jLab->ij", self.t2oc, self.t2oc)
            ret.vv += (
                + 0.5 * einsum("IJac,IJbc->ab", self.t2cc, self.t2cc)
                + 1.0 * einsum("kJac,kJbc->ab", self.t2oc, self.t2oc)
            )
            # compute extra CVS blocks
            ret.cc = -0.5 * (
                + einsum("kIab,kJab->IJ", self.t2oc, self.t2oc)
                + einsum('LIab,LJab->IJ', self.t2cc, self.t2cc)
            )
            ret.oc = -0.5 * (
                + einsum("kIab,kjab->jI", self.t2oc, self.t2oo)
                + einsum("ILab,jLab->jI", self.t2cc, self.t2oc)
            )
            ret.cv = self.ts2(b.cv, apply_cvs=apply_cvs)
        ret.reference_state = self.reference_state
        return ret.evaluate()

    def third_order_dm_correction(self, apply_cvs: bool = False
                                  ) -> OneParticleDensity:
        """
        Return the third-order contribution to the ground state
        difference density in the MO basis.
        """
        raise NotImplementedError(
            "Third-order contribution to the ground state difference "
            "density in the MO basis not implemented on "
            f"{self.__class__.__name__} class."
        )

    @cached_member_function()
    def first_order_dm_correction_2p(self, apply_cvs: bool = False
                                     ) -> TwoParticleDensity:
        """
        Return the two-particle first-order difference density correction
        in the MO basis.
        """
        if self.has_core_occupied_space:
            raise NotImplementedError("First-order 2-particle DM correction not "
                                      "implemented for a ground state with "
                                      "core orbitals.")
        assert not apply_cvs  # TODO: once implemented for core orbitals
        ret = TwoParticleDensity(
            self.mospaces, symmetry=OperatorSymmetry.HERMITIAN
        )
        ret.oovv = -1.0 * self.t2oo
        ret.reference_state = self.reference_state
        return ret.evaluate()

    @cached_member_function()
    def second_order_dm_correction_2p(self, apply_cvs: bool = False
                                      ) -> TwoParticleDensity:
        """
        Return the two-particle second-order difference density correction
        in the MO basis.
        """
        if self.has_core_occupied_space:
            raise NotImplementedError("Second-order 2-particle DM correction not "
                                      "implemented for a ground state with "
                                      "core orbitals.")
        assert not apply_cvs  # TODO: once implemented for core orbitals
        hf: ReferenceState = self.reference_state
        ret = TwoParticleDensity(
            self.mospaces, symmetry=OperatorSymmetry.HERMITIAN
        )
        p0: OneParticleDensity = self.diffdm(2)

        # constuct Kronecker Delta
        d_oo = zeros_like(hf.foo)
        d_oo.set_mask("ii", 1)

        ret.oooo = (
            + 4.0 * einsum("ik,jl->ijkl", p0.oo, d_oo)
            .antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijab,klab->ijkl", self.t2oo, self.t2oo)
        )
        ret.ooov = (
            + 2.0 * einsum("ja,ik->ijka", p0.ov, d_oo).antisymmetrise(0, 1)
        )
        ret.oovv = (
            - 1.0 * self.td2(b.oovv)
        )
        ret.ovov = (
            + 1.0 * einsum("ab,ij->iajb", p0.vv, d_oo)
            - 1.0 * einsum("jkac,ikbc->iajb", self.t2oo, self.t2oo)
        )
        ret.vvvv = (
            + 0.5 * einsum("ijab,ijcd->abcd", self.t2oo, self.t2oo)
        )
        ret.reference_state = self.reference_state
        return ret.evaluate()

    def _to_qcvars(self, gs_type: str, properties: bool = False,
                   recurse: bool = False, maxlevel: int = 2) -> dict:
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        qcvars = {}
        for level in range(2, maxlevel + 1):
            try:
                mpcorr = self.energy_correction(level)
                qcvars[f"{gs_type}{level} CORRELATION ENERGY"] = mpcorr
                qcvars[f"{gs_type}{level} TOTAL ENERGY"] = self.energy(level)
            except NotImplementedError:
                pass
            except ValueError:
                pass

        if properties:
            for level in range(2, maxlevel + 1):
                try:
                    qcvars[f"{gs_type}{level} DIPOLE"] = self.dipole_moment(level)
                except NotImplementedError:
                    pass

        if recurse:
            qcvars.update(self.reference_state.to_qcvars(properties, recurse))
        return qcvars
