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
import numpy as np

import libadcc

from . import block as b
from .functions import direct_sum
from .GroundState import GroundState
from .misc import cached_member_function
from .MoSpaces import split_spaces
from .OneParticleDensity import OneParticleDensity
from .TwoParticleDensity import TwoParticleDensity


class LazyRoDodsMp(GroundState):
    @cached_member_function()
    def ts1(self, space: str) -> libadcc.Tensor:
        split = split_spaces(space)
        if len(split) != 2 or split[1] != b.v:
            raise ValueError(f"Invalid space '{space}' for first-order singles amplitudes.")
        # negative sign, because the denominator has to be (e_i - e_a) instead of (e_a - e_i)
        return -self.reference_state.operators.fock[space] / self.df(space)

    @cached_member_function()
    def td1(self, space: str) -> libadcc.Tensor:
        hf = self.reference_state
        sp = split_spaces(space)
        if len(sp) != 4 or sp[2] != b.v or sp[3] != b.v:
            raise ValueError(f"Invalid space '{space}' for first-order doubles amplitudes.")
        # since ia and jb are connected on the df objects, it is sufficient
        # to only symmetrise the virtual orbitals resulting in
        # ia jb + ib ja
        # which is identical to the expression if only the occupied indices
        # would have been symmetrised
        # -> resulting expression has both ij and ab symmetry
        # if we instead would build the denominator from orbital energies
        # occupied and virtual indices both have to be symmetrized!
        return hf.eri(space) / direct_sum(
            "ia,jb->ijab", self.df(sp[0] + sp[2]), self.df(sp[1] + sp[3])
        ).symmetrise((2, 3))

    @cached_member_function()
    def energy_correction(self, level: int = 2) -> float:
        if level <= 0:
            raise ValueError(f"Level has to be a positive non-zero integer, got {level}.")
        if level == 2 and not self.has_core_occupied_space:
            return self.ts1(b.ov).dot(
                self.reference_state.operators.fock.ov
            ) - 0.25 * self.t2oo.dot(self.reference_state.oovv)
        else:
            raise NotImplementedError(
                f"{'CVS-RO-DODS-MP' if self.has_core_occupied_space else 'RO-DODS-MP'} energy "
                f"correction not implemented for level {level}."
            )

    def energy(self, level: int = 2) -> float:
        if level < 0:
            raise ValueError(f"Level has to be a positive integer, got {level}.")
        elif level == 0:
            # sum of orbital energies
            # TODO: core orbitals and frozen core
            return self.reference_state.nuclear_repulsion_energy + np.einsum(
                "i->", self.reference_state.foo.diagonal().to_ndarray()
            )
        # collect all the energy corrections
        res = self.reference_state.energy_scf
        for lv in range(2, level + 1):
            res += self.energy_correction(lv)
        return res

    def diffdm(self, level: int = 2, apply_cvs: bool = False) -> OneParticleDensity:
        raise NotImplementedError("Diffdm not implemented for RO-DODS-MP ground state.")

    def density(self, level: int = 2, apply_cvs: bool = False) -> OneParticleDensity:
        raise NotImplementedError("Density not implemented for RO-DODS-MP ground state.")

    def diffdm_2p(self, level: int = 2, apply_cvs: bool = False) -> TwoParticleDensity:
        raise NotImplementedError("Diffdm not implemented for RO-DODS-MP ground state.")

    def density_2p(self, level: int = 2, apply_cvs: bool = False) -> TwoParticleDensity:
        raise NotImplementedError("Diffdm not implemented for RO-DODS-MP ground state.")

    def second_order_dm_correction(self, apply_cvs: bool = False) -> OneParticleDensity:
        raise NotImplementedError(
            "second_order_dm_correction not implemented for RO-DODS-MP ground state."
        )

    def third_order_dm_correction(self, apply_cvs: bool = False) -> OneParticleDensity:
        raise NotImplementedError(
            "third_order_dm_correction not implemented for RO-DODS-MP ground state."
        )

    def first_order_dm_correction_2p(self, apply_cvs: bool = False) -> TwoParticleDensity:
        raise NotImplementedError(
            "first_order_dm_correction_2p not implemented for RO-DODS-MP ground state."
        )

    def second_order_dm_correction_2p(self, apply_cvs: bool = False) -> TwoParticleDensity:
        raise NotImplementedError(
            "second_order_dm_correction_2p not implemented for RO-DODS-MP ground state."
        )

    def to_qcvars(self, properties: bool = False, recurse: bool = False, maxlevel: int = 2) -> dict:
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        return self._to_qcvars(
            gs_type="RO-DODS-MP", properties=properties, recurse=recurse, maxlevel=maxlevel
        )
