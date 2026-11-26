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
from typing import TYPE_CHECKING, TypeVar

import libadcc

from .block import split_spaces
from .functions import einsum
from .NParticleOperator import NParticleOperator, OperatorSymmetry

if TYPE_CHECKING:
    from .GroundState import GroundState
    from .MoSpaces import MoSpaces
    from .ReferenceState import ReferenceState


_T = TypeVar("_T", bound="OneParticleOperator")


class OneParticleOperator(NParticleOperator):
    def __init__(
        self,
        spaces: "MoSpaces | ReferenceState | GroundState",
        symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN,
    ) -> None:
        """
        Construct an OneParticleOperator object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.GroundState
            Defines the orbital spaces and therefore the blocks of the operator.
            A ReferenceState or GroundState additionally sets ``reference_state``.

        symmetry : OperatorSymmetry, optional
        """
        super().__init__(spaces, n_particle_op=1, symmetry=symmetry)

    def _construct_empty(self: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        """
        Create an empty instance of an OneParticleOperator
        """
        return self.__class__(
            self.mospaces,
            symmetry=self.symmetry,
        )

    def _transform_to_ao(self, refstate: "ReferenceState") -> tuple[libadcc.Tensor, libadcc.Tensor]:
        if not self.blocks_nonzero:
            raise ValueError(
                "At least one non-zero block is needed to transform the OneParticleOperator."
            )

        coeff_map: dict[str, libadcc.Tensor] = {}
        for sp in self.orbital_subspaces:
            coeff_map[sp + "_a"] = refstate.orbital_coefficients_alpha(sp + "b")
            coeff_map[sp + "_b"] = refstate.orbital_coefficients_beta(sp + "b")
        ovlp: libadcc.Tensor = refstate.operators.overlap_ao

        dm_bb_a: int | libadcc.Tensor = 0
        dm_bb_b: int | libadcc.Tensor = 0

        for block in self.blocks_nonzero:
            # only canonical blocks
            s1, s2 = split_spaces(block)
            # (anti-)hermitian operators: scale off-diagonal block of operator
            # by 2 because only one of the blocks is actually present
            pref = self.canonical_factors[block]

            dm_bb_a += pref * einsum(
                "lm,im,ij,jv,vs->ls",
                ovlp,
                coeff_map[f"{s1}_a"],
                self.block(block),
                coeff_map[f"{s2}_a"],
                ovlp,
            )
            dm_bb_b += pref * einsum(
                "lm,im,ij,jv,vs->ls",
                ovlp,
                coeff_map[f"{s1}_b"],
                self.block(block),
                coeff_map[f"{s2}_b"],
                ovlp,
            )

        assert isinstance(dm_bb_a, libadcc.Tensor) and isinstance(dm_bb_b, libadcc.Tensor)
        if self.symmetry is OperatorSymmetry.HERMITIAN:
            dm_bb_a = dm_bb_a.symmetrise()
            dm_bb_b = dm_bb_b.symmetrise()
        elif self.symmetry is OperatorSymmetry.ANTIHERMITIAN:
            dm_bb_a = dm_bb_a.antisymmetrise()
            dm_bb_b = dm_bb_b.antisymmetrise()
        return (dm_bb_a.evaluate(), dm_bb_b.evaluate())
