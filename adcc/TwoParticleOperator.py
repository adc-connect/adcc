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
import numpy as np
from itertools import product, combinations_with_replacement, combinations

import libadcc

from .functions import evaluate, einsum
from .MoSpaces import split_spaces
from .Tensor import Tensor
from .NParticleOperator import NParticleOperator, OperatorSymmetry

class TwoParticleOperator(NParticleOperator):
    def __init__(self, spaces, symmetry=OperatorSymmetry.HERMITIAN):
        """
        Construct an TwoParticleOperator object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.LazyMp
            MoSpaces object

        symmetry : OperatorSymmetry, optional
        """
        super().__init__(spaces, n_particle_op=1, symmetry=symmetry)

    def _transform_to_ao(self, refstate_or_coefficients) -> tuple[Tensor, Tensor]:
        if not len(self.blocks_nonzero):
            raise ValueError("At least one non-zero block is needed to "
                             "transform the TwoParticleOperator.")
        if isinstance(refstate_or_coefficients, libadcc.ReferenceState):
            hf = refstate_or_coefficients
            coeff_map = {}
            for sp in self.orbital_subspaces:
                coeff_map[sp + "_a"] = hf.orbital_coefficients_alpha(sp + "b")
                coeff_map[sp + "_b"] = hf.orbital_coefficients_beta(sp + "b")
        else:
            coeff_map = refstate_or_coefficients

        dm_bb_a = 0
        dm_bb_b = 0

        for block in self.blocks_nonzero:
            # only canonical blocks
            spaces = split_spaces(block)
            assert len(spaces) == 4
            factor = self.canonical_factors[block]
            # factor = 1
            if self.symmetry == OperatorSymmetry.ANTIHERMITIAN:
                if spaces[:2] != spaces[2:]:  # non-diagonal block
                    continue
            print(block, factor)
            dm_bb_a += factor * einsum("ip,jq,ijkl,kr,ls->pqrs",
                                       coeff_map[f"{spaces[0]}_a"],
                                       coeff_map[f"{spaces[1]}_a"],
                                       self[block],
                                       coeff_map[f"{spaces[2]}_a"],
                                       coeff_map[f"{spaces[3]}_a"])
            dm_bb_b += factor * einsum("ip,jq,ijkl,kr,ls->pqrs",
                                       coeff_map[f"{spaces[0]}_b"],
                                       coeff_map[f"{spaces[1]}_b"],
                                       self[block],
                                       coeff_map[f"{spaces[2]}_b"],
                                       coeff_map[f"{spaces[3]}_b"])
        # dm_bb_a = dm_bb_a.antisymmetrise(0, 1).antisymmetrise(2, 3)
        # dm_bb_b = dm_bb_b.antisymmetrise(0, 1).antisymmetrise(2, 3)
        # if self.symmetry == OperatorSymmetry.HERMITIAN:
            # dm_bb_a = dm_bb_a.symmetrise((0, 2), (1, 3))
            # dm_bb_b = dm_bb_b.symmetrise((0, 2), (1, 3))
        # elif self.symmetry == OperatorSymmetry.ANTIHERMITIAN:
        #     dm_bb_a = dm_bb_a.antisymmetrise((0, 2), (1, 3))
        #     dm_bb_b = dm_bb_b.antisymmetrise((0, 2), (1, 3))
        return (dm_bb_a.evaluate(), dm_bb_b.evaluate())
