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
from itertools import product, combinations_with_replacement

import libadcc

from .functions import evaluate, einsum
from .MoSpaces import split_spaces
from .Tensor import Tensor
from .NParticleOperator import OperatorSymmetry
from .NParticleDensity import NParticleDensity


class OneParticleDensity(NParticleDensity):
    def __init__(self, spaces, symmetry=OperatorSymmetry.HERMITIAN):
        """
        Construct an OneParticleDensity object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.LazyMp
            MoSpaces object

        symmetry : OperatorSymmetry, optional
        """
        super().__init__(spaces, n_particle_op=1, symmetry=symmetry)

    def _construct_empty(self):
        """
        Create an empty instance of an OneParticleDensity
        """
        return self.__class__(
            self.mospaces,
            symmetry=self.symmetry,
        )

    def _transform_to_ao(self, refstate_or_coefficients):
        if not len(self.blocks_nonzero):
            raise ValueError("At least one non-zero block is needed to "
                             "transform the OneParticleOperator.")
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
            s1, s2 = split_spaces(block)
            # (anti-)hermitian operators: scale off-diagonal block of operator
            # by 2 because only one of the blocks is actually present
            pref = self.canonical_factors[block]
            dm_bb_a += pref * einsum("ip,ij,jq->pq", coeff_map[f"{s1}_a"],
                                     self[block], coeff_map[f"{s2}_a"])
            dm_bb_b += pref * einsum("ip,ij,jq->pq", coeff_map[f"{s1}_b"],
                                     self[block], coeff_map[f"{s2}_b"])
        if self.symmetry == OperatorSymmetry.HERMITIAN:
            dm_bb_a = dm_bb_a.symmetrise()
            dm_bb_b = dm_bb_b.symmetrise()
        if self.symmetry == OperatorSymmetry.ANTIHERMITIAN:
            dm_bb_a = dm_bb_a.antisymmetrise()
            dm_bb_b = dm_bb_b.antisymmetrise()
        return (dm_bb_a.evaluate(), dm_bb_b.evaluate())
