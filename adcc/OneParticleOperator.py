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
from enum import Enum

import libadcc

from .functions import evaluate, einsum
from .MoSpaces import split_spaces
from .Tensor import Tensor
from .NParticleOperator import NParticleOperator, OperatorSymmetry

class OneParticleOperator(NParticleOperator):
    def __init__(self, spaces, symmetry=OperatorSymmetry.HERMITIAN):
        """
        Construct an OneParticleOperator object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.LazyMp
            MoSpaces object

        symmetry : OperatorSymmetry, optional
        """
        super().__init__(spaces, symmetry=symmetry)
        self._n_particle_op = 1

        # Initialize all blocks; symmetry rules are applied lazily upon access.
        combs = list(product(self.orbital_subspaces, repeat=2))
        self.blocks = ["".join((comb)) for comb in combs]
        if self.symmetry is not OperatorSymmetry.NOSYMMETRY:
            self.canonical_blocks = [
                "".join(sorted(comb)) for comb in combinations_with_replacement(
                    self.orbital_subspaces, r=2)
            ]
        else:
            self.canonical_blocks = self.blocks.copy()
        self.canonical_factors = {block: 1 for block in self.canonical_blocks}
        for block in self.blocks:
            if block in self.canonical_blocks:
                continue
            from .block import get_canonical_block
            canonical_block, _, _ = get_canonical_block(block, self.symmetry)
            assert canonical_block in self.canonical_factors.keys()
            self.canonical_factors[canonical_block] += 1

    def to_ndarray(self):
        """
        Returns the OneParticleOperator as a contiguous
        np.ndarray instance including all blocks
        """
        # offsets to start index of spaces
        offsets = {
            sp: sum(
                self.mospaces.n_orbs(ss)
                for ss in self.orbital_subspaces[:self.orbital_subspaces.index(sp)]
            )
            for sp in self.orbital_subspaces
        }
        # slices for each space
        slices = {
            sp: slice(offsets[sp], offsets[sp] + self.mospaces.n_orbs(sp))
            for sp in self.orbital_subspaces
        }
        ret = np.zeros((self.shape))
        for block in self.blocks_nonzero:
            sp1, sp2 = split_spaces(block)
            rowslice, colslice = slices[sp1], slices[sp2]
            dm_block = self[block].to_ndarray()
            ret[rowslice, colslice] = dm_block
            if sp1 != sp2 and self.symmetry is not OperatorSymmetry.NOSYMMETRY:
                factor = 1.0 if self.symmetry == OperatorSymmetry.HERMITIAN else -1.0
                ret[colslice, rowslice] = factor * dm_block.T
        return ret

    def copy(self):
        """
        Return a deep copy of the OneParticleOperator
        """
        ret = OneParticleOperator(self.mospaces, self.symmetry)
        for b in self.blocks_nonzero:
            ret[b] = self.block(b).copy()
        if hasattr(self, "reference_state"):
            ret.reference_state = self.reference_state
        return ret

    def _transform_to_ao(self, refstate_or_coefficients) -> tuple[Tensor, Tensor]:
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
        block_coeffs_ov = {
            OperatorSymmetry.NOSYMMETRY: 1.0,
            OperatorSymmetry.HERMITIAN: 2.0,
            OperatorSymmetry.ANTIHERMITIAN: 0.0,
        }
        for block in self.blocks_nonzero:
            # only canonical blocks
            s1, s2 = split_spaces(block)
            # hermitian operators: scale off-diagonal block of symmetric operator 
            # by 2 because only one of the blocks is actually present
            if s1 != s2:
                pref = block_coeffs_ov[self.symmetry]
            else:
                pref = 1.0
            if pref == 0.0:
                continue
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

    def to_ao_basis(self, refstate_or_coefficients=None):
        """
        Transforms the NParticleOperator to the atomic orbital
        basis using a ReferenceState or a coefficient map. If no
        ReferenceState or coefficient map is given, the ReferenceState
        used to construct the NParticleOperator is taken instead.
        """
        if isinstance(refstate_or_coefficients, (dict, libadcc.ReferenceState)):
            return self._transform_to_ao(refstate_or_coefficients)
        elif refstate_or_coefficients is None:
            if not hasattr(self, "reference_state"):
                raise ValueError("Argument reference_state is required if no "
                                 "reference_state is stored in the "
                                 "NParticleOperator")
            return self._transform_to_ao(self.reference_state)
        else:
            raise TypeError("Argument type not supported.")

    def __iadd__(self, other):
        if self.mospaces != other.mospaces:
            raise ValueError("Cannot add OneParticleOperators with "
                             "differing mospaces.")
        if self.symmetry is not OperatorSymmetry.NOSYMMETRY \
                and other.symmetry == OperatorSymmetry.NOSYMMETRY:
            raise ValueError("Cannot add non-symmetric matrix "
                             "in-place to symmetric one.")

        for b in other.blocks_nonzero:
            if self.is_zero_block(b):
                self[b] = other.block(b).copy()
            else:
                self[b] = self.block(b) + other.block(b)

        if self.symmetry == OperatorSymmetry.NOSYMMETRY \
                and other.symmetry is not OperatorSymmetry.NOSYMMETRY:
            for b in other.blocks_nonzero:
                if b[:2] == b[2:]:
                    continue  # Done already
                brev = b[2:] + b[:2]  # Reverse block

                obT = other.block(b).transpose()
                if not self.is_zero_block(brev):
                    factor = 1.0 if other.symmetry == OperatorSymmetry.HERMITIAN else -1.0
                    obT += factor * self.block(brev)
                self[brev] = evaluate(obT)

        # Update ReferenceState pointer
        if hasattr(self, "reference_state"):
            if hasattr(other, "reference_state") \
                    and self.reference_state != other.reference_state:
                delattr(self, "reference_state")
        return self

    def __isub__(self, other):
        if self.mospaces != other.mospaces:
            raise ValueError("Cannot subtract OneParticleOperators with "
                             "differing mospaces.")
        if self.symmetry is not OperatorSymmetry.NOSYMMETRY \
                and other.symmetry == OperatorSymmetry.NOSYMMETRY:
            raise ValueError("Cannot subtract non-symmetric matrix "
                             "in-place from symmetric one.")

        for b in other.blocks_nonzero:
            if self.is_zero_block(b):
                self[b] = -1.0 * other.block(b)  # The copy is implicit
            else:
                self[b] = self.block(b) - other.block(b)

        if self.symmetry == OperatorSymmetry.NOSYMMETRY \
                and other.symmetry is not OperatorSymmetry.NOSYMMETRY:
            for b in other.blocks_nonzero:
                if b[:2] == b[2:]:
                    continue  # Done already
                brev = b[2:] + b[:2]  # Reverse block

                obT = -1.0 * other.block(b).transpose()
                if not self.is_zero_block(brev):
                    factor = 1.0 if other.symmetry == OperatorSymmetry.HERMITIAN else -1.0
                    obT += factor * self.block(brev)
                self[brev] = evaluate(obT)

        # Update ReferenceState pointer
        if hasattr(self, "reference_state"):
            if hasattr(other, "reference_state") \
                    and self.reference_state != other.reference_state:
                delattr(self, "reference_state")
        return self
