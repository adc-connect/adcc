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


class OneParticleOperator:
    def __init__(self, spaces, is_symmetric=True):
        """
        Construct an OneParticleOperator object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.Mospaces or adcc.ReferenceState or adcc.LazyMp
            MoSpaces object

        is_symmetric : bool
            Is the operator symmetric?
        """
        if hasattr(spaces, "mospaces"):
            self.mospaces = spaces.mospaces
        else:
            self.mospaces = spaces
        self.is_symmetric = is_symmetric

        # Set reference_state if possible
        if isinstance(spaces, libadcc.ReferenceState):
            self.reference_state = spaces
        elif hasattr(spaces, "reference_state"):
            assert isinstance(spaces.reference_state, libadcc.ReferenceState)
            self.reference_state = spaces.reference_state

        occs = sorted(self.mospaces.subspaces_occupied, reverse=True)
        virts = sorted(self.mospaces.subspaces_virtual, reverse=True)
        self.orbital_subspaces = occs + virts
        if self.is_symmetric:
            combs = list(
                combinations_with_replacement(self.orbital_subspaces, r=2)
            )
        else:
            combs = list(product(self.orbital_subspaces, repeat=2))
        self.blocks = ["".join(com) for com in combs]
        self._tensors = {}

    @property
    def shape(self):
        """
        Returns the shape tuple of the OneParticleOperator
        """
        size = sum(self.mospaces.n_orbs(ss) for ss in self.orbital_subspaces)
        return (size, size)

    @property
    def size(self):
        """
        Returns the number of elements of the OneParticleOperator
        """
        return np.prod(self.shape)

    @property
    def blocks_nonzero(self):
        """
        Returns a list of the non-zero block labels
        """
        return [b for b in self.blocks if b in self._tensors]

    def is_zero_block(self, block):
        """
        Checks if block is explicitly marked as zero block
        """
        if block not in self.blocks:
            return False
        return block not in self.blocks_nonzero

    def block(self, block):
        """
        Returns tensor of the given block
        """
        # TODO: sanity checks
        assert block in self.blocks_nonzero
        return self._tensors[block]

    def __getitem__(self, block):
        # TODO: sanity checks
        assert block in self.blocks
        if block not in self._tensors:
            sym = libadcc.make_symmetry_operator(
                self.mospaces, block, self.is_symmetric, "1"
            )
            self._tensors[block] = Tensor(sym)
        return self._tensors[block]

    def __setitem__(self, block, tensor):
        self.set_block(block, tensor)

    def set_block(self, block, tensor):
        """
        Assigns tensor to a given block
        """
        # TODO: sanity checks
        assert block in self.blocks
        self._tensors[block] = tensor

    def set_zero_block(self, block):
        """
        Set a given block as zero block
        """
        # TODO: sanity checks
        assert block in self.blocks
        self._tensors.pop(block)

    def to_ndarray(self):
        """
        Returns the OneParticleOperator as a contiguous
        np.ndarray instance with all blocks
        """
        ret = np.zeros((self.shape))
        row_offset = 0
        for i, sp1 in enumerate(self.orbital_subspaces):
            n_orbs1 = self.mospaces.n_orbs(sp1)
            col_offset = 0
            for j, sp2 in enumerate(self.orbital_subspaces):
                n_orbs2 = self.mospaces.n_orbs(sp2)
                if self.is_symmetric and j < i:
                    if not self.is_zero_block(sp2 + sp1):
                        dm_12 = self.block(sp2 + sp1)
                        ret[row_offset:row_offset + n_orbs1,
                            col_offset:col_offset + n_orbs2] = dm_12.to_ndarray().T
                else:
                    if not self.is_zero_block(sp1 + sp2):
                        dm_12 = self.block(sp1 + sp2)
                        ret[row_offset:row_offset + n_orbs1,
                            col_offset:col_offset + n_orbs2] = dm_12.to_ndarray()
                col_offset += n_orbs2
            row_offset += n_orbs1
        return ret

    def copy(self):
        """
        Return a deep copy of the OneParticleOperator
        """
        ret = OneParticleOperator(self.mospaces, self.is_symmetric)
        for b in self.blocks_nonzero:
            ret.set_block(b, self.block(b).copy())
        if hasattr(self, "reference_state"):
            ret.reference_state = self.reference_state
        return ret

    def __transform_to_ao(self, refstate_or_coefficients):
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
            s1, s2 = split_spaces(block)
            pref = 2.0 if (s1 != s2 and self.is_symmetric) else 1.0
            # TODO: sanity check
            dm_bb_a += pref * einsum("ip,ij,jq->pq", coeff_map[f"{s1}_a"],
                                     self[block], coeff_map[f"{s2}_a"])
            dm_bb_b += pref * einsum("ip,ij,jq->pq", coeff_map[f"{s1}_b"],
                                     self[block], coeff_map[f"{s2}_b"])
        if self.is_symmetric:
            dm_bb_a = dm_bb_a.symmetrise()
            dm_bb_b = dm_bb_b.symmetrise()
        return (dm_bb_a.evaluate(), dm_bb_b.evaluate())

    def to_ao_basis(self, refstate_or_coefficients=None):
        """
        TODO DOCME
        """
        if isinstance(refstate_or_coefficients, (dict, libadcc.ReferenceState)):
            return self.__transform_to_ao(refstate_or_coefficients)
        elif refstate_or_coefficients is None:
            if not hasattr(self, "reference_state"):
                raise ValueError("Argument reference_state is required if no "
                                 "reference_state is stored in the "
                                 "OneParticleOperator")
            return self.__transform_to_ao(self.reference_state)
        else:
            raise TypeError("Argument type not supported.")

    def __iadd__(self, other):
        if self.mospaces != other.mospaces:
            raise ValueError("Cannot add OneParticleOperators with "
                             "differing mospaces.")
        if self.is_symmetric and not other.is_symmetric:
            raise ValueError("Cannot add non-symmetric matrix "
                             "in-place to symmetric one.")

        for b in other.blocks_nonzero:
            if self.is_zero_block(b):
                self.set_block(b, other.block(b).copy())
            else:
                self.set_block(b, self.block(b) + other.block(b))

        if not self.is_symmetric and other.is_symmetric:
            for b in other.blocks_nonzero:
                if b[:2] == b[2:]:
                    continue  # Done already
                brev = b[2:] + b[:2]  # Reverse block

                obT = other.block(b).transpose()
                if not self.is_zero_block(brev):
                    obT += self.block(brev)
                self.set_block(brev, evaluate(obT))

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
        if self.is_symmetric and not other.is_symmetric:
            raise ValueError("Cannot subtract non-symmetric matrix "
                             "in-place from symmetric one.")

        for b in other.blocks_nonzero:
            if self.is_zero_block(b):
                self.set_block(b, -1.0 * other.block(b))  # The copy is implicit
            else:
                self.set_block(b, self.block(b) - other.block(b))

        if not self.is_symmetric and other.is_symmetric:
            for b in other.blocks_nonzero:
                if b[:2] == b[2:]:
                    continue  # Done already
                brev = b[2:] + b[:2]  # Reverse block

                obT = -1.0 * other.block(b).transpose()
                if not self.is_zero_block(brev):
                    obT += self.block(brev)
                self.set_block(brev, evaluate(obT))

        # Update ReferenceState pointer
        if hasattr(self, "reference_state"):
            if hasattr(other, "reference_state") \
                    and self.reference_state != other.reference_state:
                delattr(self, "reference_state")
        return self

    def __imul__(self, other):
        if not isinstance(other, (float, int)):
            return NotImplemented
        for b in self.blocks_nonzero:
            self.set_block(b, self.block(b) * other)
        return self

    def __add__(self, other):
        if not self.is_symmetric or other.is_symmetric:
            return self.copy().__iadd__(other)
        else:
            return other.copy().__iadd__(self)

    def __sub__(self, other):
        if not self.is_symmetric or other.is_symmetric:
            return self.copy().__isub__(other)
        else:
            return (-1.0 * other).__iadd__(self)

    def __mul__(self, other):
        return self.copy().__imul__(other)

    def __rmul__(self, other):
        return self.copy().__imul__(other)

    def evaluate(self):
        for b in self.blocks_nonzero:
            self.block(b).evaluate()
        return self


def product_trace(op1, op2):
    # TODO use blocks_nonzero and build the set intersection
    #      to avoid the is_zero_block( ) checks below.
    #      I'm a bit hesitant to do this right now, because I'm lacking
    #      the time at the moment to build a more sophisticated test,
    #      which could potentially catch an arising error.
    all_blocks = list(set(op1.blocks + op2.blocks))

    if op1.is_symmetric and op2.is_symmetric:
        ret = 0
        assert op1.blocks == op2.blocks
        for b in all_blocks:
            if op1.is_zero_block(b) or op2.is_zero_block(b):
                continue
            tb = b[2:] + b[:2]  # transposed block string
            if b == tb:
                ret += op1.block(b).dot(op2.block(b))
            else:
                ret += 2.0 * op1.block(b).dot(op2.block(b))
        return ret
    elif op1.is_symmetric and not op2.is_symmetric:
        ret = 0
        for b in all_blocks:
            if op1.is_zero_block(b) or op2.is_zero_block(b):
                continue
            tb = b[2:] + b[:2]  # transposed block string
            if b in op1.blocks:
                ret += op1.block(b).dot(op2.block(b))
            elif tb in op1.blocks:
                ret += op1.block(tb).transpose().dot(op2.block(b))
        return ret
    elif not op1.is_symmetric and op2.is_symmetric:
        return product_trace(op2, op1)
    else:
        ret = 0
        assert op1.blocks == op2.blocks
        for b in all_blocks:
            if op1.is_zero_block(b) or op2.is_zero_block(b):
                continue
            ret += op1.block(b).dot(op2.block(b))
        return ret
