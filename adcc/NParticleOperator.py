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
from enum import Enum

import libadcc

from .MoSpaces import split_spaces
from .Tensor import Tensor


class OperatorSymmetry(Enum):
    NOSYMMETRY = 0
    HERMITIAN = 1
    ANTIHERMITIAN = 2

    def to_str(self) -> str:
        return self.name.lower()


class NParticleOperator:
    def __init__(self, spaces, symmetry=OperatorSymmetry.HERMITIAN):
        """
        Construct an NParticleOperator object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.LazyMp
            MoSpaces object

        symmetry : OperatorSymmetry, optional
            Symmetry type of the operator. Can be:
            - `OperatorSymmetry.NOSYMMETRY` : No symmetry is enforced
            - `OperatorSymmetry.HERMITIAN` : Operator is Hermitian (O^\\dagger = O)
            - `OperatorSymmetry.ANTIHERMITIAN` : Operator is Antihermitian
                                                 (O^\\dagger = -O)
            Default is `OperatorSymmetry.HERMITIAN`.
        """
        if hasattr(spaces, "mospaces"):
            self.mospaces = spaces.mospaces
        else:
            self.mospaces = spaces
        self.symmetry = symmetry

        # Set reference_state if possible
        if isinstance(spaces, libadcc.ReferenceState):
            self.reference_state = spaces
        elif hasattr(spaces, "reference_state"):
            assert isinstance(spaces.reference_state, libadcc.ReferenceState)
            self.reference_state = spaces.reference_state

        occs = sorted(self.mospaces.subspaces_occupied, reverse=True)
        virts = sorted(self.mospaces.subspaces_virtual, reverse=True)
        self.orbital_subspaces = occs + virts

        # check that orbital subspaces are correct
        assert sum(self.mospaces.n_orbs(ss) for ss in self.orbital_subspaces) \
            == self.mospaces.n_orbs("f")
        self._tensors = {}

        # need to be set explicitly in One- or TwoParticleOperator inits

        self.blocks = []
        self.canonical_blocks = []

        # dictionary with factors for each block {blockstring: factor, ...}
        self.canonical_factors = {}
        self._n_particle_op = None

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape tuple of the NParticleOperator
        """
        size = self.mospaces.n_orbs("f")
        return (size, size,) * self._n_particle_op

    @property
    def size(self) -> int:
        """
        Returns the number of elements of the NParticleOperator
        """
        return np.prod(self.shape)

    @property
    def blocks_nonzero(self) -> list[str, ...]:
        """
        Returns a list of the non-zero block labels
        """
        return [b for b in self.blocks if b in self._tensors]

    def is_zero_block(self, block):
        """
        Checks if block is explicitly marked as zero block.
        Returns False if the block does not exist.
        """
        if block not in self.canonical_blocks:
            return False
        return block not in self.blocks_nonzero

    def block(self, block) -> Tensor:
        """
        Returns tensor of the given block.
        Does not create a block in case it is marked as a zero block.
        Use __getitem__ for that purpose.
        """
        from . import block as b
        if block in self.blocks_nonzero:
            return self._tensors[block]
        canonical_block, factor, perm = b.get_canonical_block(block, self.symmetry)
        if canonical_block in self.blocks_nonzero:
            return factor * self._tensors[canonical_block].transpose(perm)
        else:
            raise KeyError("The block function does not support "
                           "access to zero-blocks. Available non-zero "
                           f"blocks are: {self.blocks_nonzero}.")
        pass

    def __getitem__(self, blk) -> Tensor:
        if blk not in self.blocks:
            raise KeyError(f"Invalid block {blk} requested. "
                           f"Available blocks are: {self.blocks}.")
        if blk not in self._tensors:
            if blk in self.canonical_blocks:
                sym = libadcc.make_symmetry_operator(
                    self.mospaces, blk, self.symmetry.to_str(), "1"
                )
                self._tensors[blk] = Tensor(sym)
                return self._tensors[blk]
            else:
                from . import block as b
                c_block, factor, perm = b.get_canonical_block(
                    blk, self.symmetry
                )
                if c_block not in self._tensors:
                    sym = libadcc.make_symmetry_operator(
                        self.mospaces, c_block, self.symmetry.to_str(), "1"
                    )
                    self._tensors[c_block] = Tensor(sym)
                return factor * self._tensors[c_block].transpose(perm)
        return self._tensors[blk]

    def __getattr__(self, attr) -> Tensor:
        from . import block as b
        return self.__getitem__(b.__getattr__(attr))

    def __setitem__(self, block, tensor):
        """
        Assigns a tensor to the specified block
        """
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} assigned. "
                           f"Available blocks are: {self.blocks}.")
        spaces = split_spaces(block)
        expected_shape = tuple(self.mospaces.n_orbs(space) for space in spaces)
        if expected_shape != tensor.shape:
            raise ValueError("Invalid shape of incoming tensor. "
                             f"Expected shape {expected_shape}, but "
                             f"got shape {tensor.shape} instead.")
        self._tensors[block] = tensor

    def __setattr__(self, attr, value):
        try:
            from . import block as b
            self.__setitem__(b.__getattr__(attr), value)
        except AttributeError:
            super().__setattr__(attr, value)

    def set_zero_block(self, block):
        """
        Set a given block as zero block
        """
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} set as zero block. "
                           f"Available blocks are: {self.blocks}.")
        self._tensors.pop(block)

    def to_ndarray(self) -> np.ndarray:
        """
        Returns the NParticleOperator as a contiguous
        np.ndarray instance including all blocks
        """
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

    def _transform_to_ao(self, refstate_or_coefficients) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("implement it")

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
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        if not isinstance(other, (float, int)):
            return NotImplemented
        for b in self.blocks_nonzero:
            self[b] = self.block(b) * other
        return self

    def __add__(self, other):
        if self.symmetry == OperatorSymmetry.NOSYMMETRY \
                or other.symmetry is not OperatorSymmetry.NOSYMMETRY:
            return self.copy().__iadd__(other)
        else:
            return other.copy().__iadd__(self)

    def __sub__(self, other):
        if self.symmetry == OperatorSymmetry.NOSYMMETRY \
                or other.symmetry is not OperatorSymmetry.NOSYMMETRY:
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
    assert op1.blocks == op2.blocks
    if op1.symmetry == OperatorSymmetry.NOSYMMETRY:
        factors = op1.canonical_factors.copy()
    elif op2.symmetry == OperatorSymmetry.NOSYMMETRY:
        factors = op2.canonical_factors.copy()
    else:
        assert op1.canonical_factors == op2.canonical_factors
        factors = op1.canonical_factors.copy()
        if op1.symmetry is not op2.symmetry:
            to_remove = []
            for b in list(factors.keys()):
                spaces = split_spaces(b)
                if op1._n_particle_op == 1:
                    assert len(spaces) == 2
                    if spaces[0] != spaces[1]:
                        to_remove.append(b)
                elif op1._n_particle_op == 2:
                    assert len(spaces) == 4
                    if spaces.count(spaces[0]) in [1, 3]:
                        to_remove.append(b)
                    elif spaces[0] == spaces[1] and spaces[2] == spaces[3]:
                        to_remove.append(b)
                # remove offdiagonals
            for b in to_remove:
                factors.pop(b)
    ret = 0
    for b, factor in factors.items():
        if op1.is_zero_block(b) or op2.is_zero_block(b):
            continue
        if factor == 0:
            continue

        ret += factor * op1.block(b).dot(op2.block(b))
    return ret
