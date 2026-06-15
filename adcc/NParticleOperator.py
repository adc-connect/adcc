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
from dataclasses import dataclass
from enum import Enum
from itertools import product
import math
import numpy as np

import libadcc

from .functions import evaluate
from .MoSpaces import split_spaces
from .Tensor import Tensor


class OperatorSymmetry(Enum):
    NOSYMMETRY = 0
    HERMITIAN = 1
    ANTIHERMITIAN = 2

    def to_str(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class BlockInfo:
    canonical: str
    factor: int
    transpose: tuple[int, ...]


class CanonicalBlock:
    def __init__(self):
        self._symmetry_factor = 0
        self._tensor = None

    @property
    def symmetry_factor(self):
        return self._symmetry_factor

    @property
    def tensor(self):
        return self._tensor

    @tensor.setter
    def tensor(self, tensor):
        self._tensor = tensor


class NParticleOperator:
    def __init__(self, spaces, n_particle_op, symmetry=OperatorSymmetry.HERMITIAN):
        """
        Construct an NParticleOperator object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.LazyMp
            MoSpaces object

        n_particle_operator: int
            Particle rank of the operator, i.e. the number of creation and
            annihilation operators involved.

        symmetry : OperatorSymmetry, optional
            Symmetry type of the operator. Can be:
            - `OperatorSymmetry.NOSYMMETRY` : No symmetry is enforced
            - `OperatorSymmetry.HERMITIAN` : Operator is Hermitian (O^\\dagger = O)
            - `OperatorSymmetry.ANTIHERMITIAN` : Operator is Antihermitian
                                                 (O^\\dagger = -O)
            Default is `OperatorSymmetry.HERMITIAN`.
        """
        self._n_particle_op = n_particle_op
        if hasattr(spaces, "mospaces"):
            self.mospaces = spaces.mospaces
        else:
            self.mospaces = spaces
        self.symmetry = symmetry

        self.reference_state = None
        # Set reference_state if possible
        if isinstance(spaces, libadcc.ReferenceState):
            self.reference_state = spaces
        elif hasattr(spaces, "reference_state"):
            assert isinstance(spaces.reference_state, libadcc.ReferenceState)
            self.reference_state = spaces.reference_state

        occs = sorted(self.mospaces.subspaces_occupied)
        virts = sorted(self.mospaces.subspaces_virtual)
        self.orbital_subspaces = (*occs, *virts)

        # check that orbital subspaces are correct
        assert sum(self.mospaces.n_orbs(ss) for ss in self.orbital_subspaces) \
            == self.mospaces.n_orbs("f")

        # Initialize all blocks; symmetry rules are applied lazily upon access.
        combs = list(product(self.orbital_subspaces,
                             repeat=2 * self._n_particle_op))
        blocks = tuple("".join((comb)) for comb in combs)

        self._block_info: dict[str, BlockInfo] = {}
        self._canonical_blocks: dict[str, CanonicalBlock] = {}

        for block in blocks:
            from .block import get_canonical_block
            bra = block[:2 * self.n_particle_op]
            ket = block[2 * self.n_particle_op:]
            canonical_block, factor, transpose = get_canonical_block(
                bra, ket, self.symmetry
            )
            self._block_info[block] = BlockInfo(
                canonical=canonical_block,
                factor=factor,
                transpose=transpose,
            )

            # add new canonical block to dictionary
            if canonical_block not in self._canonical_blocks.keys():
                self._canonical_blocks[canonical_block] = CanonicalBlock()

            # increment factor counter
            self._canonical_blocks[canonical_block]._symmetry_factor += 1

    @property
    def canonical_factors(self) -> dict[str, int]:
        """
        Returns canonical block multiplicity factors.
        """
        return {
            cb: value.symmetry_factor
            for cb, value in self._canonical_blocks.items()
        }

    @property
    def canonical_blocks(self) -> tuple[str, ...]:
        """
        Returns tuple of canonical block labels.
        """
        return tuple(self._canonical_blocks.keys())

    @property
    def blocks(self) -> tuple[str, ...]:
        """
        Returns tuple of all block labels.
        """
        return tuple(self._block_info.keys())

    @property
    def n_particle_op(self) -> int:
        """
        Returns the particle rank of the operator.
        """
        return self._n_particle_op

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Returns the shape tuple of the NParticleOperator
        """
        size = self.mospaces.n_orbs("f")
        return (size, size,) * self.n_particle_op

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
        return [cb for cb, value
                in self._canonical_blocks.items() if value.tensor is not None]

    def is_zero_block(self, block):
        """
        Checks if block is explicitly marked as zero block.
        Returns False if the block does not exist.
        """
        if block not in self.canonical_blocks:
            return False
        return block not in self.blocks_nonzero

    def block(self, block):
        """
        Returns tensor of the given block.
        Does not create a block in case it is marked as a zero block.
        Use __getitem__ for that purpose.
        """
        block_info = self._block_info.get(block, None)
        if block_info is None:
            raise KeyError(f"Invalid block {block} requested. "
                           f"Available blocks are: {self.blocks}.")

        canonical_block = self._canonical_blocks[block_info.canonical]
        if canonical_block.tensor is None:
            raise KeyError("The block function does not support "
                           "access to zero-blocks. Available non-zero "
                           f"blocks are: {self.blocks_nonzero}.")

        if block == canonical_block:
            return canonical_block.tensor

        return (
            block_info.factor
            * canonical_block.tensor.transpose(block_info.transpose)
        )

    def __getitem__(self, block) -> Tensor:
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} requested. "
                           f"Available blocks are: {self.blocks}.")

        block_info = self._block_info[block]
        canonical_block = self._canonical_blocks[block_info.canonical]

        if canonical_block.tensor is None:
            sym = libadcc.make_symmetry_operator(
                self.mospaces, block_info.canonical, self.symmetry.to_str(), "1"
            )
            canonical_block.tensor = Tensor(sym)

        if block == block_info.canonical:
            return canonical_block.tensor

        return (
            block_info.factor
            * canonical_block.tensor.transpose(block_info.transpose)
        )

    def __getattr__(self, attr) -> Tensor:
        from . import block as b
        return self.__getitem__(b.__getattr__(attr))

    def __setitem__(self, block, tensor):
        block_info = self._block_info.get(block, None)
        if block_info is None or block_info.canonical != block:
            raise KeyError(f"Invalid block {block} assigned. "
                           f"Available blocks are: {self.canonical_blocks}.")

        spaces = split_spaces(block)
        expected_shape = tuple(self.mospaces.n_orbs(space) for space in spaces)
        if expected_shape != tensor.shape:
            raise ValueError("Invalid shape of incoming tensor. "
                             f"Expected shape {expected_shape}, but "
                             f"got shape {tensor.shape} instead.")
        canonical_block = self._canonical_blocks[block_info.canonical]
        canonical_block.tensor = tensor

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
        if block not in self.canonical_blocks:
            raise KeyError(f"Invalid block {block} set as zero block. "
                           f"Available blocks are: {self.blocks}.")
        block_info = self._block_info[block]
        canonical_block = self._canonical_blocks[block_info.canonical]
        canonical_block.tensor = None

    def to_ndarray(self) -> np.ndarray:
        """
        Returns the NParticleOperator as a contiguous
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
        # build complete np.ndarray -> go through all blocks
        ret = np.zeros((self.shape))
        for block in self.blocks:
            if self.is_zero_block(block):
                continue
            spaces = split_spaces(block)
            slice_list = tuple(slices[sp] for sp in spaces)
            dm_block = self[block].to_ndarray()
            ret[slice_list] = dm_block
        return ret

    def _construct_empty(self):
        """
        Create an empty instance of an NParticleOperator
        """
        return NParticleOperator(
            self.mospaces,
            self.n_particle_op,
            symmetry=self.symmetry,
        )

    def copy(self):
        """
        Return a deep copy of the NParticleOperator
        """
        ret = self._construct_empty()
        for b in self.blocks_nonzero:
            ret[b] = self.block(b).copy()
        if self.reference_state is not None:
            ret.reference_state = self.reference_state
        return ret

    def _transform_to_ao(self, refstate):
        raise NotImplementedError("Needs to be implemented on the "
                                  f"{self.__class__.__name__} class.")

    def to_ao_basis(self, refstate=None):
        """
        Transforms the NParticleOperator to the atomic orbital
        basis using a ReferenceState. If no ReferenceState is given,
        the ReferenceState used to construct the NParticleOperator is taken instead.
        """
        if isinstance(refstate, libadcc.ReferenceState):
            return self._transform_to_ao(refstate)
        elif refstate is None:
            if self.reference_state is None:
                raise ValueError("Argument reference_state is required if no "
                                 "reference_state is stored in the "
                                 "NParticleOperator")
            return self._transform_to_ao(self.reference_state)
        else:
            raise TypeError("Argument type not supported.")

    def __iadd__(self, other):
        from . import block as b
        if self.mospaces != other.mospaces:
            raise ValueError(f"Cannot add {self.__class__.__name__}s with "
                             "differing mospaces.")
        if self.symmetry is not OperatorSymmetry.NOSYMMETRY and \
                self.symmetry is not other.symmetry:
            raise ValueError("Cannot add non-symmetric matrix "
                             "in-place to symmetric one.")

        # other.canonical_blocks is subset of self.canonical_blocks
        assert all(b in self.canonical_blocks for b in other.canonical_blocks)
        for block in other.blocks_nonzero:
            if self.is_zero_block(block):
                self[block] = other.block(block).copy()
            else:
                self[block] += other.block(block)

        if self.symmetry is OperatorSymmetry.NOSYMMETRY \
                and other.symmetry is not OperatorSymmetry.NOSYMMETRY:
            for block in self.blocks_nonzero:
                bra = block[:2 * self.n_particle_op]
                ket = block[2 * self.n_particle_op:]
                c_block, factor, transpose = b.get_canonical_block(
                    bra, ket, other.symmetry
                )
                if block == c_block:
                    continue  # Done already
                obT = 0
                if not other.is_zero_block(c_block):
                    obT = factor * other.block(c_block).transpose(transpose)
                if not self.is_zero_block(block):
                    obT += self.block(block)
                self[block] = evaluate(obT)

        # Update ReferenceState pointer
        if self.reference_state is not None:
            if other.reference_state is not None \
                    and self.reference_state != other.reference_state:
                self.reference_state = None
        return self

    def __isub__(self, other):
        from . import block as b
        if self.mospaces != other.mospaces:
            raise ValueError(f"Cannot add {self.__class__.__name__}s with "
                             "differing mospaces.")
        if self.symmetry is not OperatorSymmetry.NOSYMMETRY and \
                self.symmetry is not other.symmetry:
            raise ValueError("Cannot add non-symmetric matrix "
                             "in-place to symmetric one.")

        # other.canonical_blocks is subset of self.canonical_blocks
        assert all(b in self.canonical_blocks for b in other.canonical_blocks)
        for block in other.blocks_nonzero:
            if self.is_zero_block(block):
                self[block] = -1.0 * other.block(block)  # The copy is implicit
            else:
                self[block] -= other.block(block)

        if self.symmetry is OperatorSymmetry.NOSYMMETRY \
                and other.symmetry is not OperatorSymmetry.NOSYMMETRY:
            for block in self.blocks_nonzero:
                bra = block[:2 * self.n_particle_op]
                ket = block[2 * self.n_particle_op:]
                c_block, factor, transpose = b.get_canonical_block(
                    bra, ket, other.symmetry
                )
                if block == c_block:
                    continue  # Done already
                obT = 0
                if not other.is_zero_block(c_block):
                    obT = -1 * factor * other.block(c_block).transpose(transpose)
                if not self.is_zero_block(block):
                    obT += self.block(block)
                self[block] = evaluate(obT)

        # Update ReferenceState pointer
        if self.reference_state is not None:
            if other.reference_state is not None \
                    and self.reference_state != other.reference_state:
                self.reference_state = None
        return self

    def __imul__(self, other):
        if not isinstance(other, (float, int)):
            return NotImplemented
        for b in self.blocks_nonzero:
            self[b] = self.block(b) * other
        return self

    def __add__(self, other):
        if (
            self.symmetry is not other.symmetry
            and self.symmetry is not OperatorSymmetry.NOSYMMETRY
            and other.symmetry is not OperatorSymmetry.NOSYMMETRY
        ):
            raise ValueError("Addition of Hermitian and Antihermitian "
                             "operators is not implemented.")

        if self.symmetry is OperatorSymmetry.NOSYMMETRY \
                or other.symmetry is not OperatorSymmetry.NOSYMMETRY:
            return self.copy().__iadd__(other)
        else:
            return other.copy().__iadd__(self)

    def __sub__(self, other):
        if (
            self.symmetry is not other.symmetry
            and self.symmetry is not OperatorSymmetry.NOSYMMETRY
            and other.symmetry is not OperatorSymmetry.NOSYMMETRY
        ):
            raise ValueError("Substraction of Hermitian and Antihermitian "
                             "operators is not implemented.")

        if self.symmetry is OperatorSymmetry.NOSYMMETRY \
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

    def set_random(self):
        for b in self.canonical_blocks:
            self[b].set_random()
        return self


def product_trace(op1, op2) -> float:
    """
    Compute the expectation value (inner product) of two N-particle operators.

    For the special case of a density operator and a general operator, the
    inner product is identical in the AO and MO basis.

    Parameters
    ----------
    op1, op2 : NParticleOperator or NParticleDensity
        Operators or densities to compute the inner product of.

    Returns
    -------
    float
        The trace / expectation value <op1, op2>.
    """
    # TODO use blocks_nonzero and build the set intersection
    #      to avoid the is_zero_block( ) checks below.
    #      I'm a bit hesitant to do this right now, because I'm lacking
    #      the time at the moment to build a more sophisticated test,
    #      which could potentially catch an arising error.
    if op1.n_particle_op != op2.n_particle_op:
        raise TypeError(
            "Both operators must have the same number of N-particle operators "
            f"(got {op1.n_particle_op} and {op2.n_particle_op})."
        )
    assert op1.blocks == op2.blocks
    if op1.symmetry == OperatorSymmetry.NOSYMMETRY:
        factors = op1.canonical_factors.copy()
    elif op2.symmetry == OperatorSymmetry.NOSYMMETRY:
        factors = op2.canonical_factors.copy()
    else:
        assert op1.canonical_factors == op2.canonical_factors
        factors = op1.canonical_factors.copy()
        # Off-diagonal terms can be removed since a Hermitian times
        # an anti-Hermitian operator yields for off diagonal terms
        # (e.g. for a one-particle operator) ov * ov and vo * vo = (ov).T * (−ov.T),
        # which cancel exactly to zero.
        if op1.symmetry is not op2.symmetry:
            to_remove = []
            for b in list(factors.keys()):
                spaces = split_spaces(b)
                n = op1.n_particle_op
                if spaces[:n] != spaces[n:]:
                    to_remove.append(b)
            for b in to_remove:
                factors.pop(b)
    ret = 0
    for b, factor in factors.items():
        if op1.is_zero_block(b) or op2.is_zero_block(b):
            continue

        ret += (1 / math.factorial(op1.n_particle_op) ** 2 * factor
                * op1.block(b).dot(op2.block(b)))
    return ret
