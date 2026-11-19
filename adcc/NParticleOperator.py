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
import math
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

import libadcc

from . import block as b
from .MoSpaces import MoSpaces
from .Tensor import Tensor
from .typing import FloatArray

if TYPE_CHECKING:
    from .GroundState import GroundState
    from .ReferenceState import ReferenceState


_T = TypeVar("_T", bound="NParticleOperator")


class OperatorSymmetry(Enum):
    NOSYMMETRY = 0
    HERMITIAN = 1
    ANTIHERMITIAN = 2

    def to_str(self) -> str:
        return self.name.lower()


@dataclass(frozen=True)
class BlockInfo:
    # name of the canonical block
    canonical: str
    # the factor (+-1) and the transpose that has to be applied to construct the given
    # non-canonical block from the canonical operator block.
    factor: int
    transpose: tuple[int, ...]

    def construct_tensor_from_canonical(self, canonical_tensor: libadcc.Tensor) -> libadcc.Tensor:
        """
        Constructs the tensor of the current block from ``canonical_tensor``, the tensor of the
        canonical block by applying the symmetry factor and the transpose. If the current block
        is the canonical block, ``canonical_tensor`` is returned unchanged.
        """
        if self.transpose == tuple(range(len(self.transpose))):
            assert self.factor == 1
            return canonical_tensor
        # Note: the non-canonical block is not actually evaluated and allocated in memory!
        # Consequently, numerical changes in the canonical block will also be reflected in the
        # non-canonical block even if they are applied after the non-canonical block was
        # constructed.
        return self.factor * canonical_tensor.transpose(self.transpose)


class CanonicalBlock:
    def __init__(self) -> None:
        self._symmetry_factor: int = 1
        self._tensor: libadcc.Tensor | None = None

    @property
    def symmetry_factor(self) -> int:
        """
        The number of blocks that can be mapped onto the canonical block
        including the canonical block itself. Therefore, ``symmetry_factor`` is always
        at least 1.
        """
        return self._symmetry_factor

    @property
    def tensor(self) -> libadcc.Tensor | None:
        return self._tensor

    @tensor.setter
    def tensor(self, tensor: libadcc.Tensor | None) -> None:
        self._tensor = tensor


class NParticleOperator:
    def __init__(
        self,
        spaces: "MoSpaces | ReferenceState | GroundState",
        n_particle_op: int,
        symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN,
    ) -> None:
        """
        Construct an NParticleOperator object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.GroundState
            Defines the orbital spaces and therefore the blocks of the operator.
            A ReferenceState or GroundState additionally sets ``reference_state``.

        n_particle_op : int
            Particle rank of the operator, i.e. the number of creation or
            annihilation operators involved.

        symmetry : OperatorSymmetry, optional
            Symmetry type of the operator. Can be:
            - `OperatorSymmetry.NOSYMMETRY` : No symmetry is enforced
            - `OperatorSymmetry.HERMITIAN` : Operator is Hermitian (O^\\dagger = O)
            - `OperatorSymmetry.ANTIHERMITIAN` : Operator is Antihermitian
                                                 (O^\\dagger = -O)
            Default is `OperatorSymmetry.HERMITIAN`.
        """
        from .GroundState import GroundState
        from .ReferenceState import ReferenceState

        if n_particle_op <= 0:
            raise ValueError(f"n_particle_op has to be a positive integer, got {n_particle_op}.")
        self._n_particle_op: int = n_particle_op

        self.mospaces: MoSpaces
        if isinstance(spaces, MoSpaces):
            self.mospaces = spaces
        elif isinstance(spaces, (ReferenceState, GroundState)):
            self.mospaces = spaces.mospaces
        else:
            raise TypeError(
                f"Invalid spaces object of type {spaces.__class__.__name__}. Supported are "
                "MoSpaces, ReferenceState and GroundState."
            )
        if not isinstance(symmetry, OperatorSymmetry):
            raise TypeError(
                f"symmetry needs to be an OperatorSymmetry, got {symmetry.__class__.__name__}."
            )
        self.symmetry: OperatorSymmetry = symmetry

        # libadcc.ReferenceState is not sufficient, because the MO->AO transformation might
        # require the overlap matrix, which is only available through adcc.ReferenceState
        self.reference_state: ReferenceState | None = None
        if isinstance(spaces, ReferenceState):
            self.reference_state = spaces
        elif isinstance(spaces, GroundState):
            self.reference_state = spaces.reference_state

        occs = sorted(self.mospaces.subspaces_occupied)
        virts = sorted(self.mospaces.subspaces_virtual)
        self.orbital_subspaces: tuple[str, ...] = (*occs, *virts)

        # check that orbital subspaces are correct
        assert sum(
            self.mospaces.n_orbs(ss) for ss in self.orbital_subspaces
        ) == self.mospaces.n_orbs("f")

        # Initialize all blocks; symmetry rules are applied lazily upon access.
        blocks = (
            "".join(comb)
            for comb in product(self.orbital_subspaces, repeat=2 * self._n_particle_op)
        )

        self._block_info: dict[str, BlockInfo] = {}
        self._canonical_blocks: dict[str, CanonicalBlock] = {}

        for block in blocks:
            bra = block[: 2 * self.n_particle_op]
            ket = block[2 * self.n_particle_op :]
            canonical_block, factor, transpose = b.get_canonical_block(bra, ket, self.symmetry)
            self._block_info[block] = BlockInfo(
                canonical=canonical_block,
                factor=factor,
                transpose=transpose,
            )

            # add new canonical block to dictionary
            if canonical_block not in self._canonical_blocks:
                self._canonical_blocks[canonical_block] = CanonicalBlock()
            else:
                # increment factor counter
                self._canonical_blocks[canonical_block]._symmetry_factor += 1

    @property
    def canonical_factors(self) -> dict[str, int]:
        """
        Returns canonical block multiplicity factors, i.e., the number of blocks that can
        be mapped onto a given canonical block (including the canonical block itself).
        """
        return {cb: value.symmetry_factor for cb, value in self._canonical_blocks.items()}

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
        return (
            size,
            size,
        ) * self.n_particle_op

    @property
    def size(self) -> int:
        """
        Returns the number of elements of the NParticleOperator
        """
        return int(np.prod(self.shape))

    @property
    def blocks_nonzero(self) -> tuple[str, ...]:
        """
        Returns a tuple of the non-zero canonical block labels
        """
        return tuple(cb for cb, value in self._canonical_blocks.items() if value.tensor is not None)

    def is_zero_block(self, block: str) -> bool:
        """
        Checks if ``block`` is explicitly marked as zero block. Non-canonical blocks are resolved
        to their canonical block.
        """
        block_info = self._block_info.get(block, None)
        if block_info is None:
            raise ValueError(f"Invalid block {block}. Available blocks are {self.blocks}.")
        return self._canonical_blocks[block_info.canonical].tensor is None

    def block(self, block: str) -> libadcc.Tensor:
        """
        Returns the tensor of the given block if the block is set.
        For canonical blocks the stored tensor is returned
        (in-place modification also modifies the operator). For non-canonical blocks a lazy,
        unevaluated tensor built from the canonical block is returned.
        Note that no zero tensor is created if the block is not set (marked as zero block).
        Use :py:meth:`NParticleOperator.__getitem__` for that purpose.
        """
        block_info = self._block_info.get(block, None)
        if block_info is None:
            raise KeyError(f"Invalid block {block} requested. Available blocks are {self.blocks}.")

        canonical_block = self._canonical_blocks[block_info.canonical]
        if canonical_block.tensor is None:
            raise KeyError(
                "The block function does not support access to zero-blocks. Available "
                f"non-zero canonical blocks are {self.blocks_nonzero}."
            )
        return block_info.construct_tensor_from_canonical(canonical_block.tensor)

    def __getitem__(self, block: str) -> libadcc.Tensor:
        block_info = self._block_info.get(block, None)
        if block_info is None:
            raise KeyError(f"Invalid block {block} requested. Available blocks are {self.blocks}.")

        canonical_block = self._canonical_blocks[block_info.canonical]
        if canonical_block.tensor is None:
            sym = libadcc.make_symmetry_operator(
                self.mospaces, block_info.canonical, self.symmetry.to_str(), "1"
            )
            canonical_block.tensor = Tensor(sym)
        return block_info.construct_tensor_from_canonical(canonical_block.tensor)

    def __getattr__(self, attr: str) -> libadcc.Tensor:
        block = b.__getattr__(attr)  # AttributeError for invalid block shorthands
        try:
            # Note: This has the VERY weird side effect that 'hasattr(op, "ov")'
            # allocates a new tensor for the ov block if it was a zero block before
            # but I don't see how we can fix that without changing the behaviour of
            # __getitem__
            return self.__getitem__(block)
        except KeyError as e:
            raise AttributeError(
                f"{self.__class__.__name__} has no block {block} (derived from {attr})."
            ) from e

    def __setitem__(self, block: str, tensor: libadcc.Tensor) -> None:
        if not isinstance(tensor, libadcc.Tensor):
            raise TypeError(
                f"Expected a libadcc.Tensor for block {block}, got {tensor.__class__.__name__}."
            )
        canonical_block = self._canonical_blocks.get(block, None)
        if canonical_block is None:
            raise KeyError(
                f"Invalid or non-canonical block {block} assigned. "
                f"Available canonical blocks are {self.canonical_blocks}."
            )

        spaces = b.split_spaces(block)
        expected_shape = tuple(self.mospaces.n_orbs(space) for space in spaces)
        if expected_shape != tensor.shape:
            raise ValueError(
                "Invalid shape of incoming tensor. "
                f"Expected shape {expected_shape}, but got shape {tensor.shape} instead."
            )
        canonical_block.tensor = tensor

    def __setattr__(self, attr: str, value: Any) -> None:
        try:
            block = b.__getattr__(attr)
        except AttributeError:  # some regular attribute and no block shorthand
            return super().__setattr__(attr, value)
        self.__setitem__(block, value)

    def set_zero_block(self, block: str) -> None:
        """
        Set a given canonical block as zero block
        """
        canonical_block = self._canonical_blocks.get(block, None)
        if canonical_block is None:
            raise KeyError(
                f"Invalid or non-canonical block {block} set as zero block. "
                f"Available canonical blocks are {self.canonical_blocks}."
            )
        canonical_block.tensor = None

    def to_ndarray(self) -> FloatArray[tuple[int, ...]]:
        """
        Returns the NParticleOperator as a contiguous np.ndarray instance including all blocks
        """
        # offsets to start index of spaces
        offsets = {
            sp: sum(
                self.mospaces.n_orbs(ss)
                for ss in self.orbital_subspaces[: self.orbital_subspaces.index(sp)]
            )
            for sp in self.orbital_subspaces
        }
        # slices for each space
        slices = {
            sp: slice(offsets[sp], offsets[sp] + self.mospaces.n_orbs(sp))
            for sp in self.orbital_subspaces
        }
        # build complete np.ndarray -> go through all blocks
        ret = np.zeros(self.shape, dtype=np.float64)
        for block in self.blocks:
            if self.is_zero_block(block):
                continue
            spaces = b.split_spaces(block)
            slice_list = tuple(slices[sp] for sp in spaces)
            dm_block = self.block(block).to_ndarray()
            ret[slice_list] = dm_block
        return ret

    def _construct_empty(self: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        """
        Create a new empty instance of an NParticleOperator
        """
        return self.__class__(
            self.mospaces,
            self.n_particle_op,
            symmetry=self.symmetry,
        )

    def copy(self: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        """
        Return a deep copy of the NParticleOperator
        """
        ret = self._construct_empty()
        for block in self.blocks_nonzero:
            ret[block] = self.block(block).copy()
        ret.reference_state = self.reference_state
        return ret

    # return type: not sure about the number of tensors returned for general n-particle operators
    # -> go with the general variant
    def _transform_to_ao(self, refstate: "ReferenceState") -> tuple[libadcc.Tensor, ...]:
        raise NotImplementedError(
            f"Needs to be implemented on the {self.__class__.__name__} class."
        )

    def to_ao_basis(self, refstate: "ReferenceState | None" = None) -> tuple[libadcc.Tensor, ...]:
        """
        Transforms the NParticleOperator to the atomic orbital basis using ``refstate``.
        If ``refstate`` is not given, the ReferenceState used to construct the NParticleOperator
        is used instead.
        """
        from .ReferenceState import ReferenceState

        if refstate is None:
            if self.reference_state is None:
                raise ValueError(
                    "Argument 'refstate' is required if no reference state is stored in the "
                    "NParticleOperator."
                )
            refstate = self.reference_state
        if not isinstance(refstate, ReferenceState):
            raise TypeError(f"Invalid reference state of type {refstate.__class__.__name__}.")
        return self._transform_to_ao(refstate)

    # use Self once we drop 3.10
    def _add_inplace(self: _T, other: _T, factor: float) -> _T:  # noqa: PYI019
        """
        Computes ``self`` += ``factor`` * ``other``
        """
        # There is no __eq__ implementation -> check whether the same MoSpaces object is used.
        if self.mospaces is not other.mospaces:
            raise ValueError(f"Cannot add {self.__class__.__name__}s with differing mospaces.")
        if self.symmetry is not OperatorSymmetry.NOSYMMETRY and self.symmetry is not other.symmetry:
            raise ValueError(
                f"Cannot add operator with '{other.symmetry.to_str()}' symmetry to an "
                f"operator with '{self.symmetry.to_str()}' symmetry."
            )
        if self.blocks != other.blocks:
            raise ValueError(
                f"Cannot add operators with differing blocks, got\n{self.blocks}\n{other.blocks}"
            )

        if factor != 0:
            # we want to write to all canonical blocks in self
            for block in self.canonical_blocks:
                if other.is_zero_block(block):
                    continue
                # factor * ...  creates a new (lazy) tensor so self and other never share a
                # reference to the same tensor (after evaluation!)
                contrib = factor * other.block(block)
                if self.is_zero_block(block):
                    self[block] = contrib
                else:
                    self[block] += contrib

        # Update ReferenceState pointer (there is no __eq__ implementation)
        if self.reference_state is None:
            self.reference_state = other.reference_state
        elif (
            other.reference_state is not None and self.reference_state is not other.reference_state
        ):
            # both objects have a different reference state: which one should we choose?
            self.reference_state = None
        return self

    def __iadd__(self: _T, other: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        # explicitly compare the types to avoid weird non-symmetric behaviour, such as
        # NParticleOperator + OneParticleOperator -> works
        # OneParticleOperator + NParticleOperator -> raises
        if type(self) is not type(other):
            return NotImplemented
        return self._add_inplace(other, 1.0)

    def __isub__(self: _T, other: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        if type(self) is not type(other):
            return NotImplemented
        return self._add_inplace(other, -1.0)

    # use Self once we drop 3.10
    def __imul__(self: _T, other: float) -> _T:  # noqa: PYI019
        if not isinstance(other, (float, int)):
            return NotImplemented
        # multiply with a number in place
        for block in self.blocks_nonzero:
            self[block] = self.block(block) * other
        return self

    def __add__(self: _T, other: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        if not isinstance(other, NParticleOperator):  # other.symmetry has to work
            return NotImplemented
        # no/no, herm/herm, anti/anti, no/herm, no/anti
        if self.symmetry is other.symmetry or self.symmetry is OperatorSymmetry.NOSYMMETRY:
            return self.copy().__iadd__(other)
        # herm/no, anti/no
        elif other.symmetry is OperatorSymmetry.NOSYMMETRY:
            return other.copy().__iadd__(self)
        # herm/anti, anti/herm
        raise ValueError("Addition of hermitian and antihermitian operators is not implemented.")

    def __sub__(self: _T, other: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        if not isinstance(other, NParticleOperator):  # other.symmetry has to work
            return NotImplemented
        # no/no, herm/herm, anti/anti, no/herm, no/anti
        if self.symmetry is other.symmetry or self.symmetry is OperatorSymmetry.NOSYMMETRY:
            return self.copy().__isub__(other)
        # herm/no, anti/no
        elif other.symmetry is OperatorSymmetry.NOSYMMETRY:
            # -1 * other already creates a copy
            return (-1.0 * other).__iadd__(self)
        # herm/anti, anti/herm
        raise ValueError("Subtraction of hermitian and antihermitian operators is not implemented.")

    # use Self once we drop 3.10
    def __mul__(self: _T, other: float) -> _T:  # noqa: PYI019
        return self.copy().__imul__(other)

    # use Self once we drop 3.10
    def __rmul__(self: _T, other: float) -> _T:  # noqa: PYI019
        return self.copy().__imul__(other)

    def evaluate(self: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        """
        Ensure the operator is fully evaluated and allocated in memory.
        """
        for canonical_block in self._canonical_blocks.values():
            if canonical_block.tensor is not None:
                canonical_block.tensor.evaluate()
        return self

    def set_random(self: _T) -> _T:  # noqa: PYI019  # use Self once we drop 3.10
        """
        Fill all canonical blocks with random values.
        """
        for block in self.canonical_blocks:
            self[block].set_random()
        return self


def product_trace(op1: NParticleOperator, op2: NParticleOperator) -> float:
    """
    Compute the expectation value (inner product) of two N-particle operators.

    For the special case of a density operator and a general operator, the
    inner product is identical in the AO and MO basis.

    Parameters
    ----------
    op1, op2 : NParticleOperator
        Operators or densities to compute the inner product of.

    Returns
    -------
    float
        The trace / expectation value <op1, op2>.
    """
    if op1.n_particle_op != op2.n_particle_op:
        raise ValueError(
            "Both operators (or densities) must have the same dimensionality, got "
            f"a {op1.n_particle_op}-particle density/operator and a "
            f"{op2.n_particle_op}-particle density/operator."
        )
    elif op1.blocks != op2.blocks:
        raise ValueError(
            "Cannot compute the product trace of operators (or densities) with differing blocks, "
            f"got\n{op1.blocks}\n{op2.blocks}"
        )

    if op1.symmetry is OperatorSymmetry.NOSYMMETRY:
        factors = op1.canonical_factors
    elif op2.symmetry is OperatorSymmetry.NOSYMMETRY:
        factors = op2.canonical_factors
    else:
        assert op1.canonical_factors == op2.canonical_factors
        factors = op1.canonical_factors
        # Off-diagonal blocks can be removed since a Hermitian times an anti-Hermitian operator
        # yields for off diagonal terms (e.g. for a one-particle operator)
        # ov * ov + vo * vo = ov * ov + (ov).T * (−ov.T) = ov * ov - ov.T * ov.T = 0
        if op1.symmetry is not op2.symmetry:
            n = op1.n_particle_op
            factors = {
                block: factor
                for block, factor in factors.items()
                if (sp := b.split_spaces(block))[:n] == sp[n:]
            }
    ret: float = 0.0
    for block, factor in factors.items():
        if op1.is_zero_block(block) or op2.is_zero_block(block):
            continue

        ret += (
            1 / math.factorial(op1.n_particle_op) ** 2
            * factor
            * op1.block(block).dot(op2.block(block))
        )  # fmt: skip
    return ret
