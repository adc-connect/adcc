#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2021 by the adcc authors
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
import adcc.block as b
from adcc.MoSpaces import split_spaces
from adcc.Tensor import Tensor
from adcc.functions import einsum


class TwoParticleDensityMatrix:
    """
    Two-particle density matrix (TPDM) used for gradient evaluations
    """
    def __init__(self, spaces):
        if hasattr(spaces, "mospaces"):
            self.mospaces = spaces.mospaces
        else:
            self.mospaces = spaces
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
        # set the canonical blocks explicitly
        self.blocks = [
            b.oooo, b.ooov, b.oovv,
            b.ovov, b.ovvv, b.vvvv,
        ]
        if self.mospaces.has_core_occupied_space:
            self.blocks += [
                b.cccc, b.ococ, b.cvcv,
                b.ocov, b.cccv, b.cocv, b.ocoo,
                b.ccco, b.occv, b.ccvv, b.ocvv,
            ]
        # make sure we didn't add any block twice!
        assert len(list(set(self.blocks))) == len(self.blocks)
        self._tensors = {}

    @property
    def shape(self):
        """
        Returns the shape tuple of the TwoParticleDensityMatrix
        """
        size = self.mospaces.n_orbs("f")
        return 4 * (size,)

    @property
    def size(self):
        """
        Returns the number of elements of the TwoParticleDensityMatrix
        """
        return np.prod(self.shape)

    def __setitem__(self, block, tensor):
        """
        Assigns a tensor to the specified block
        """
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} assigned. "
                           f"Available blocks are: {self.blocks}.")
        s1, s2, s3, s4 = split_spaces(block)
        expected_shape = (self.mospaces.n_orbs(s1),
                          self.mospaces.n_orbs(s2),
                          self.mospaces.n_orbs(s3),
                          self.mospaces.n_orbs(s4))
        if expected_shape != tensor.shape:
            raise ValueError("Invalid shape of incoming tensor. "
                             f"Expected shape {expected_shape}, but "
                             f"got shape {tensor.shape} instead.")
        self._tensors[block] = tensor

    def __getitem__(self, block):
        if block not in self.blocks:
            raise KeyError(f"Invalid block {block} requested. "
                           f"Available blocks are: {self.blocks}.")
        if block not in self._tensors:
            sym = libadcc.make_symmetry_eri(self.mospaces, block)
            self._tensors[block] = Tensor(sym)
        return self._tensors[block]

    def to_ndarray(self):
        raise NotImplementedError("ndarray export not implemented for TPDM.")

    def copy(self):
        """
        Return a deep copy of the TwoParticleDensityMatrix
        """
        ret = TwoParticleDensityMatrix(self.mospaces)
        for bl in self.blocks_nonzero:
            ret[bl] = self.block(bl).copy()
        if hasattr(self, "reference_state"):
            ret.reference_state = self.reference_state
        return ret

    @property
    def blocks_nonzero(self):
        """
        Returns a list of the non-zero block labels
        """
        return [b for b in self.blocks if b in self._tensors]

    def is_zero_block(self, block):
        """
        Checks if block is explicitly marked as zero block.
        Returns False if the block does not exist.
        """
        if block not in self.blocks:
            return False
        return block not in self.blocks_nonzero

    def block(self, block):
        """
        Returns tensor of the given block.
        Does not create a block in case it is marked as a zero block.
        Use __getitem__ for that purpose.
        """
        if block not in self.blocks_nonzero:
            raise KeyError("The block function does not support "
                           "access to zero-blocks. Available non-zero "
                           f"blocks are: {self.blocks_nonzero}.")
        return self._tensors[block]

    def __getattr__(self, attr):
        return self.__getitem__(b.__getattr__(attr))

    def __setattr__(self, attr, value):
        try:
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

    def __transform_to_ao(self, refstate_or_coefficients):
        if not len(self.blocks_nonzero):
            raise ValueError("At least one non-zero block is needed to "
                             "transform the TwoParticleDensityMatrix.")
        if isinstance(refstate_or_coefficients, libadcc.ReferenceState):
            hf = refstate_or_coefficients
            coeff_map = {}
            for sp in self.orbital_subspaces:
                coeff_map[sp + "_a"] = hf.orbital_coefficients_alpha(sp + "b")
                coeff_map[sp + "_b"] = hf.orbital_coefficients_beta(sp + "b")
        else:
            coeff_map = refstate_or_coefficients

        g2_ao_1 = 0
        g2_ao_2 = 0
        transf = "ip,jq,ijkl,kr,ls->pqrs"
        cc = coeff_map
        for block in self.blocks_nonzero:
            s1, s2, s3, s4 = split_spaces(block)
            ten = self[block]
            aaaa = einsum(transf, cc[f"{s1}_a"], cc[f"{s2}_a"],
                          ten, cc[f"{s3}_a"], cc[f"{s4}_a"])
            bbbb = einsum(transf, cc[f"{s1}_b"], cc[f"{s2}_b"],
                          ten, cc[f"{s3}_b"], cc[f"{s4}_b"])
            g2_ao_1 += (
                + aaaa
                + bbbb
                + einsum(transf, cc[f"{s1}_a"], cc[f"{s2}_b"],
                         ten, cc[f"{s3}_a"], cc[f"{s4}_b"])  # abab
                + einsum(transf, cc[f"{s1}_b"], cc[f"{s2}_a"],
                         ten, cc[f"{s3}_b"], cc[f"{s4}_a"])  # baba
            )
            g2_ao_2 += (
                + aaaa
                + bbbb
                + einsum(transf, cc[f"{s1}_a"], cc[f"{s2}_b"],
                         ten, cc[f"{s3}_b"], cc[f"{s4}_a"])  # abba
                + einsum(transf, cc[f"{s1}_b"], cc[f"{s2}_a"],
                         ten, cc[f"{s3}_a"], cc[f"{s4}_b"])  # baab
            )
        return (g2_ao_1.evaluate(), g2_ao_2.evaluate())

    def to_ao_basis(self, refstate_or_coefficients=None):
        """
        Transform the density to the AO basis for contraction
        with two-electron integrals.
        ALL coefficients are already accounted for in the density matrix.
        Two blocks are returned, the first one needs to be contracted with
        prqs, the second one with -psqr (in Chemists' notation).
        """
        if isinstance(refstate_or_coefficients, (dict, libadcc.ReferenceState)):
            return self.__transform_to_ao(refstate_or_coefficients)
        elif refstate_or_coefficients is None:
            if not hasattr(self, "reference_state"):
                raise ValueError("Argument reference_state is required if no "
                                 "reference_state is stored in the "
                                 "TwoParticleDensityMatrix")
            return self.__transform_to_ao(self.reference_state)
        else:
            raise TypeError("Argument type not supported.")

    def __iadd__(self, other):
        if self.mospaces != other.mospaces:
            raise ValueError("Cannot add TwoParticleDensityMatrices with "
                             "differing mospaces.")

        for bl in other.blocks_nonzero:
            if self.is_zero_block(bl):
                self[bl] = other.block(bl).copy()
            else:
                self[bl] = self.block(bl) + other.block(bl)

        # Update ReferenceState pointer
        if hasattr(self, "reference_state"):
            if hasattr(other, "reference_state") \
                    and self.reference_state != other.reference_state:
                delattr(self, "reference_state")
        return self

    def __isub__(self, other):
        if self.mospaces != other.mospaces:
            raise ValueError("Cannot subtract TwoParticleDensityMatrix with "
                             "differing mospaces.")

        for bl in other.blocks_nonzero:
            if self.is_zero_block(bl):
                self[bl] = -1.0 * other.block(bl)  # The copy is implicit
            else:
                self[bl] = self.block(bl) - other.block(bl)

        # Update ReferenceState pointer
        if hasattr(self, "reference_state"):
            if hasattr(other, "reference_state") \
                    and self.reference_state != other.reference_state:
                delattr(self, "reference_state")
        return self

    def __imul__(self, other):
        if not isinstance(other, (float, int)):
            return NotImplemented
        for bl in self.blocks_nonzero:
            self[bl] = self.block(bl) * other
        return self

    def __add__(self, other):
        return self.copy().__iadd__(other)

    def __sub__(self, other):
        return self.copy().__isub__(other)

    def __mul__(self, other):
        return self.copy().__imul__(other)

    def __rmul__(self, other):
        return self.copy().__imul__(other)

    def evaluate(self):
        for bl in self.blocks_nonzero:
            self.block(bl).evaluate()
        return self
