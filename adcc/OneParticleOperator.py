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
from .functions import evaluate

import libadcc


class OneParticleOperator(libadcc.OneParticleOperator):
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
        if isinstance(spaces, libadcc.ReferenceState):
            super().__init__(spaces.mospaces, is_symmetric, "1")
            self.reference_state = spaces
        elif isinstance(spaces, libadcc.LazyMp):
            super().__init__(spaces.mospaces, is_symmetric, "1")
            self.reference_state = spaces.reference_state
        else:
            super().__init__(spaces, is_symmetric, "1")

    @classmethod
    def from_cpp(cls, cpp_operator):
        assert cpp_operator.cartesian_transform == "1"
        ret = cls(cpp_operator.mospaces, cpp_operator.is_symmetric)
        for b in cpp_operator.blocks_nonzero:
            ret.set_block(b, cpp_operator.block(b))
        return ret

    def copy(self):
        """
        Return a deep copy of the OneParticleOperator
        """
        ret = OneParticleOperator.from_cpp(super().copy())
        if hasattr(self, "reference_state"):
            ret.reference_state = self.reference_state
        return ret

    def to_ao_basis(self, refstate_or_coefficients=None):
        """
        TODO DOCME
        """
        if isinstance(refstate_or_coefficients, (dict, libadcc.ReferenceState)):
            return super().to_ao_basis(refstate_or_coefficients)
        elif refstate_or_coefficients is None:
            if not hasattr(self, "reference_state"):
                raise ValueError("Argument reference_state is required if no "
                                 "reference_state is stored in the "
                                 "OneParticleOperator")
            return super().to_ao_basis(self.reference_state)
        else:
            raise TypeError("Argument type not supported.")

    def __iadd__(self, other):
        if not isinstance(other, libadcc.OneParticleOperator):
            return NotImplemented
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
        if not isinstance(other, libadcc.OneParticleOperator):
            return NotImplemented
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
