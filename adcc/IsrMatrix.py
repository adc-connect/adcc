#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2022 by the adcc authors
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
import libadcc

from .AdcMatrix import AdcMatrixlike
from .AdcMethod import AdcMethod
from .adc_pp import bmatrix as ppbmatrix
from .AmplitudeVector import AmplitudeVector
from .LazyMp import LazyMp
from .OneParticleOperator import OneParticleOperator
from .timings import Timer, timed_member_call


class IsrMatrix(AdcMatrixlike):

    def __init__(self, method, hf_or_mp, operator, block_orders=None):
        """
        Initialise an ISR matrix of a given one-particle operator
        for the provided ADC method.

        Parameters
        ----------
        method : str or adcc.AdcMethod
            Method to use.
        hf_or_mp : adcc.ReferenceState or adcc.LazyMp
            HF reference or MP ground state.
        operator : adcc.OneParticleOperator or list of adcc.OneParticleOperator
                    objects
            One-particle matrix elements associated with a one-particle operator.
        block_orders : optional
            The order of perturbation theory to employ for each matrix block.
            If not set, defaults according to the selected ADC method are chosen.
        """
        if isinstance(hf_or_mp, (libadcc.ReferenceState,
                                 libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)
        if not isinstance(hf_or_mp, LazyMp):
            raise TypeError("hf_or_mp is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)

        if isinstance(operator, (list, tuple)):
            self.operator = tuple(operator)
        else:
            self.operator = (operator,)
        if not all(isinstance(op, OneParticleOperator) for op in self.operator):
            raise TypeError("operator is not a valid object. It needs to be "
                            "either an OneParticleOperator or a list of "
                            "OneParticleOperator objects.")

        self.timer = Timer()
        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = method.is_core_valence_separated
        self.ndim = 2
        self.extra_terms = []

        self.block_orders = self._default_block_orders(self.method)
        if block_orders is None:
            # only implemented through PP-ADC(2)
            if method.adc_type != "pp" or method.name.endswith("adc2x") \
                    or method.level > 2:
                raise NotImplementedError("The B-matrix is not implemented "
                                          f"for method {method.name}.")
        else:
            self.block_orders.update(block_orders)
        self._validate_block_orders(
            block_orders=self.block_orders, method=self.method,
            allow_missing_diagonal_blocks=True
        )

        # Build the blocks
        with self.timer.record("build"):
            variant = None
            if self.is_core_valence_separated:
                variant = "cvs"
            blocks = [{
                block: ppbmatrix.block(self.ground_state, op,
                                       block.split("_"), order=order,
                                       variant=variant)
                for block, order in self.block_orders.items() if order is not None
            } for op in self.operator]
            self.blocks = [{
                b: bl[b].apply for b in bl
            } for bl in blocks]

    @timed_member_call()
    def matvec(self, v):
        """
        Compute the matrix-vector product of the ISR one-particle
        operator and return the result.

        If a list of OneParticleOperator objects was passed to the class
        instantiation operator, a list of AmplitudeVector objects is returned.
        """
        ret = [
            sum(block(v) for block in bl_ph.values())
            for bl_ph in self.blocks
        ]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def rmatvec(self, v):
        # Hermitian operators
        if all(op.is_symmetric for op in self.operator):
            return self.matvec(v)
        else:
            diffv = [op.ov + op.vo.transpose((1, 0)) for op in self.operator]
            # anti-Hermitian operators
            if all(dv.dot(dv) < 1e-12 for dv in diffv):
                return [
                    AmplitudeVector(ph=-1.0 * mv.ph, pphh=-1.0 * mv.pphh)
                    for mv in self.matvec(v)
                ]
            # operators without any symmetry
            else:
                return NotImplemented

    def __matmul__(self, other):
        """
        If the matrix-vector product is to be calculated with multiple vectors,
        a list of AmplitudeVector objects is returned.

        Consequently, applying the ISR matrix with multiple operators to multiple
        vectors results in an N_vectors x N_operators 2D list of AmplitudeVector
        objects.
        """
        if isinstance(other, AmplitudeVector):
            return self.matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return [self.matvec(ov) for ov in other]
        return NotImplemented
