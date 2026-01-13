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
import string

import libadcc

from .functions import einsum
from .MoSpaces import split_spaces
from .NParticleOperator import NParticleOperator, OperatorSymmetry
from .Tensor import Tensor


class NParticleDensity(NParticleOperator):
    def __init__(self, spaces, n_particle_op, symmetry=OperatorSymmetry.HERMITIAN):
        """
        N-particle density operator.

        Density matrices require a different MO-to-AO transformation than general
        operators due to the non-orthonormal AO basis (C^\\dagger S C= 1).
        While general operators transform as

        d_MO = C^\\dagger S d_AO S C

        density matrices transform as

        D_MO = C^\\dagger d_AO C

        Expectation values (inner products) of a density and an operator are
        identical in the AO and MO basis. Note that it is important that only one
        of them is transformed using the overlap matrix.

        It therefore makes sense to consistently choose the transformations of the
        Fock matrix and the HF density matrix for all operators and densities,
        respectively.
        """
        super().__init__(spaces, n_particle_op, symmetry=symmetry)

    def _transform_to_ao(self, refstate_or_coefficients) -> tuple[Tensor, Tensor]:
        if not len(self.blocks_nonzero):
            raise ValueError("At least one non-zero block is needed to "
                             "transform the TwoParticleDensity.")
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
            # skip blocks that add up to zero
            factor = self.canonical_factors[block]

            dm_bb_a += factor * self._construct_einsum(
                tensor_mo=self.block(block),
                coeff_map=coeff_map,
                bra_spaces=spaces[:self.n_particle_op],
                ket_spaces=spaces[self.n_particle_op:],
                spin="a",
            )
            dm_bb_b += factor * self._construct_einsum(
                tensor_mo=self.block(block),
                coeff_map=coeff_map,
                bra_spaces=spaces[:self.n_particle_op],
                ket_spaces=spaces[self.n_particle_op:],
                spin="b",
            )

        if self.n_particle_op == 1:
            if self.symmetry == OperatorSymmetry.HERMITIAN:
                dm_bb_a = dm_bb_a.symmetrise()
                dm_bb_b = dm_bb_b.symmetrise()
            elif self.symmetry == OperatorSymmetry.ANTIHERMITIAN:
                dm_bb_a = dm_bb_a.antisymmetrise()
                dm_bb_b = dm_bb_b.antisymmetrise()
        else:
            raise NotImplementedError
        return (dm_bb_a.evaluate(), dm_bb_b.evaluate())

    def _construct_einsum(self, tensor_mo, coeff_map, bra_spaces,
                          ket_spaces, spin="a") -> Tensor:
        rank = 2 * self.n_particle_op

        letters = string.ascii_lowercase
        # ensure that there are enough letters
        assert rank * 2 <= len(letters)

        mo_indices = letters[:rank]
        ao_indices = letters[rank:2 * rank]

        einsum_inputs = []
        einsum_indices = []

        for i, sp in enumerate(bra_spaces):
            einsum_inputs.append(coeff_map[f"{sp}_{spin}"])
            einsum_indices.append(f"{mo_indices[i]}{ao_indices[i]}")

        einsum_inputs.append(tensor_mo)
        einsum_indices.append(mo_indices)

        for i, sp in enumerate(ket_spaces):
            einsum_inputs.append(coeff_map[f"{sp}_{spin}"])
            einsum_indices.append(f"{mo_indices[i + int(rank / 2)]}"
                                  f"{ao_indices[i + int(rank / 2)]}")

        einsum_str = ",".join(einsum_indices) + "->" + "".join(ao_indices)
        return einsum(einsum_str, *einsum_inputs)
