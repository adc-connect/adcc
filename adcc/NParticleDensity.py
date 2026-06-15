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
from .NParticleOperator import NParticleOperator, OperatorSymmetry


class NParticleDensity(NParticleOperator):
    def __init__(self, spaces, n_particle_op, symmetry=OperatorSymmetry.HERMITIAN):
        """
        General N-particle density operator.

        Density matrices require a different MO-to-AO transformation than general
        operators due to the non-orthonormal AO basis (C^\\dagger S C= 1).
        While general operators transform as

        d_MO = C^\\dagger S d_AO S C

        density matrices transform as

        D_MO = C^\\dagger d_AO C

        Expectation values (inner products) of a density and an operator are
        identical in the AO and MO basis. Note that it is important that only one
        of them is transformed using the overlap matrix!

        It therefore makes sense to consistently choose the transformations of the
        Fock matrix and the HF density matrix for all operators and densities,
        respectively.
        """
        super().__init__(spaces, n_particle_op, symmetry=symmetry)
