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
from .NParticleDensity import NParticleDensity
from .NParticleOperator import OperatorSymmetry


class TwoParticleDensity(NParticleDensity):
    def __init__(self, spaces, symmetry=OperatorSymmetry.HERMITIAN):
        """
        Construct a TwoParticleDensity object. All blocks are initialised
        as zero blocks.

        Parameters
        ----------
        spaces : adcc.MoSpaces or adcc.ReferenceState or adcc.LazyMp
            MoSpaces object

        symmetry : OperatorSymmetry, optional
        """
        super().__init__(spaces, n_particle_op=2, symmetry=symmetry)

    def _construct_empty(self):
        """
        Create an empty instance of a TwoParticleDensity
        """
        return TwoParticleDensity(
            self.mospaces,
            symmetry=self.symmetry,
        )

    def _transform_to_ao(self, refstate_or_coefficients):
        """
        MO -> AO transformation of antisymmetrized densities/integrals is
        not implemented, because information about spin blocks is irreversibly lost
        after antisymmetrization. Example: Product Trace of a TwoParticleDensity
        with an ERI:
        aaaa block: + Gamma^pq_rs <pq||rs>
        bbbb block: + Gamma^pq_rs <pq||rs>
        abab block: + Gamma^pq_rs <pq|rs>
        abba block: - Gamma^pq_rs <pq|sr>
        """
        raise NotImplementedError(
            "MO -> AO transformation of antisymmetrized densities/integrals is "
            "not implemented, because information about spin blocks is "
            "irreversibly lost after antisymmetrization."
        )
