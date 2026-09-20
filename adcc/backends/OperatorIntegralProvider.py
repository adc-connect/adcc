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
from abc import ABC, abstractmethod

import libadcc

from ..typing import Array2D, Coordinate, DipoleLikeArray, QuadrupoleLikeArray


class OperatorIntegralProvider(ABC):
    """
    Base class that defines the interface for accessing operator integrals
    from the different backends. Integrals are generally imported in the AO
    basis as numpy arrays.
    """

    @property
    def available(self) -> tuple[str, ...]:
        """Lists integrals that are available in the backend."""
        # check the methods avilable on the child class (resolved along the MRO)
        # and return all whose definition differs from the one on this class.
        # This implementation does not work for classmethods or staticmethods
        blacklist = ("backend", "available")
        return tuple(
            name
            for name in dir(self.__class__)
            if not name.startswith("_")
            and name not in blacklist
            and hasattr(OperatorIntegralProvider, name)
            and getattr(self.__class__, name) is not getattr(OperatorIntegralProvider, name)
        )

    @property
    @abstractmethod
    def backend(self) -> str:
        """Name of the backend providing the integrals."""

    @property
    def overlap(self) -> Array2D:
        """The AO overlap matrix"""
        raise NotImplementedError(
            f"Overlap operator not implemented for the {self.backend} backend"
        )

    @property
    def electric_dipole(self) -> DipoleLikeArray:
        """
        The electric dipole operator
        -sum_i r_i
        """
        raise NotImplementedError(
            f"Electric dipole operator not implemented for the {self.backend} backend"
        )

    @property
    def electric_dipole_velocity(self) -> DipoleLikeArray:
        """
        The imaginary part of the integral is returned.
        -sum_i p_i
        """
        raise NotImplementedError(
            "Electric dipole operator in the velocity gauge not implemented for the "
            f"{self.backend} backend"
        )

    def magnetic_dipole(self, gauge_origin: Coordinate | str = "origin") -> DipoleLikeArray:
        """
        The imaginary part of the integral is returned.
        -0.5 * sum_i r_i x p_i
        """
        raise NotImplementedError(
            f"Magnetic dipole operator not implemented for the {self.backend} backend"
        )

    def electric_quadrupole(self, gauge_origin: Coordinate | str = "origin") -> QuadrupoleLikeArray:
        """
        The electric quadrupole operator
        -sum_i r_{i, alpha} r_{i, beta}
        """
        raise NotImplementedError(
            f"Electric quadrupole operator not implemented for the {self.backend} backend"
        )

    def electric_quadrupole_traceless(
        self, gauge_origin: Coordinate | str = "origin"
    ) -> QuadrupoleLikeArray:
        """
        -0.5 * sum_i (3 * r_{i, alpha} r_{i, beta}
        - delta_{alpha, beta} r_{i}^2)
        """
        raise NotImplementedError(
            f"Traceless electric quadrupole operator not implemented for the {self.backend} backend"
        )

    def electric_quadrupole_velocity(
        self, gauge_origin: Coordinate | str = "origin"
    ) -> QuadrupoleLikeArray:
        """
        The imaginary part of the integral is returned.
        -sum_i (r_{i, beta} p_{i, alpha} - i delta_{alpha, beta}
        + r_{i, alpha} p_{i, beta})
        """
        raise NotImplementedError(
            "Electric quadrupole operator in the velocity gauge not implemented for "
            f"the {self.backend} backend"
        )

    def diamagnetic_magnetizability(
        self, gauge_origin: Coordinate | str = "origin"
    ) -> QuadrupoleLikeArray:
        """
        0.25 * sum_i (r_{i, alpha} r_{i, beta}
        - delta_{alpha, beta} r_{i}^2)
        """
        raise NotImplementedError(
            f"Diagmagnetic magnetizability operator not implemented for the {self.backend} backend"
        )

    def pe_induction_elec(self, dm: libadcc.Tensor) -> Array2D:
        raise NotImplementedError(
            f"Polarizable embedding induction operator not implemented for "
            f"the {self.backend} backend"
        )

    def pcm_potential_elec(self, dm: libadcc.Tensor) -> Array2D:
        raise NotImplementedError(
            f"Polarizable continuum potential operator not implemented for "
            f"the {self.backend} backend"
        )
