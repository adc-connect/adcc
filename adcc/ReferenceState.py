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

from .misc import cached_property
from .Tensor import Tensor
from .MoSpaces import MoSpaces
from .backends import import_scf_results
from .OperatorIntegrals import OperatorIntegrals
from .OneParticleOperator import OneParticleOperator, product_trace

import libadcc

__all__ = ["ReferenceState"]


class ReferenceState(libadcc.ReferenceState):
    def __init__(self, hfdata, core_orbitals=None, frozen_core=None,
                 frozen_virtual=None, symmetry_check_on_import=False,
                 import_all_below_n_orbs=10):
        """Construct a ReferenceState holding information about the employed
        SCF reference.

        The constructed object is lazy and will at construction only setup
        orbital energies and coefficients. Fock matrix blocks and
        electron-repulsion integral blocks are imported as needed.

        Orbital subspace selection: In order to specify `frozen_core`,
        `core_orbitals` and `frozen_virtual`, adcc allows a range of
        specifications including

           a. A number: Just put this number of alpha orbitals and this
              number of beta orbitals into the respective space. For frozen
              core and core orbitals these are counted from below, for
              frozen virtual orbitals, these are counted from above. If both
              frozen core and core orbitals are specified like this, the
              lowest-energy, occupied orbitals will be put into frozen core.
           b. A range: The orbital indices given by this range will be put
              into the orbital subspace.
           c. An explicit list of orbital indices to be placed into the
              subspace.
           d. A pair of (a) to (c): If the orbital selection for alpha and
              beta orbitals should differ, a pair of ranges, or a pair of
              index lists or a pair of numbers can be specified.

        Parameters
        ----------
        hfdata
            Object with Hartree-Fock data (e.g. a molsturm scf state, a pyscf
            SCF object or any class implementing the
            :py:class:`adcc.HartreeFockProvider` interface or in fact any python
            object representing a pointer to a C++ object derived off
            the :cpp:class:`adcc::HartreeFockSolution_i`.

        core_orbitals : int or list or tuple, optional
            The orbitals to be put into the core-occupied space. For ways to
            define the core orbitals see the description above.

        frozen_core : int or list or tuple, optional
            The orbitals to be put into the frozen core space. For ways to
            define the core orbitals see the description above. For an automatic
            selection of the frozen core space one may also specify
            ``frozen_core=True``.

        frozen_virtuals : int or list or tuple, optional
            The orbitals to be put into the frozen virtual space. For ways to
            define the core orbitals see the description above.

        symmetry_check_on_import : bool, optional
            Should symmetry of the imported objects be checked explicitly during
            the import process. This massively slows down the import and has a
            dramatic impact on memory usage. Thus one should enable this only
            for debugging (e.g. for testing import routines from the host
            programs). Do not enable this unless you know what you are doing.

        import_all_below_n_orbs : int, optional
            For small problem sizes lazy make less sense, since the memory
            requirement for storing the ERI tensor is neglibile and thus the
            flexiblity gained by having the full tensor in memory is
            advantageous. Below the number of orbitals specified by this
            parameter, the class will thus automatically import all ERI tensor
            and Fock matrix blocks.

        Examples
        --------
        To start a calculation with the 2 lowest alpha and beta orbitals
        in the core occupied space, construct the class as

        >>> ReferenceState(hfdata, core_orbitals=2)

        or

        >>> ReferenceState(hfdata, core_orbitals=range(2))

        or

        >>> ReferenceState(hfdata, core_orbitals=[0, 1])

        or

        >>> ReferenceState(hfdata, core_orbitals=([0, 1], [0, 1]))

        There is no restriction to choose the core occupied orbitals
        from the bottom end of the occupied orbitals. For example
        to select the 2nd and 3rd orbital setup the class as

        >>> ReferenceState(hfdata, core_orbitals=range(1, 3))

        or

        >>> ReferenceState(hfdata, core_orbitals=[1, 2])

        If different orbitals should be placed in the alpha and
        beta orbitals, this can be achievd like so

        >>> ReferenceState(hfdata, core_orbitals=([1, 2], [0, 1]))

        which would place the 2nd and 3rd alpha and the 1st and second
        beta orbital into the core space.
        """
        if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
            hfdata = import_scf_results(hfdata)

        self._mospaces = MoSpaces(hfdata, frozen_core=frozen_core,
                                  frozen_virtual=frozen_virtual,
                                  core_orbitals=core_orbitals)
        super().__init__(hfdata, self._mospaces, symmetry_check_on_import)

        if import_all_below_n_orbs is not None and \
           hfdata.n_orbs < import_all_below_n_orbs:
            super().import_all()

        self.operators = OperatorIntegrals(
            hfdata.operator_integral_provider, self._mospaces,
            self.orbital_coefficients, self.conv_tol
        )

        self.environment = None  # no environment attached by default
        for name in ["excitation_energy_corrections", "environment"]:
            if hasattr(hfdata, name):
                setattr(self, name, getattr(hfdata, name))

    def __getattr__(self, attr):
        from . import block as b

        if attr.startswith("f"):
            return self.fock(b.__getattr__(attr[1:]))
        else:
            return self.eri(b.__getattr__(attr))

    @property
    def mospaces(self):
        return self._mospaces

    @property
    def timer(self):
        ret = super().timer
        ret.attach(self.operators.timer)
        return ret

    @property
    def is_aufbau_occupation(self):
        """
        Returns whether the molecular orbital occupation in this reference
        is according to the Aufbau principle (lowest-energy orbitals are occupied)
        """
        eHOMO = max(max(self.orbital_energies(space).to_ndarray())
                    for space in self.mospaces.subspaces_occupied)
        eLUMO = min(min(self.orbital_energies(space).to_ndarray())
                    for space in self.mospaces.subspaces_virtual)
        return eHOMO < eLUMO

    def to_qcvars(self, properties=False, recurse=False):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        qcvars = {"HF TOTAL ENERGY": self.energy_scf, }
        if properties:
            qcvars["HF DIPOLE"] = self.dipole_moment
        return qcvars

    @property
    def density(self):
        """
        Return the Hartree-Fock density in the MO basis
        """
        density = OneParticleOperator(self.mospaces, is_symmetric=True)
        for block in density.blocks:
            sym = libadcc.make_symmetry_operator(self.mospaces, block, True, "1")
            density[block] = Tensor(sym)
        for ss in self.mospaces.subspaces_occupied:
            density[ss + ss].set_mask("ii", 1)
        density.reference_state = self
        return density

    @cached_property
    def dipole_moment(self):
        """
        Return the HF dipole moment of the reference state (that is the sum of
        the electronic and the nuclear contribution.)
        """
        dipole_integrals = self.operators.electric_dipole
        # Notice the negative sign due to the negative charge of the electrons
        return self.nuclear_dipole - np.array([product_trace(comp, self.density)
                                               for comp in dipole_integrals])

# TODO some nice describe method
