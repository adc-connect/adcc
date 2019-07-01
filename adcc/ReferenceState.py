#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import numpy as np

from .misc import cached_property
from .Tensor import Tensor
from .backends import import_scf_results
from .memory_pool import memory_pool
from .OperatorIntegrals import OperatorIntegrals
from .OneParticleOperator import OneParticleOperator, product_trace

from collections.abc import Iterable

import libadcc

__all__ = ["ReferenceState"]


# TODO Documentation
# TODO The arbitrary space stuff is not exactly convenient at the moment,
#      since the + n_mo stuff to get to the beta orbitals depends
#      on the basis used and is not intuitive.
#      One should have a wrapper around this here, which makes it more
#      convenient to use.
#
#      Allow to pass a list, which is interpreted as both
#      alpha and beta indices (added + n_orbs_alpha automatically)
#      or two lists one for alpha, one for beta
#
#      e.g. core_orbitals=[0,1] would be equivalent to core_orbitals=2
#           and core_orbitals=([0,1], [0,1]) and core_orbitals=range(2)

class ReferenceState(libadcc.ReferenceState):
    def __init__(self, hfdata, core_orbitals=None, frozen_core=None,
                 frozen_virtual=None, symmetry_check_on_import=False,
                 import_all_below_n_orbs=10):
        """
        Construct a ReferenceState object. The object is lazy and will only
        import orbital energies and coefficients. Fock matrix blocks and
        electron-repulsion integral blocks are imported as needed.

        @param hfdata
        Object with Hartree-Fock data (e.g. a molsturm scf state, a pyscf SCF
        object or any class implementing the adcc.HartreeFockProvider interface
        or in fact any python object representing a pointer to a C++ object
        derived off the adcc::HartreeFockSolution_i.

        @param core_orbitals
        (a) The number of alpha and beta core orbitals to use. The first
        orbitals (in the original ordering of the hfdata object), which are not
        part of the frozen_core will be selected.
        (b) Explicit list of orbital indices (in the ordering of the hfdata
        object) to put into the core-occupied orbital space. The same number of
        alpha and beta orbitals have to be selected. These will be forcibly
        occupied.

        @param frozen_core
        (a) The number of alpha and beta frozen core orbitals to use. The first
        orbitals (in the original ordering of the hfdata object) will be
        selected.
        (b) Explicit list of orbital indices (in the ordering of the hfdata
        object) to put into the fropen core. The same number of alpha and beta
        orbitals have to be selected. These will be forcibly occupied.

        @param frozen_virtuals
        (a) The number of alpha and beta frozen virtual orbitals to use. The
        last orbitals will be selected.
        (b) Explicit list of orbital indices to put into the frozen virtual
        orbital subspace. The same number of alpha and beta orbitals have to be
        selected. These will be forcibly unoccupied.

        @param symmetry_check_on_import
        Should symmetry of the imported objects be checked explicitly during the
        import process. This massively slows down the import and has a dramatic
        impact on memory usage. Thus one should enable this only for debugging
        (e.g. for testing import routines from the host programs). Do not enable
        this unless you know what you are doing.

        @import_all_below_n_orbs
        For small problem sizes lazy make less sense, since the memory
        requirement for storing the ERI tensor is neglibile and thus the
        flexiblity gained by having the full tensor in memory is advantageous.
        Below the number of orbitals specified by this parameter, the class
        will thus automatically import all ERI tensor and fock matrix blocks.
        """
        if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
            hfdata = import_scf_results(hfdata)

        if isinstance(frozen_core, bool) and frozen_core:
            # Determine number of frozen core electrons automatically
            # TODO The idea is to look at the energy gap in the HF orbital
            #      energies and exclude the ones, which are very far from the
            #      HOMO-LUMO gap.
            raise NotImplementedError("Automatic determination of frozen-core "
                                      "electrons not implemented.")

        #
        # Normalise and deal with space arguments
        #
        spaceargs = {"frozen_core": frozen_core, "core_orbitals": core_orbitals,
                     "frozen_virtual": frozen_virtual}

        def expand_to_list(space, entry, from_min=None, from_max=None):
            if from_min is None and from_max is None:
                raise ValueError("Both from_min and from_max is None")
            if entry is None:
                return np.array([])
            elif isinstance(entry, int):
                if from_min is not None:
                    return from_min + np.arange(entry)
                return np.arange(from_max - entry, from_max)
            elif isinstance(entry, Iterable):
                return np.array(entry)
            else:
                raise TypeError("Unsupported type {} passed to argument {}"
                                "".format(type(entry), space))

        for key in spaceargs:
            if not isinstance(spaceargs[key], tuple):
                spaceargs[key] = (spaceargs[key], spaceargs[key])

        any_iterable = False
        for key in spaceargs:
            for spin in [0, 1]:
                if isinstance(spaceargs[key][spin], Iterable):
                    any_iterable = True
                elif spaceargs[key][spin] is not None:
                    if any_iterable:
                        raise ValueError("If one of the values of frozen_core, "
                                         "core_orbitals, frozen_virtual is an "
                                         "iterable, all must be.")

        n_orbs = [0, 0]
        for key in ["frozen_core", "core_orbitals"]:
            noa = hfdata.n_orbs_alpha
            list_alpha = expand_to_list(key, spaceargs[key][0],
                                        from_min=n_orbs[0])
            list_beta = noa + expand_to_list(key, spaceargs[key][1],
                                             from_min=n_orbs[1])
            spaceargs[key] = np.concatenate((list_alpha, list_beta)).tolist()
            n_orbs[0] += len(list_alpha)
            n_orbs[1] += len(list_beta)

        key = "frozen_virtual"
        spaceargs[key] = np.concatenate((
            expand_to_list(key, spaceargs[key][0],
                           from_max=hfdata.n_orbs_alpha),
            noa + expand_to_list(key, spaceargs[key][1],
                                 from_max=hfdata.n_orbs_beta)
        )).tolist()

        super().__init__(hfdata, memory_pool, spaceargs["core_orbitals"],
                         spaceargs["frozen_core"], spaceargs["frozen_virtual"],
                         symmetry_check_on_import)

        if import_all_below_n_orbs is not None and \
           hfdata.n_orbs < import_all_below_n_orbs:
            super().import_all()

        self.operators = OperatorIntegrals(
            hfdata.operator_integral_provider, self.mospaces,
            self.orbital_coefficients, self.conv_tol
        )

    @property
    def timer(self):
        ret = super().timer
        ret.attach(self.operators.timer)
        return ret

    @property
    def density(self):
        """
        Return the Hartree-Fock density in the MO basis
        """
        density = OneParticleOperator(self.mospaces, is_symmetric=True)
        for b in density.blocks:
            sym = libadcc.make_symmetry_operator(self.mospaces, b, True, "1")
            density.set_block(b, Tensor(sym))
        for ss in self.mospaces.subspaces_occupied:
            density[ss + ss].set_mask("ii", 1)
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
