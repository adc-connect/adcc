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
import warnings
import itertools

import numpy as np

from .guess import guess_kwargs_kind, guess_symmetries
from .Tensor import Tensor
from .MoSpaces import expand_spaceargs
from .Symmetry import Symmetry
from .AdcMatrix import AdcMatrixlike
from .AmplitudeVector import AmplitudeVector

__all__ = ["SubspacePartitioning", "Projector"]


class SubspacePartitioning:
    def __init__(self, mospaces, core_orbitals=None, outer_virtuals=None,
                 partitions=None, aliases=None):
        """
        Partition the "o1" and "v1" occupied and virtual orbital subspace into two
        (or more) parts. In combination with the :py:`adcc.Projector`
        class this allows to disable some excitations in an amplitude vector.

        Parameters
        ----------
        mospaces : MoSpaces
            MoSpaces to base partitioning on.
        core_orbitals : int or list or tuple, optional
            The orbitals to be put into the ``c`` core space.
            For details on the option how to select these spaces, see the
            documentation in :py:`adcc.ReferenceState.__init__` (``outer_virtuals``
            follows the same rules as ``frozen_virtuals``).
        outer_virtuals : int or list or tuple, optional
            The virtuals to be put into the ``w`` outer-virtual space.
        partitions : dict, optional
            Partitioning of the spaces: Mapping from the partitioned spaces to
            the indices of the orbitals which belong to this partition. The
            indices are given relative to the corresponding full orbital subspace
            (advanced argument).
        aliases : dict, optional
            Aliase mapping to give partitions a more user-friendly identifier
            (advanced argument).
        """
        if mospaces.has_core_occupied_space or "v2" in mospaces.subspaces_virtual:
            raise ValueError("Cannot use MoSpaces with core-valence separation "
                             "or frozen virtuals.")
        assert mospaces.subspaces_occupied == ["o1"]

        if not partitions or not aliases:
            if core_orbitals is None and outer_virtuals is None:
                raise ValueError("One of core_orbitals or outer_virtuals should "
                                 "be passed.")

            # core / outer virtual part of the "normal" occupied / virtual subspace
            noa = mospaces.n_orbs_alpha("o1")
            nob = mospaces.n_orbs_beta("o1")
            nva = mospaces.n_orbs_alpha("v1")
            nvb = mospaces.n_orbs_beta("v1")
            core = expand_spaceargs((noa, nob), core_orbitals=core_orbitals)
            core = core["core_orbitals"]
            fv = expand_spaceargs((nva, nvb), frozen_virtual=outer_virtuals)
            fv = fv["frozen_virtual"]

            # The active core and virtual parts:
            occ = [i for i in range(mospaces.n_orbs("o1")) if i not in core]
            av = [i for i in range(mospaces.n_orbs("v1")) if i not in fv]
            assert occ and av

            # Setup the space partitionings and aliases
            partitions = {"o1.1": occ, "v1.1": av, }
            aliases = {"o": "o1.1", "v": "v1.1", }
            if core:
                partitions["o1.2"] = core
                aliases["c"] = "o1.2"
            if fv:
                partitions["v1.2"] = fv
                aliases["w"] = "v1.2"
            self.__init__(mospaces, partitions=partitions, aliases=aliases)
        else:
            assert core_orbitals is None
            assert outer_virtuals is None

            self.partitions = partitions
            self.aliases = aliases
            self.mospaces = mospaces

            # Determine whether the partitioning keeps spin symmetry.
            self.keeps_spin_symmetry = mospaces.restricted
            for (label, indices) in partitions.items():
                if not self.keeps_spin_symmetry:
                    break
                nalpha = mospaces.n_orbs_alpha(label[:2])
                alpha_part = [i for i in indices if i < nalpha]
                beta_part = [i - nalpha for i in indices if i >= nalpha]
                self.keeps_spin_symmetry = alpha_part == beta_part

            if mospaces.restricted and not self.keeps_spin_symmetry:
                warnings.warn(
                    "For restricted references the case of a space "
                    "partitioning, which breaks spin (i.e. where excitations "
                    "differing only in spin are placed in different "
                    "partitions) has not been tested. You are on your own."
                )

    def list_space_partitions(self, space):
        """
        Return the partitions into which the passed space is split.
        Resolves aliases, i.e. does not return not the list of strings
        such as ['o1.1', 'o1.2], but the aliased equivalent ["o", "c"].
        """
        partitions = [s for s in self.partitions if s.startswith(space + ".")]
        partitions = [key for (key, value) in self.aliases.items()
                      if value in partitions]
        return sorted(partitions)

    def get_partition(self, label):
        """Return the index list corresponding to a partition label"""
        return self.partitions[self.aliases[label]]


class Projector:
    def __init__(self, subspaces, partitioning, blocks_to_keep):
        """
        Initialise a projector, which upon multiplication with a tensor sets
        a number of partitioning blocks of this tensor to zero. These are defined
        by the passed `partitioning` and the list of `blocks_to_keep`.

        Parameters
        ----------
        subspaces : list
            List of subspaces to act upon (e.g. ``["o1", "v1", "v1"]``)
        partitioning : SubspacePartitioning
            Partitioning of the ``o1`` occupied and ``v1`` virtual space.
        blocks_to_keep : list
            Blocks which are kept as non-zero (e.g. ``["cv", "ccvv", "ocvv"]``).
        """
        if len(subspaces) == 3 or len(subspaces) > 4 \
                or (len(subspaces) > 2 and subspaces[1] == subspaces[2]):
            raise NotImplementedError(
                f"Projector not implemented for subspaces {subspaces}"
            )

        def normalise_block(labels):
            labels = list(labels)
            if len(subspaces) <= 1:
                return labels
            if subspaces[0] == subspaces[1] and labels[0] > labels[1]:
                labels[0], labels[1] = labels[1], labels[0]
            if len(subspaces) <= 2:
                return labels
            if len(subspaces) == 3 or subspaces[1] == subspaces[2] \
                    or len(subspaces) > 4:
                raise NotImplementedError("not implemented")
            if subspaces[2] == subspaces[3] and labels[2] > labels[3]:
                labels[2], labels[3] = labels[3], labels[2]
            return labels

        blocks_to_keep = ["".join(normalise_block(block))
                          for block in blocks_to_keep]
        partitions = [partitioning.list_space_partitions(space)
                      for space in subspaces]
        all_partition_blocks = ["".join(normalise_block(block))
                                for block in itertools.product(*partitions)]
        for block in blocks_to_keep:
            if block not in all_partition_blocks:
                raise ValueError(f"Partition block {block} not known.")

        sym = Symmetry(partitioning.mospaces, "".join(subspaces))
        sym.irreps_allowed = ["A"]
        if partitioning.keeps_spin_symmetry:
            if len(subspaces) == 1:
                sym.spin_block_maps = [("a", "b", 1)]
            elif len(subspaces) == 2:
                sym.spin_block_maps = [("aa", "bb", 1), ("ab", "ba", 1)]
            elif len(subspaces) == 4:
                sym.spin_block_maps = [
                    ("aaaa", "bbbb", 1), ("aaab", "bbba", 1),
                    ("aaba", "bbab", 1), ("abaa", "babb", 1),
                    ("baaa", "abbb", 1), ("aabb", "bbaa", 1),
                    ("abba", "baab", 1), ("abab", "baba", 1)
                ]
            else:
                raise NotImplementedError("not implemented")
        if len(subspaces) == 4:
            assert subspaces[0] == subspaces[1]
            assert subspaces[2] == subspaces[3]
            sym.permutations = ["ijkl", "jikl", "ijlk"]

        # Allocate zero tensor to hold projection kernel:
        kernel = Tensor(sym)

        # Generic implementation not using symmetry
        for block in blocks_to_keep:
            ranges = [partitioning.get_partition(b) for b in block]
            for index in itertools.product(*ranges):
                kernel[index] = 1.0

        self.subspaces = subspaces
        self.partitioning = partitioning
        self.blocks_to_keep = blocks_to_keep
        self.kernel = kernel

    def matvec(self, v):
        return self.kernel * v

    def __matmul__(self, other):
        if isinstance(other, libadcc.Tensor):
            return self.matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, libadcc.Tensor) for elem in other):
                return [self.matvec(ov) for ov in other]
        return NotImplemented


def transfer_amplitude_cvs_to_full(mospaces_cvs, cvs, symmetry_full, tol=1e-12):
    """
    Take a CVS vector `cvs` taken from the MoSpace `mospaces_cvs` and copy
    its data into a "full" ADC amplitude constructed according to the symmetry
    `symmetry_full`. Notice that this will fail if the symmetry of `cvs` is
    not compatible (e.g. differs in targeted spin manifold, restricted versus
    unrestricted etc.).
    """
    if cvs.keys() != symmetry_full.keys():
        raise ValueError("Blocks present in CVS and full vector should agree.")
    if not mospaces_cvs.has_core_occupied_space:
        raise ValueError("Should have CVS enabled in mospaces_cvs")
    if not isinstance(cvs, AmplitudeVector):
        raise TypeError("Expected cvs to be an AmplitudeVector.")
    mospaces_full = symmetry_full["ph"].mospaces
    n_c_a = mospaces_cvs.n_orbs_alpha("o2")
    n_o_a = mospaces_cvs.n_orbs_alpha("o1")
    n_of_a = mospaces_full.n_orbs_alpha("o1")

    cvs_contiguous_in_full = (
        mospaces_cvs.core_orbitals[:n_c_a]
        == mospaces_full.occupied_orbitals[:n_c_a]
        and mospaces_cvs.core_orbitals[n_c_a:]
        == mospaces_full.occupied_orbitals[n_of_a:n_of_a + n_c_a]
    )
    if not cvs_contiguous_in_full:
        raise NotImplementedError(
            "For transfer CVS to full only the case where the CVS space is "
            "contiguous in the full space has been implemented, i.e. when the "
            "`core_orbitals` are chosen as an integer in the run_adc routines."
        )
    full = AmplitudeVector(**{block: Tensor(sym)
                              for block, sym in symmetry_full.items()})

    # Singles block
    assert cvs.ph.subspaces == ["o2", "v1"]
    assert full.ph.subspaces == ["o1", "v1"]
    assert cvs.ph.shape == (mospaces_cvs.n_orbs("o2"), mospaces_cvs.n_orbs("v1"))

    cvs_ph = cvs.ph.to_ndarray()
    full_ph = np.zeros(full.ph.shape)
    full_ph[0:n_c_a, :] = cvs_ph[:n_c_a, :]
    full_ph[n_of_a:n_of_a + n_c_a, :] = cvs_ph[n_c_a:, :]
    full.ph.set_from_ndarray(full_ph, tol)

    # Doubles block
    if "pphh" in cvs:
        assert cvs.pphh.subspaces == ["o1", "o2", "v1", "v1"]
        assert full.pphh.subspaces == ["o1", "o1", "v1", "v1"]

        # Note 1/sqrt(2) factor is there because the ocvv ocurrs twice
        # in the full amplitude, but only once in the CVS amplitude.
        cvs_pphh = cvs.pphh.to_ndarray() / np.sqrt(2)
        cvs_pphh_aa = cvs_pphh[:n_o_a, :n_c_a, :, :]
        cvs_pphh_ab = cvs_pphh[:n_o_a, n_c_a:, :, :]
        cvs_pphh_ba = cvs_pphh[n_o_a:, :n_c_a, :, :]
        cvs_pphh_bb = cvs_pphh[n_o_a:, n_c_a:, :, :]

        full_pphh = np.zeros(full.pphh.shape)
        full_pphh[n_c_a:n_of_a, 0:n_c_a, :, :] = cvs_pphh_aa
        full_pphh[0:n_c_a, n_c_a:n_of_a, :, :] = (
            -cvs_pphh_aa.transpose((1, 0, 2, 3))
        )
        full_pphh[n_c_a:n_of_a, n_of_a:n_of_a + n_c_a, :, :] = cvs_pphh_ab
        full_pphh[n_of_a:n_of_a + n_c_a, n_c_a:n_of_a, :, :] = (
            -cvs_pphh_ab.transpose((1, 0, 2, 3))
        )
        full_pphh[n_of_a + n_c_a:, 0:n_c_a, :, :] = cvs_pphh_ba
        full_pphh[0:n_c_a, n_of_a + n_c_a:, :, :] = (
            -cvs_pphh_ba.transpose((1, 0, 2, 3))
        )
        full_pphh[n_of_a + n_c_a:, n_of_a:n_of_a + n_c_a, :, :] = cvs_pphh_bb
        full_pphh[n_of_a:n_of_a + n_c_a, n_of_a + n_c_a:, :, :] = (
            -cvs_pphh_bb.transpose((1, 0, 2, 3))
        )
        full.pphh.set_from_ndarray(full_pphh, tol)
    return full


def transfer_cvs_to_full(state_matrix_cvs, matrix_full, vector=None, kind=None,
                         spin_change=0, spin_block_symmetrisation="none"):
    """
    Transfer `vector` (or a list of vectors) from the CVS space to the full space
    defined by the ADC matrix `matrix_full`. `state_matrix_cvs` is either the
    CVS matrix or the state containing the solved CVS excitations. To properly
    set up the symmetry of the returned vectors, set the `kind`, `spin_change`
    and `spin_block_symmetrisation` parameters. If the first argument is an
    `ExcitedStates` object, the symmetry is automatically detected.
    """
    if kind is None and hasattr(state_matrix_cvs, "kind"):
        kind = state_matrix_cvs.kind

    if spin_change is None and spin_block_symmetrisation is None:
        if kind is None:
            raise ValueError("kind needs to be given if first argument is not an "
                             "ExcitedStates object and spin symmetry setup is not "
                             "explicitly given.")
        return transfer_cvs_to_full(state_matrix_cvs, matrix_full, vector, kind,
                                    **guess_kwargs_kind(kind))

    if vector is None:
        if hasattr(state_matrix_cvs, "excitation_vector"):
            vector = state_matrix_cvs.excitation_vector
        else:
            raise ValueError("vector needs to be given if first argument is not an "
                             "ExcitedStates object.")
    if isinstance(vector, list):
        return [transfer_cvs_to_full(state_matrix_cvs, matrix_full, v, kind,
                                     **guess_kwargs_kind(kind)) for v in vector]

    if isinstance(state_matrix_cvs, AdcMatrixlike):
        mospaces_cvs = state_matrix_cvs.mospaces
    else:
        mospaces_cvs = state_matrix_cvs.matrix.mospaces

    sym = guess_symmetries(matrix_full, spin_change=spin_change,
                           spin_block_symmetrisation=spin_block_symmetrisation)
    return transfer_amplitude_cvs_to_full(mospaces_cvs, vector, sym)
