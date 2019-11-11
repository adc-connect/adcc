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

from libadcc import MoSpaces, ReferenceState


class FormatIndexBase:
    def __init__(self, *args, **kwargs):
        pass

    def optimise_formatting(self, space_index_pairs):
        pass

    def format(self, space, idx, concat_spin=True):
        raise NotImplementedError("Implement the format function.")


class FormatIndexHfProvider(FormatIndexBase):
    def __init__(self, refstate, max_digits=1):
        """
        Format the given index for the given orbital subspace
        looking up it's equivalent index in the HF Provider's
        indexing convention. Returns a tuple, namely the formatted
        index and the corresponding spin block.

        Parameters
        ----------

        refstate : adcc.ReferenceState
            ReferenceState to use to get host program information

        max_digits : int, optional
            Number of digits to reserve for the host index
        """
        if not isinstance(refstate, ReferenceState):
            raise TypeError(f"Unsupported type: {str(refstate)}")
        # TODO some backends such as pyscf have a 1-based indexing
        #      ... this should be worked in here!
        self.base_index = 0
        self.mospaces = refstate.mospaces
        self.max_digits = max_digits
        self.noa = self.mospaces.n_orbs_alpha("f")

    def _translate_index(self, space, idx):
        ihf = self.mospaces.map_index_hf_provider[space][idx]
        if ihf < self.noa:
            return self.base_index + ihf, "a"
        else:
            return self.base_index + ihf - self.noa, "b"

    def optimise_formatting(self, space_index_pairs):
        """
        Optimise the formatting parameters of this class in order to be able to
        nicely produce equivalently formatted tensor format strings for all the
        passed spaces-index pairs.

        This function can be called multiple times.
        """
        log_max_idx = int(np.log(max(self._translate_index(space, idx)[0]
                                     for space, idx in space_index_pairs)))
        self.max_digits = max(log_max_idx, self.max_digits, 1)

    def format(self, space, idx, concat_spin=True):
        """Format the provided space-index pair and return resulting string"""
        fstr = "{:" + str(self.max_digits) + "d}"
        tidx, spin = self._translate_index(space, idx)
        if concat_spin:
            return fstr.format(tidx) + spin
        else:
            return fstr.format(tidx), spin

    @property
    def max_n_characters(self):
        """
        The maximum number of characters needed for a formatted index (excluding
        the "a" or "b" spin string) according to the current optimised formatting.
        """
        return self.max_digits


class FormatIndexAdcc(FormatIndexBase):
    def __init__(self, refstate_or_mospaces, max_digits=1):
        """
        Format the given index for the given orbital subspace
        by simply pretty-printing the passed space and index data.

        Parameters
        ----------

        refstate_or_mospaces : adcc.ReferenceState or adcc.MoSpaces
            ReferenceState to use to get host program information

        max_digits : int, optional
            Number of digits to reserve for the host index
        """
        if isinstance(refstate_or_mospaces, ReferenceState):
            self.mospaces = refstate_or_mospaces.mospaces
        elif isinstance(refstate_or_mospaces, MoSpaces):
            self.mospaces = refstate_or_mospaces
        else:
            raise TypeError(f"Unsupported type: {str(refstate_or_mospaces)}")
        self.max_digits = max_digits

    def _translate_index(self, space, idx):
        noa = self.mospaces.n_orbs_alpha(space)
        if idx < noa:
            return space, idx, "a"
        else:
            return space, idx - noa, "b"

    def optimise_formatting(self, space_index_pairs):
        """
        Optimise the formatting parameters of this class in order to be able to
        nicely produce equivalently formatted tensor format strings for all the
        passed spaces-index pairs.

        This function can be called multiple times.
        """
        maxlen = max(self._translate_index(space, idx)[1]
                     for space, idx in space_index_pairs)
        log_max_idx = int(np.log(max(1, maxlen)))
        self.max_digits = max(log_max_idx, self.max_digits, 1)

    def format(self, space, idx, concat_spin=True):
        """Format the provided space-index pair and return resulting string"""
        fstr = "({:2s} {:" + str(self.max_digits) + "d})"
        space, tidx, spin = self._translate_index(space, idx)
        if concat_spin:
            return fstr.format(space, tidx) + spin
        else:
            return fstr.format(space, tidx), spin

    @property
    def max_n_characters(self):
        """
        The maximum number of characters needed for a formatted index (excluding
        the "a" or "b" spin string) according to the current optimised formatting.
        """
        return 5 + self.max_digits


class FormatIndexHomoLumo(FormatIndexBase):
    def __init__(self, refstate, max_digits=1, use_hoco=True):
        """
        Format the given index for the given orbital subspace
        by translating it into a form HOMO-1, LUMO+4 and so on.
        The special label HOCO can be optionally used to refer
        to the highest occupied orbital of the core space
        of CVS calculations.

        Parameters
        ----------
        refstate : adcc.ReferenceState
            ReferenceState to use to get host program information

        max_digits : int, optional
            Number of digits to reserve for the HOMO/LUMO/HOCO
            offset.
        use_hoco : bool, optional
            Use the special HOCO indicator to point to the highest-energy
            orbital of the CVS core space (True) or treat it relative
            to the HOMO as well (False).
        """
        if not isinstance(refstate, ReferenceState):
            raise TypeError(f"Unsupported type: {str(refstate)}")
        self.mospaces = refstate.mospaces
        self.maxlen_offset = max_digits + 1  # + 1 for "+"/"-" string
        self.n_alpha = refstate.n_alpha
        self.n_beta = refstate.n_beta
        self.use_hoco = use_hoco

        if not self.mospaces.has_core_occupied_space:
            self.use_hoco = False

        # I do not know the convention or whether this even makes
        # sense in the cases where there is no closed-shell or no Aufbau
        # TODO Expand this class there is something sensible
        closed_shell = refstate.n_alpha == refstate.n_beta
        if not refstate.is_aufbau_occupation or not closed_shell:
            raise ValueError("format_homolumo only produces the right results "
                             "for closed-shell references with an Aufbau "
                             "occupation")

    def _translate_index(self, space, idx):
        # Deal with core-occupied orbitals first:
        if self.use_hoco and space == "o2":
            noa = self.mospaces.n_orbs_alpha("o2")
            no = self.mospaces.n_orbs("o2")
            if idx == noa - 1:
                return "HOCO", "", "a"
            elif idx == no - 1:
                return "HOCO", "", "b"
            elif idx < noa:
                return "HOCO", "-" + str(noa - 1 - idx), "a"
            else:
                return "HOCO", "-" + str(no - 1 - idx), "b"
        else:
            ihf = self.mospaces.map_index_hf_provider[space][idx]
            ifull = self.mospaces.map_index_hf_provider["f"].index(ihf)
            spin = "a"
            noa = self.mospaces.n_orbs_alpha("f")
            assert noa == self.mospaces.n_orbs_beta("f")
            if ifull >= noa:
                spin = "b"
                ifull = ifull - noa

            ihomo = self.n_alpha - 1
            ilumo = ihomo + 1
            if ifull < ihomo:
                return "HOMO", "-" + str(ihomo - ifull), spin
            elif ifull == ihomo:
                return "HOMO", "", spin
            elif ifull == ilumo:
                return "LUMO", "", spin
            elif ifull > ilumo:
                return "LUMO", "+" + str(ifull - ilumo), spin

    def optimise_formatting(self, space_index_pairs):
        """
        Optimise the formatting parameters of this class in order to be able to
        nicely produce equivalently formatted tensor format strings for all the
        passed spaces-index pairs.

        This function can be called multiple times.
        """
        maxlen = max(len(self._translate_index(space, idx)[1])
                     for space, idx in space_index_pairs)
        self.maxlen_offset = max(maxlen, self.maxlen_offset, 2)

    def format(self, space, idx, concat_spin=True):
        word, offset, spin = self._translate_index(space, idx)
        fmt = "{:s}{:>" + str(self.maxlen_offset) + "s}"
        if concat_spin:
            return fmt.format(word, offset) + spin
        else:
            return fmt.format(word, offset), spin

    @property
    def max_n_characters(self):
        """
        The maximum number of characters needed for a formatted index (excluding
        the "a" or "b" spin string) according to the current optimised formatting.
        """
        return 4 + self.maxlen_offset
