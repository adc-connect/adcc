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


def format_hfprovider(refstate, space, idx):
    """
    Format the given index for the given orbital subspace
    looking up it's equivalent index in the HF Provider's
    indexing convention. Returns a tuple, namely the formatted
    index and the corresponding spin block.
    """
    mospaces = refstate.mospaces
    noa = mospaces.n_orbs_alpha("f")
    ihf = mospaces.map_index_hf_provider[space][idx]
    if ihf < noa:
        return f"{ihf:3d}", "a"
    else:
        return f"{ihf:3d}", "b"


def format_adcc(refstate, space, idx):
    """
    Format the given index for the given orbital subspace
    by simply pretty-printing the passed data.
    """
    mospaces = refstate.mospaces
    noa = mospaces.n_orbs_alpha(space)
    if idx < noa:
        return f"({space:2s} {idx:3d})", "a"
    else:
        ridx = idx - noa
        return f"({space:2s} {ridx:3d})", "b"


def format_homolumo(refstate, space, idx):
    """
    Format the given index for the given orbital subspace
    by translating it into a form HOMO-1, LUMO+4 and so on.
    """
    mospaces = refstate.mospaces

    # I do not know the convention or whether this even makes
    # sense in the cases where there is no closed-shell or no Aufbau
    # TODO Expand this if there is something sensible
    closed_shell = refstate.n_alpha == refstate.n_beta
    if not refstate.is_aufbau_occupation or not closed_shell:
        raise ValueError("format_homolumo only produces the right results for "
                         "closed-shell references with an Aufbau occupation")

    # Deal with core-occupied orbitals first:
    if mospaces.has_core_occupied_space and space == "o2":
        noa = mospaces.n_orbs_alpha("o2")
        no = mospaces.n_orbs("o2")
        if idx == noa - 1:
            return "HOCO", "a"
        elif idx == no - 1:
            return "HOCO", "b"
        elif idx < noa:
            diff = "-" + str(noa - 1 - idx)
            return f"HOCO{diff:>3s}", "a"
        else:
            diff = "-" + str(no - 1 - idx)
            return f"HOCO{diff:>3s}", "b"
    else:
        ihf = mospaces.map_index_hf_provider[space][idx]
        ifull = mospaces.map_index_hf_provider["f"].index(ihf)
        noa = mospaces.n_orbs_alpha("f")
        assert noa == mospaces.n_orbs_beta("f")
        if ifull < noa:
            spin = "a"
        else:
            spin = "b"
            ifull = ifull - noa

        ihomo = refstate.n_alpha - 1
        ilumo = ihomo + 1
        if ifull < ihomo:
            diff = "-" + str(ihomo - ifull)
            return f"HOMO{diff:>3s}", spin
        elif ifull == ihomo:
            return "HOMO", spin
        elif ifull == ilumo:
            return "LUMO", spin
        elif ifull > ilumo:
            diff = "+" + str(ifull - ilumo)
            return f"LUMO{diff:>3s}", spin
