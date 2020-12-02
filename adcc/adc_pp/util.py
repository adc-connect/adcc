#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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


def check_singles_amplitudes(spaces, *amplitudes):
    check_have_singles_block(*amplitudes)
    check_singles_subspaces(spaces, *amplitudes)


def check_doubles_amplitudes(spaces, *amplitudes):
    check_have_doubles_block(*amplitudes)
    check_doubles_subspaces(spaces, *amplitudes)


def check_have_singles_block(*amplitudes):
    if any("ph" not in amplitude.blocks_ph for amplitude in amplitudes):
        raise ValueError("ADC(0) level and "
                         "beyond expects an excitation amplitude with a "
                         "singles part.")


def check_have_doubles_block(*amplitudes):
    if any("pphh" not in amplitude.blocks_ph for amplitude in amplitudes):
        raise ValueError("ADC(2) level and "
                         "beyond expects an excitation amplitude with a "
                         "singles and a doubles part.")


def check_singles_subspaces(spaces, *amplitudes):
    for amplitude in amplitudes:
        u1 = amplitude.ph
        if u1.subspaces != spaces:
            raise ValueError("Mismatch in subspaces singles part "
                             f"(== {u1.subspaces}), where {spaces} "
                             "was expected.")


def check_doubles_subspaces(spaces, *amplitudes):
    for amplitude in amplitudes:
        u2 = amplitude.pphh
        if u2.subspaces != spaces:
            raise ValueError("Mismatch in subspaces doubles part "
                             f"(== {u2.subspaces}), where "
                             f"{spaces} was expected.")
