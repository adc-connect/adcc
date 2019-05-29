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


def is_pyscf_available():
    try:
        from pyscf import scf
        return True
    except ImportError:
        return False


def is_psi4_available():
    try:
        import psi4
        return True
    except ImportError:
        return False


def is_molsturm_available():
    try:
        import molsturm
        return True
    except ImportError:
        return False


def is_veloxchem_available():
    try:
        import veloxchem

        return True
    except ImportError:
        return False


status = {
    "pyscf": is_pyscf_available(),
    "psi4": is_psi4_available(),
    "veloxchem": is_veloxchem_available(),
    "molsturm": is_molsturm_available(),
}


available = sorted([b for b in status if status[b]])


def first_available():
    if len(available) == 0:
        raise RuntimeError("No backend available.")
    else:
        return available[0]


def available():
    return sorted([b for b in status if status[b]])


def have_backend(backend):
    return status.get(backend, False)
