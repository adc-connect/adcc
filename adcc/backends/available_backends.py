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


def is_module_available(module):
    import importlib

    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


status = {
    "pyscf": is_module_available("pyscf"),
    "psi4": is_module_available("psi4") and is_module_available("psi4.core"),
    "veloxchem": is_module_available("veloxchem"),
    "molsturm": is_module_available("molsturm"),
}


available = sorted([b for b in status if status[b]])


def first_available():
    if len(available) == 0:
        raise RuntimeError("No backend available.")
    else:
        return available[0]


def have_backend(backend):
    return status.get(backend, False)
