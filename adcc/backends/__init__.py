#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import os
import warnings

from pkg_resources import parse_version

import h5py

from .InvalidReference import InvalidReference

__all__ = ["import_scf_results", "run_hf", "have_backend", "available",
           "InvalidReference"]


def is_module_available(module, min_version=None):
    """Check using importlib if a module is available."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False

    if not min_version:  # No version check
        return True

    if not hasattr(mod, "__version__"):
        warnings.warn(
            "Could not check host program {} minimal version, "
            "since __version__ tag not found. Proceeding anyway."
            "".format(module)
        )
        return True

    if parse_version(mod.__version__) < parse_version(min_version):
        warnings.warn(
            "Found host program module {}, but its version {} is below "
            "the least required (== {}). This host program will be ignored."
            "".format(module, mod.__version__, min_version)
        )
        return False
    return True


# Cache for the list of available backends ... cannot be filled right now,
# since this can lead to import loops when adcc is e.g. used from Psi4
__status = dict()


def available():
    global __status
    if not __status:
        status = {
            "pyscf": is_module_available("pyscf", "1.5.0"),
            "psi4": (is_module_available("psi4", "1.3.0")
                     and is_module_available("psi4.core")),
            "veloxchem": is_module_available("veloxchem"),  # No version info
            "molsturm": is_module_available("molsturm"),    # No version info
        }
    return sorted([b for b in status if status[b]])


def have_backend(backend):
    """Is a particular backend available?"""
    return backend in available()


def import_scf_results(res):
    """
    Import an scf_result from an SCF program. Tries to be smart
    and guess what the host program was and how to import it.
    """
    if have_backend("pyscf"):
        from . import pyscf as backend_pyscf
        from pyscf import scf

        if isinstance(res, scf.hf.SCF):
            return backend_pyscf.import_scf(res)

    if have_backend("molsturm"):
        from . import molsturm as backend_molsturm
        from molsturm.State import State

        if isinstance(res, State):
            return backend_molsturm.import_scf(res)

    if have_backend("veloxchem"):
        import veloxchem as vlx

        from . import veloxchem as backend_veloxchem

        if isinstance(res, vlx.scfrestdriver.ScfRestrictedDriver):
            return backend_veloxchem.import_scf(res)

    if have_backend("psi4"):
        import psi4
        from . import psi4 as backend_psi4

        if isinstance(res, psi4.core.HF):
            return backend_psi4.import_scf(res)

    from libadcc import HartreeFockSolution_i
    if isinstance(res, HartreeFockSolution_i):
        return res

    if isinstance(res, (dict, h5py.File)):
        from adcc.DataHfProvider import DataHfProvider
        return DataHfProvider(res)

    if isinstance(res, str):
        if os.path.isfile(res) and (res.endswith(".h5")
                                    or res.endswith(".hdf5")):
            return import_scf_results(h5py.File(res, "r"))
        else:
            raise ValueError("Unrecognised path or file extension: {}"
                             "".format(res))

    # Note: Add more backends here

    raise NotImplementedError("No means to import an SCF result of "
                              "type " + str(type(res)) + " implemented.")


def run_hf(backend, xyz, basis, **kwargs):
    """
        Run a HF calculation with a specified backend, molecule, and SCF
        parameters

        backend:        name of the backend (pyscf, psi4, or veloxchem)
        xyz:            string with coordinates in Bohr
        basis:          basis set name
        charge:         charge of the molecule
        multiplicity:   spin multiplicity 2S + 1
        conv_tol:       energy convergence tolerance
        conv_tol_grad:  convergence tolerance of the electronic gradient
        max_iter:       maximum number of SCF iterations

        Note: This function only exists for testing purposes and should
        not be used in production calculations.
    """

    if not backend:
        if len(available()) == 0:
            raise RuntimeError(
                "No supported host-program available as SCF backend. "
                "See https://adc-connect.org/latest/"
                "installation.html#install-hostprogram "
                "for installation instructions."
            )
        else:
            backend = available()[0]
        warnings.warn("No backend specified. Using {}.".format(backend))

    if not have_backend(backend):
        raise ValueError("Backend {} not found.".format(backend))
    if backend == "psi4":
        from . import psi4 as backend_hf

    elif backend == "pyscf":
        from . import pyscf as backend_hf

    elif backend == "veloxchem":
        from . import veloxchem as backend_hf

    elif backend == "molsturm":
        from . import molsturm as backend_hf

    else:
        raise NotImplementedError("No run_hf function implemented for backend "
                                  "{}.".format(backend))

    return backend_hf.run_hf(xyz, basis, **kwargs)
