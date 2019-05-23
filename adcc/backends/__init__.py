#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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

from .available_backends import have_backend


def import_scf_results(res):
    """
    Import an scf_result from an SCF program. Tries to be smart
    and guess what the host program was and how to import it.
    """

    if have_backend("pyscf"):
        from pyscf import scf
        from . import pyscf as backend_pyscf
        if isinstance(res, scf.hf.SCF):
            return backend_pyscf.import_scf(res)

    if have_backend("molsturm"):
        from molsturm.State import State
        from . import molsturm as backend_molsturm
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

    from libadcc import HfData
    if isinstance(res, dict):
        return HfData.from_dict(res)

    # Note: Add more backends here

    raise NotImplementedError("No means to import an SCF result of "
                              "type " + str(type(res)) + " implemented.")


def run_hf(backend, **kwargs):
    if not have_backend(backend):
        raise ValueError("Backend {} not found.".format(backend))
    if backend == "psi4":
        from . import psi4 as backend_psi4
        return backend_psi4.run_hf(**kwargs)
    elif backend == "pyscf":
        from . import pyscf as backend_pyscf
        return backend_pyscf.run_hf(**kwargs)
    elif backend == "veloxchem":
        from . import veloxchem as backend_vlx
        if "conv_tol" in kwargs:
            kwargs.pop("conv_tol")
        return backend_vlx.run_hf(**kwargs)
    else:
        raise NotImplementedError(
            "No run_hf function implemented for backend {}.".format(backend)
        )
