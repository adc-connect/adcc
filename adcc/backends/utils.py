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

import tempfile
import os


def run_pyscf_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
                 conv_tol_grad=1e-8):
    from pyscf import scf, gto
    mol = gto.M(
        atom=xyz,
        basis=basis,
        unit="Bohr"
    )
    mol.charge = charge
    # spin in the pyscf world is 2S
    mol.spin = multiplicity - 1
    mf = scf.HF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.kernel()
    return mf


def run_psi4_hf(xyz, basis, charge=0, multiplicity=1, e_convergence=1e-12,
                d_convergence=1e-8):
    import psi4
    mol = psi4.geometry("""
        {xyz}
        symmetry c1
        units au
        {charge} {multiplicity}
        """.format(xyz=xyz, charge=charge, multiplicity=multiplicity))
    psi4.core.be_quiet()
    reference = "RHF"
    if multiplicity != 1:
        reference = "UHF"
    psi4.set_options({'basis': basis,
                      'scf_type': 'pk',
                      'e_convergence': e_convergence,
                      'd_convergence': d_convergence,
                      'reference': reference})
    _, wfn = psi4.energy('SCF', return_wfn=True, molecule=mol)
    return wfn


def run_vlx_hf(xyz, basis, charge=0, multiplicity=1, conv_thresh=1e-8):
    from mpi4py import MPI
    import veloxchem as vlx

    basis_dir = os.path.abspath(os.path.join(vlx.__path__[-1],
                                             "..", "..", "..", "basis"))
    with tempfile.TemporaryDirectory() as tmpdir:
        infile = os.path.join(tmpdir, "vlx.in")
        outfile = os.path.join(tmpdir, "vlx.out")
        with open(infile, "w") as fp:
            lines = ["@jobs", "task: hf", "@end", ""]
            lines += ["@method settings",
                      "basis: {}".format(basis),
                      "basis path: {}".format(basis_dir), "@end", ""]
            lines += ["@molecule",
                      "charge: {}".format(charge),
                      "multiplicity: {}".format(multiplicity),
                      "units: bohr",
                      "xyz:\n{}".format("\n".join(xyz.split(";"))),
                      "@end"]
            fp.write("\n".join(lines))
        task = vlx.MpiTask([infile, outfile], MPI.COMM_WORLD)

        scfdrv = vlx.ScfRestrictedDriver(task.mpi_comm, task.ostream)
        # elec. gradient norm
        scfdrv.conv_thresh = conv_thresh
        scfdrv.compute(task.molecule, task.ao_basis, task.min_basis)
        scfdrv.task = task
    return scfdrv
