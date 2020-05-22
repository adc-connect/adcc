#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import os
import tempfile

from cclib.parser import QChem

from ..geometry import xyz
from adcc import hdf5io

_qchem_template = """
$rem
method                   {method}
basis                    {basis}
mem_total                {memory}
pe                       {pe}
ee_singlets              {singlet_states}
ee_triplets              {triplet_states}
input_bohr               {bohr}
sym_ignore               true
adc_davidson_maxiter     {maxiter}
adc_davidson_conv        {conv_tol}
adc_nguess_singles       {n_guesses}
adc_davidson_maxsubspace {max_ss}
adc_prop_es              true
cc_rest_occ              {cc_rest_occ}
$end

$molecule
{charge} {multiplicity}
{xyz}
$end
"""

_pe_template = """
$pe
potfile {potfile}
$end
"""

_method_dict = {
    "adc0": "adc(0)",
    "adc1": "adc(1)",
    "adc2": "adc(2)",
    "adc2x": "adc(2)-x",
    "adc3": "adc(3)",
    "cvs-adc0": "cvs-adc(0)",
    "cvs-adc1": "cvs-adc(1)",
    "cvs-adc2": "cvs-adc(2)",
    "cvs-adc2x": "cvs-adc(2)-x",
    "cvs-adc3": "cvs-adc(3)",
}

basis_remap = {
    "sto3g": "sto-3g",
    "def2tzvp": "def2-tzvp",
    "ccpvdz": "cc-pvdz",
}


def clean_xyz(xyz):
    return "\n".join(x.strip() for x in xyz.splitlines())


def generate_qchem_input_file(fname, method, basis, coords, potfile=None,
                              charge=0, multiplicity=1, memory=3000,
                              singlet_states=5, triplet_states=0, bohr=True,
                              maxiter=60, conv_tol=6, n_core_orbitals=0):
    nguess_singles = 2 * max(singlet_states, triplet_states)
    max_ss = 5 * nguess_singles
    pe = potfile is not None
    qci = _qchem_template.format(
        method=_method_dict[method],
        basis=basis_remap[basis],
        memory=memory,
        singlet_states=singlet_states,
        triplet_states=triplet_states,
        n_guesses=nguess_singles,
        bohr=bohr,
        maxiter=maxiter,
        conv_tol=conv_tol,
        max_ss=max_ss,
        charge=charge,
        multiplicity=multiplicity,
        cc_rest_occ=n_core_orbitals,
        xyz=clean_xyz(coords),
        pe=pe
    )
    if pe:
        qci += _pe_template.format(potfile=potfile)
    qc_file = open(fname, "w")
    qc_file.write(qci)
    qc_file.close()


def dump_qchem(molecule, method, basis, **kwargs):
    with tempfile.TemporaryDirectory() as tmpdir:
        pe = kwargs.get("potfile", None)
        if pe:
            basename = f"{molecule}_{basis}_pe_{method}"
        else:
            basename = f"{molecule}_{basis}_{method}"
        infile = os.path.join(tmpdir, basename + ".in")
        outfile = os.path.join(tmpdir, basename + ".out")
        geom = xyz[molecule].strip()
        ret = {
            "molecule": molecule,
            "method": method,
            "basis": basis,
            "pe": pe,
            "geometry": geom
        }
        generate_qchem_input_file(
            infile, method, basis, geom, **kwargs
        )
        # only works with my (ms) fork of cclib
        # github.com/maxscheurer/cclib, branch dev-qchem
        os.system("qchem {} {}".format(infile, outfile))
        res = QChem(outfile).parse()
        ret["oscillator_strengths"] = res.etoscs
        ret["state_dipole_moments_debye"] = res.etdipmoms
        ret["excitation_energies_ev"] = res.etenergies
        print(dir(res))
        if pe:
            ret["pe_ptss_corrections_ev"]
            ret["pe_ptlr_corrections_ev"]
        print(ret)
        # hdf5io.save(fname="{}_qc.hdf5".format(basename),
        #             dictionary=ret)
        # bla = hdf5io.load(fname="{}_qc.hdf5".format(basename))
        # print(bla)


def main():
    for basis in ["sto3g", "ccpvdz"]:
        dump_qchem("h2o", "adc2", basis)
        # basename = "h2o_{}_cvs_adc2".format(basis)
        # dump_qc_h2o(basename, "cvs-adc2", basis, n_core_orbitals=1)


if __name__ == "__main__":
    main()