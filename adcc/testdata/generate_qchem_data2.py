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
import h5py
import os
import sh
import struct
import sys
import periodictable
#import tempfile
#import yaml

from cclib.parser import QChem
import numpy as np

from static_data import xyz
from static_data import pe_potentials

from scipy import constants

eV = constants.value("Hartree energy in eV")

_qchem_template = """
$rem
method                   {method}
basis                    {basis}{purecart}
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
thresh                   15
s2thresh                 15

! scf stuff
use_libqints             true
gen_scfman               true
scf_guess                read
max_scf_cycles           0

!QSYS mem={qsys_mem}
!QSYS mem={qsys_vmem}
!QSYS wt=00:10:00
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

_basis_template = """
$basis
{basis_string}
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
    "631g": "6-31g",
    "6311g": "6-311g",
}

molecule_remap = {
    "methox": "r2methyloxirane",
    "hf3": "hf",
}

def get_multiplicity(xyz, charge=0):
    atoms = [l.strip().split()[0].capitalize()
             for l in xyz.splitlines() if l.strip()]
    n_electrons = sum([getattr(periodictable, a).number for a in atoms])
    n_electrons -= charge
    if n_electrons % 2 == 1:
        return 2
    else:
        return 1

def clean_xyz(xyz):
    return "\n".join(x.strip() for x in xyz.splitlines())


def generate_qchem_input_file(fname, method, basis, coords, basis_string=None,
                              purecart=None,
                              potfile=None, charge=0, multiplicity=1, memory=8000,
                              singlet_states=5, triplet_states=0, bohr=True,
                              maxiter=160, conv_tol=10, n_core_orbitals=0):
    nguess_singles = 2 * max(singlet_states, triplet_states)
    max_ss = 5 * nguess_singles
    qsys_mem = "{:d}gb".format(max(memory // 1000, 1) + 5)
    qsys_vmem = qsys_mem
    pe = potfile is not None
    if basis_string is not None:
        assert purecart is not None
        purecart = "\npurecart                 {:d}".format(purecart)
        basis = "gen"
    else:
        purecart = ""
        basis = basis_remap[basis]
    qci = _qchem_template.format(
        method=_method_dict[method],
        basis=basis,
        purecart=purecart,
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
        pe=pe,
        qsys_mem=qsys_mem,
        qsys_vmem=qsys_vmem
    )
    if pe:
        qci += _pe_template.format(potfile=potfile)
    if basis_string is not None:
        qci += _basis_template.format(basis_string=basis_string)
    qc_file = open(fname, "w")
    qc_file.write(qci)
    qc_file.close()

def mkdir(dirname):
    os.makedirs(dirname, exist_ok=True)
    return dirname

class QchemSavedir(object):
    filename_dict = {
        "mo_coeffs":          53,
        "density_matrix_ao":  54,
        "fock_matrix_ao":     58,
        "mult_matrix_ao":     63,
        "energies":           99,
        "overlap_matrix_ao": 320,
        "dimensions":        819,
    }

    def __init__(self, dirname, hdf5_filename, basis):
        self.hdf5_file = h5py.File(hdf5_filename, "r")
        self.savedir = mkdir(dirname)

    def write(self):
        self._write_mo_coeffs()
        self._write_dims()
        self._write_energy()
        self._write_density()
        self._write_fock()

    def _write_mo_coeffs(self):
        coeffs = np.array(self.hdf5_file["orbcoeff_fb"])
        orbe = np.array(self.hdf5_file["orben_f"])
        self._write_data(coeffs.tobytes() + orbe.tobytes(), "mo_coeffs")

    def _write_dims(self):
        n_bas, n_orbs = np.array(self.hdf5_file["orbcoeff_fb"]).shape
        cartesian = int(np.array(self.hdf5_file["cartesian_angular_functions"]))
        purecart = 2222 if cartesian else 1111
        n_fragments = 0
        self._write_data(struct.pack("<iiii", n_bas, n_orbs,
                         purecart, n_fragments), "dimensions")

    def _write_energy(self):
        scf_energy = float(np.array(self.hdf5_file["energy_scf"]))
        data = struct.pack("<"+"d"*12, 0., scf_energy, *[0., ]*10)
        self._write_data(data, "energies")

    def _write_density(self):
        n_orbs_alpha = int(np.array(self.hdf5_file["n_orbs_alpha"]))
        coeffs = np.array(self.hdf5_file["orbcoeff_fb"])
        coeffs_a = coeffs[:n_orbs_alpha, :]
        coeffs_b = coeffs[n_orbs_alpha:, :]
        density_aa = coeffs_a @ coeffs_a.T
        density_bb = coeffs_b @ coeffs_b.T
        self._write_data(density_aa.tobytes() + density_bb.tobytes(),
                         "density_matrix_ao")

    def _write_fock(self):
        fock_fb = np.array(self.hdf5_file["fock_fb"])
        self._write_data(fock_fb.tobytes(), "fock_matrix_ao")

    def _write_data(self, data, name):
        with open(self._get_filename(name), "wb") as out:
            out.write(data)

    def _get_filename(self, name):
        filename_id = self.filename_dict[name]
        return os.path.join(self.savedir, "{:d}.0".format(filename_id))

def run_dumper_script(path, rundir):
    curdir = os.getcwd()
    os.chdir(rundir)
    out = sh.python3(path)
    #print(out)
    os.chdir(curdir)

def generate_qchem_savedir(savedir, molecule, method, basis):
    tmpdir = os.path.split(savedir)[0] or "."
    pyscf_data_dumper_script = os.path.join(
        os.path.abspath(os.path.split(sys.argv[0])[0]),
        "generate_hfdata_{:s}_{:s}.py".format(molecule, basis))
    pyscf_hfdata_hdf5_filename = os.path.join(
        tmpdir, "{:s}_{:s}_hfdata.hdf5".format(molecule, basis))
    try:
        os.stat(pyscf_hfdata_hdf5_filename)
    except FileNotFoundError:
        run_dumper_script(pyscf_data_dumper_script, tmpdir)
    QchemSavedir(savedir, pyscf_hfdata_hdf5_filename, basis).write()
    f = h5py.File(pyscf_hfdata_hdf5_filename)
    basis_string = "\n".join(f["qchem_formatted_basis"].asstr())
    multiplicity = int(np.array(f["spin_multiplicity"]))
    cartesian_angular_functions = bool(np.array(
        f["cartesian_angular_functions"]))
    charge = int(np.array(f["charge"]))
    return basis_string, multiplicity, cartesian_angular_functions, charge

def dump_qchem(molecule, method, basis, **kwargs):
    #with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = mkdir("tmpdir")
    if True:
        pe = kwargs.get("potfile", None) is not None
        if pe:
            basename = f"{molecule}_{basis}_pe_{method}"
        else:
            basename = f"{molecule}_{basis}_{method}"
        infile = os.path.join(tmpdir, basename + ".in")
        outfile = os.path.join(tmpdir, basename + ".out")
        savedir = os.path.join(tmpdir, basename + "_savedir")
        geom = xyz[molecule_remap.get(molecule, molecule)].strip()
        """
        ret = {
            "molecule": molecule,
            "method": method,
            "basis": basis,
            "pe": pe,
            "geometry": geom
        }
        """

        basis_string, multiplicity, cartesian_angular_functions, charge = generate_qchem_savedir(savedir, molecule, method, basis)
        purecart = 2222 if cartesian_angular_functions else 1111
        kwargs.update(
            {"charge": charge,
             "multiplicity": multiplicity,
             "basis_string": basis_string,
             "purecart": purecart,
             "charge": charge}
        )
        generate_qchem_input_file(infile, method, basis,
                                  geom, **kwargs)
        """
        # only works with my (ms) fork of cclib
        # github.com/maxscheurer/cclib, branch dev-qchem
        sh.qchem(infile, outfile)
        res = QChem(outfile).parse()
        ret["oscillator_strength"] = res.etoscs
        ret["excitation_energy"] = res.etenergies / eV
        if pe:
            ret["pe_ptss_correction"] = np.array(res.peenergies["ptSS"]) / eV
            ret["pe_ptlr_correction"] = np.array(res.peenergies["ptLR"]) / eV
        for key in ret:
            if isinstance(ret[key], np.ndarray):
                ret[key] = ret[key].tolist()
        return basename, ret
        """

def determine_systems():
    base_path = os.path.abspath(os.path.split(sys.argv[0])[0])
    systems = []
    hfdata_generator_scripts = [
        f for f in os.listdir(base_path) if
        f.startswith("generate_hfdata") and os.path.splitext(f)[1] == ".py"]
    for script in hfdata_generator_scripts:
        molecule, basis = os.path.splitext(script)[0].split("_")[2:4]
        systems.append({"molecule": molecule, "basis": basis})
    return systems

def main():
    methods = ["adc1", "adc2", "adc3"]
    systems = determine_systems()
    for system in systems:
        molecule = system["molecule"]
        basis = system["basis"]
        for method in methods:
            dump_qchem(molecule, method, basis)
    """
    basissets = ["sto3g", "ccpvdz"]
    qchem_results = {}
    for method in methods:
        for basis in basissets:
            key, ret = dump_qchem("formaldehyde", method, basis,
                                  potfile=pe_potentials["fa_6w"])
            qchem_results[key] = ret
            print(f"Dumped {key}.")
    with open("qchem_dump.yml", "w") as yamlout:
        yaml.safe_dump(qchem_results, yamlout)
    """


if __name__ == "__main__":
    main()
