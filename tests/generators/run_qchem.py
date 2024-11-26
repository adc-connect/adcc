from import_qchem_data import import_excited_states, import_groundstate
from qchem_savedir import QchemSavedir
import test_cases

from adcc.hdf5io import _extract_dataset
from adcc.AdcMethod import AdcMethod

from pathlib import Path
import numpy as np
import h5py
import os
import subprocess
import tempfile


_testdata_dirname = "data"
_qchem_context_file = "context.hdf5"


def run_qchem(test_case: test_cases.TestCase, method: AdcMethod,
              import_states: bool = True,
              import_gs: bool = False) -> tuple[dict | None, dict | None]:
    """
    Run a qchem calculation for the given test case and method on top
    of the previously generated pyscf results. The results of the qchem
    calculation will are imported depending on the given parameters:
    - if "import_states" is set, the excited states data will be imported.
    - if "import_gs" is set, the ground state MP data will be imported.

    Returns
    -------
    tuple[dict | None, dict | None]
        Tuple containing the excited states and ground state data.
    """
    # load the pyscf result from the hdf5 file
    hdf5_file = open_pyscf_result(test_case)
    # TODO: switch delete to True once everything works
    # create a temporary directoy in the generators folder for the qchem calculation
    with tempfile.TemporaryDirectory(prefix=test_case.file_name,
                                     dir=Path(__file__).parent,
                                     delete=False) as tmpdir:
        # create the savedir in the temporary dir
        tmpdir = Path(tmpdir)
        savedir = tmpdir / "savedir"
        savedir.mkdir()
        generate_qchem_savedir(pyscf_data=hdf5_file, savedir=savedir)
        # extrac the relevant data from the pyscf data and write the qchem infile
        _, basis_def = _extract_dataset(hdf5_file["qchem_formatted_basis"])
        _, xyz = _extract_dataset(hdf5_file["xyz"])
        _, xyz_unit = _extract_dataset(hdf5_file["xyz_unit"])
        bohr = xyz_unit.lower() == "bohr"
        purecart = determine_purecart(hdf5_file)
        infile = tmpdir / f"{test_case.file_name}_{method.name}.in"
        outfile = f"{test_case.file_name}_{method.name}.out"
        generate_qchem_input_file(
            infile=infile, method=method.name, basis=test_case.basis, xyz=xyz,
            charge=test_case.charge, multiplicity=test_case.multiplicity,
            basis_definition=basis_def, purecart=purecart, potfile=None,
            bohr=bohr
        )
        # call qchem and wait for completion
        execute_qchem(infile.name, outfile, savedir.name, tmpdir.resolve())
        # after the calculation we should have the context file in the tmpdir
        context_file = tmpdir / _qchem_context_file
        if not context_file.exists():
            raise FileNotFoundError(f"Expected the qchem data in {context_file} "
                                    "after the qchem calculation.")
        context_file = h5py.File(context_file, "r")
        # import the excited state data to a nested dict {kind: {n: data}}
        states = None
        if import_states:
            states = import_excited_states(context_file, method=method)
        # import the ground state data as flat dict
        gs_data = None
        if import_gs:
            gs_data = import_groundstate(context_file)
    return states, gs_data


def open_pyscf_result(test_case: test_cases.TestCase) -> h5py.File:
    """
    Load the result of the dumped pyscf SCF calculation.
    """
    data_folder = Path(__file__).resolve().parent.parent / _testdata_dirname
    fname = f"{test_case.file_name}_hfdata.hdf5"
    hdf5_file = data_folder / fname
    if not hdf5_file.exists():
        raise FileNotFoundError("Could not find hdf5 file with pyscf HF data "
                                f"{test_case.file_name}. Have been looking for "
                                f"{hdf5_file}.")
    return h5py.File(hdf5_file, "r")


def generate_qchem_savedir(pyscf_data: h5py.File, savedir: str) -> None:
    """
    Writes the pyscf SCF result in the given savedir folder.
    """
    savedir_writer = QchemSavedir(savedir)
    _, scf_energy = _extract_dataset(pyscf_data["energy_scf"])
    _, mo_coeffs = _extract_dataset(pyscf_data["orbcoeff_fb"])
    _, fock_ao = _extract_dataset(pyscf_data["fock_bb"])
    _, orb_energies = _extract_dataset(pyscf_data["orben_f"])
    # compute the density in the ao basis
    ao_density_aa, ao_density_bb = compute_ao_density(pyscf_data)
    purecart = determine_purecart(pyscf_data)
    savedir_writer.write(
        scf_energy=scf_energy, mo_coeffs=mo_coeffs, fock_ao=fock_ao,
        orb_energies=orb_energies, ao_density_aa=ao_density_aa,
        ao_density_bb=ao_density_bb, purecart=purecart
    )


def compute_ao_density(pyscf_data: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the SCF density matrix in the AO basis from the pyscf MO
    coefficients.
    """
    _, n_orbs_alpha = _extract_dataset(pyscf_data["n_orbs_alpha"])
    _, mo_coeffs = _extract_dataset(pyscf_data["orbcoeff_fb"])
    mo_coeffs_a = mo_coeffs[:n_orbs_alpha, :]
    mo_coeffs_b = mo_coeffs[n_orbs_alpha:, :]
    return mo_coeffs_a @ mo_coeffs_a.T, mo_coeffs_b @ mo_coeffs_b.T


def determine_purecart(pyscf_data: h5py.File) -> int:
    """
    Checks the key "cartesian_angular_functions" in the pyscf data to determine
    the qchem purecart input variable.
    """
    _, use_cart_angular_funcs = (
        _extract_dataset(pyscf_data["cartesian_angular_functions"])
    )
    return 2222 if use_cart_angular_funcs else 1111


def execute_qchem(infile: str, outfile: str, savedir: str, workdir: str) -> None:
    """
    Execute Qchem by calling 'qchem' which needs to be some executable available in
    the path. The function defines the QCSCRATCH environment variable to point to
    the workdir folder. The QCAUX variable needs to be set either before calling the
    function or inside the 'qchem' script.
    The arguments 'infile', 'outfile' and 'savedir' are forwarded to the 'qchem'
    executable/script.

    Parameters
    ----------
    infile: str
        Name of the input file.
    outfile: str
        Name of the output file.
    savedir: str
        Name of the directory that contains the SCF result to read in.
    workdir: str
        Directory in which infile, oufile and savedir are located.
    """
    # modify the current working directory to workdir
    cwd = Path.cwd().resolve()
    os.chdir(workdir)
    # set the QCSCRATCH environment variable for the command
    env = os.environ.copy()
    env["QCSCRATCH"] = workdir
    # check_returncode raises an error if the return code is not zero.
    subprocess.run(
        ["qchem", infile, outfile, savedir], env=env
    ).check_returncode()
    # revert back to the original workdir
    os.chdir(cwd)


def clean_xyz(xyz: str):
    return "\n".join(line.strip() for line in xyz.splitlines())


def generate_qchem_input_file(infile: str, method: str, basis: str, xyz: str,
                              charge: int = 0, multiplicity: int = 1,
                              basis_definition: str = None, purecart: int = None,
                              potfile: str = None, memory: int = 10000,  # in mb
                              bohr: bool = True,
                              singlet_states: int = 5, triplet_states: int = 0,
                              maxiter: int = 160, conv_tol: int = 10,
                              n_core_orbitals: int = 0) -> None:
    """
    Generates a qchem input file for the given test case and method.
    """
    nguess_singles = 2 * max(singlet_states, triplet_states)
    max_ss = 5 * nguess_singles
    qsys_mem = "{:d}gb".format(max(memory // 1000, 1) + 5)
    qsys_vmem = qsys_mem
    pe = potfile is not None
    custom_basis = basis_definition is not None

    if custom_basis:
        assert purecart is not None  # has to be provided for a custom basis
        basis = "gen\npurecart                 {:d}".format(purecart)

    input = _qchem_template.format(
        method=_method_dict[method],
        basis=basis,
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
        xyz=clean_xyz(xyz),
        pe=pe,
        qsys_mem=qsys_mem,
        qsys_vmem=qsys_vmem
    )
    if pe:
        input += _pe_template.format(potfile=potfile)
    if custom_basis:
        input += _basis_template.format(basis_definition=basis_definition)
    # build the inputfile name
    with open(infile, "w") as qc_file:
        qc_file.write(input)


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
adc_print                3
adc_export_test_data     2  ! exports the full context to context.hdf5

! integral thresholds
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
{basis_definition}
$end
"""

_method_dict = {
    "adc0": "adc(0)",
    "adc1": "adc(1)",
    "adc2": "adc(2)",
    "adc2x": "adc(2)-x",
    "adc3": "adc(3)"
}
