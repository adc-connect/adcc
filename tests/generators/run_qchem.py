from qchem_savedir import QchemSavedir
from read_qchem_data import read_qchem_data, mp_context_paths
import test_cases

from adcc.hdf5io import emplace_dict

from pathlib import Path
import numpy as np
import h5py
import subprocess
import tempfile


_testdata_dirname = "data"
_qchem_context_file = "context.hdf5"


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


def generate_qchem_savedir(pyscf_data: h5py.File, savedir: str) -> None:
    """
    Writes the pyscf SCF result in the given savedir folder.
    """
    savedir_writer = QchemSavedir(savedir)
    scf_energy = pyscf_data["energy_scf"][()]
    mo_coeffs = np.array(pyscf_data["orbcoeff_fb"])
    fock_ao = np.array(pyscf_data["fock_bb"])
    orb_energies = np.array(pyscf_data["orben_f"])
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
    n_orbs_alpha = pyscf_data["n_orbs_alpha"][()]
    mo_coeffs = np.array(pyscf_data["orbcoeff_fb"])
    mo_coeffs_a = mo_coeffs[:n_orbs_alpha, :]
    mo_coeffs_b = mo_coeffs[n_orbs_alpha:, :]
    return mo_coeffs_a @ mo_coeffs_a.T, mo_coeffs_b @ mo_coeffs_b.T


def determine_purecart(pyscf_data: h5py.File) -> int:
    """
    Checks the key "cartesian_angular_functions" in the pyscf data to determine
    the qchem purecart input variable.
    """
    use_cart_angular_funcs = pyscf_data["cartesian_angular_functions"][()]
    return 2222 if use_cart_angular_funcs else 1111


def get_qchem_formatted_basis(pyscf_data: h5py.File) -> str:
    """
    Checks the key "qchem_formatted_basis" in the pyscf data to get the
    used basis in qchem format.
    """
    # decode the byte strings as utf8 string
    return pyscf_data["qchem_formatted_basis"][()].decode()


def get_xyz_geometry(pyscf_data: h5py.File) -> str:
    """
    Checks the key "xyz" in the pyscf data for the coordinates in xyz format.
    """
    # decode the byte strings as utf8 strings
    return pyscf_data["xyz"][()].decode(), pyscf_data["xyz_unit"][()].decode()


def exec_qchem(infile: str, outfile: str, savedir: str) -> None:
    """
    Execute Qchem with the given infile by calling 'qchem' which needs
    to be available in the path.
    """
    # check_returncode raises an error if the return code is not zero.
    subprocess.run(["qchem", infile, outfile, savedir]).check_returncode()


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


def run_qchem(test_case: test_cases.TestCase, method: str) -> dict:
    """
    Run a qchem calculation for the given test case and method on top
    of the previously generated pyscf results. The results of the qchem
    calculation will are imported and returned.
    """
    # load the pyscf result from the hdf5 file
    hdf5_file = open_pyscf_result(test_case)
    # TODO: switch delete to True once everything works
    with tempfile.TemporaryDirectory(prefix=test_case.file_name,
                                     dir=Path(__file__).parent,
                                     delete=False) as tmpdir:
        # create the savedir in the temporary dir
        savedir = Path(tmpdir) / "savedir"
        savedir.mkdir()
        generate_qchem_savedir(pyscf_data=hdf5_file, savedir=savedir)
        # extrac the relevant data from the pyscf data and write the qchem infile
        basis_def = get_qchem_formatted_basis(hdf5_file)
        xyz, xyz_unit = get_xyz_geometry(hdf5_file)
        bohr = xyz_unit.lower() == "bohr"
        purecart = determine_purecart(hdf5_file)
        infile = Path(tmpdir) / f"{test_case.file_name}_{method}.in"
        outfile = Path(tmpdir) / f"{test_case.file_name}_{method}.out"
        generate_qchem_input_file(
            infile=infile, method=method, basis=test_case.basis, xyz=xyz,
            charge=test_case.charge, multiplicity=test_case.multiplicity,
            basis_definition=basis_def, purecart=purecart, potfile=None,
            bohr=bohr
        )
        # call qchem and wait for completion
        exec_qchem(infile, outfile, savedir)
        # after the calculation we should have the context file in the tmpdir
        context_file = Path(tmpdir) / _qchem_context_file
        if not context_file.exists():
            raise FileNotFoundError(f"Expected the qchem data in {context_file} "
                                    "after the qchem calculation.")
        # read the data for the MP ground state
        mp_data = read_qchem_data(context_file.name, **mp_context_paths)
    return mp_data


def dump_qchem_data(test_case: test_cases.TestCase, gs_data: dict,
                    gs_type: str = "mp") -> None:
    data_folder = Path(__file__).resolve().parent.parent / _testdata_dirname
    # dump the ground state data
    file = data_folder / f"{test_case.file_name}_adcman_{gs_type}data.hdf5"
    emplace_dict(
        mp_data, h5py.File(file, "w"), compression="gzip", compression_opts=8
    )


if __name__ == "__main__":
    cases = test_cases.get(n_expected_cases=2, name="h2o")
    for case in cases:
        mp_data = run_qchem(case, "adc2")
        dump_qchem_data(case, gs_data=mp_data)
