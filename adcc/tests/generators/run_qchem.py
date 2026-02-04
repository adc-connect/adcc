from adcc.tests.generators.import_qchem_data import (
    import_excited_states, import_groundstate, DataImportError
)
from adcc.tests.generators.qchem_savedir import QchemSavedir
from adcc.tests import testcases

from adcc.hdf5io import _extract_dataset
from adcc.AdcMethod import AdcMethod
from adcc import ReferenceState

from collections.abc import Sequence
from pathlib import Path
from typing import cast
import numpy as np
import h5py
import itertools
import os
import shutil
import subprocess
import tempfile


_testdata_dirname = "data"
_qchem_context_file = "context.hdf5"


def run_qchem(test_case: testcases.TestCase, method: AdcMethod, case: str,
              import_states: bool = True, import_gs: bool = False,
              run_qchem_scf: bool = False,
              import_nstates: int | None = None, n_states: int = 0,
              n_singlets: int = 0, n_triplets: int = 0, n_spin_flip: int = 0,
              **kwargs) -> tuple[dict | None, dict | None]:
    """
    Run a qchem calculation for the given test case and method on top
    of the previously generated pyscf results.

    Parameters
    ----------
    test_case: test_case.TestCase
        The test case to run qchem for.
    method: AdcMethod
        The adc method to run qchem with, e.g., adc2, or cvs-adc2.
    case: str
        The "adc-case" to run with qchem, e.g., "gen" for generic adcn, "cvs"
        for cvs-adcn or "fc-cvs" for a frozen core cvs-adcn calculation.
    run_qchem_scf: bool, optional
        If set, a Qchem SCF calculation will be performed. Otherwise, it will be
        attempted to read and import the pyscf SCF data from the testdata data
        directory. (default: False)
    import_states: bool, optional
        Import the excited states data (default: True).
    import_gs: bool, optional
        Import the MP ground state data (default: False).
    import_nstates: int, optional
        The number of excited states to import after the calculation. By default
        all states are imported.

    Returns
    -------
    tuple[dict | None, dict | None]
        Tuple containing the excited states and ground state data.
    """
    # sanitize the input
    assert method.is_core_valence_separated == ("cvs" in case)
    if method.is_core_valence_separated and method.level == 0:
        raise ValueError("CVS-ADC(0) is not available in adcman.")

    if kwargs.get("gs_density_order", None) is not None:
        if method.level < 3:
            raise ValueError("gs_density_order is only available for ADC(n) with "
                             "n > 2")
        if method.is_core_valence_separated:
            raise ValueError("gs_density_order not available for CVS-ADC.")

    # get the number of core, frozen core and frozen virtual orbitals
    # from the test case.
    n_core_orbitals = test_case.core_orbitals if "cvs" in case else 0
    n_frozen_core = test_case.frozen_core if "fc" in case else 0
    n_frozen_virtual = test_case.frozen_virtual if "fv" in case else 0
    # create a temporary directoy for the qchem calculation
    with tempfile.TemporaryDirectory(prefix=test_case.file_name,
                                     dir=Path.cwd()) as tmpdir:
        # create the savedir in the temporary dir
        tmpdir = Path(tmpdir)
        args = {}
        # we don't want to read the pyscf SCF data:
        # -> extract the data from the TestCase
        if run_qchem_scf:
            args["bohr"] = test_case.unit.lower() == "bohr"
            args["xyz"] = test_case.xyz
            savedir = None
        else:  # Load the SCF data and create the savedir
            # load the pyscf result from the hdf5 file
            with open_pyscf_result(test_case) as hdf5_file:
                savedir = tmpdir / "savedir"
                savedir.mkdir()
                generate_qchem_savedir(
                    pyscf_data=hdf5_file, savedir=savedir,
                    core_orbitals=n_core_orbitals,
                    frozen_core=n_frozen_core, frozen_virtual=n_frozen_virtual
                )
                savedir = savedir.name
                # extrac the relevant data from the pyscf data for the infile.
                args["basis_definition"] = (
                    _extract_dataset(hdf5_file["qchem_formatted_basis"])[1]
                )
                args["xyz"] = _extract_dataset(hdf5_file["xyz"])[1]
                _, xyz_unit = _extract_dataset(hdf5_file["xyz_unit"])
                assert isinstance(xyz_unit, str)
                args["bohr"] = xyz_unit.lower() == "bohr"
                args["purecart"] = determine_purecart(hdf5_file)
        # add the user input
        args.update(kwargs)
        # use the data to write the infile
        infile = tmpdir / f"{test_case.file_name}_{method.name}.in"
        outfile = tmpdir / f"{test_case.file_name}_{method.name}.out"
        generate_qchem_input_file(
            infile=infile, adc_method=method, basis=test_case.basis,
            charge=test_case.charge, multiplicity=test_case.multiplicity,
            n_core_orbitals=n_core_orbitals, n_frozen_core=n_frozen_core,
            n_frozen_virtual=n_frozen_virtual, any_states=n_states,
            singlet_states=n_singlets, triplet_states=n_triplets,
            sf_states=n_spin_flip, run_qchem_scf=run_qchem_scf, **args
        )
        # call qchem and wait for completion
        execute_qchem(
            infile=infile.name, outfile=outfile.name, savedir=savedir,
            workdir=str(tmpdir.resolve())
        )
        # after the calculation we should have the context file in the tmpdir
        context_file = tmpdir / _qchem_context_file
        if not context_file.exists():
            raise FileNotFoundError(f"Expected the qchem data in {context_file} "
                                    "after the qchem calculation.")
        # import all relevant data from the context hdf5 file and return the
        # imported data in dictionaries.
        with h5py.File(context_file, "r") as context:
            # import the excited state data as nested dict {kind: {prop: list}}
            states = None
            if import_states:
                try:
                    states = import_excited_states(
                        context, method=method,
                        only_full_mode=test_case.only_full_mode,
                        is_spin_flip=bool(n_spin_flip),
                        import_nstates=import_nstates
                    )
                except DataImportError as e:
                    # something (expected) went wrong during import
                    # copy the output file to the working directory and abort.
                    if outfile.exists():
                        shutil.copy(outfile, Path.cwd())
                    raise e
            # import the ground state data as flat dict
            gs_data = None
            if import_gs:
                gs_data = import_groundstate(
                    context, only_full_mode=test_case.only_full_mode
                )
    return states, gs_data


def open_pyscf_result(test_case: testcases.TestCase) -> h5py.File:
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


def generate_qchem_savedir(pyscf_data: h5py.File, savedir: str | Path,
                           core_orbitals: int | None = None,
                           frozen_core: int | None = None,
                           frozen_virtual: int | None = None) -> None:
    """
    Writes the pyscf SCF result in the given savedir folder.
    """
    savedir_writer = QchemSavedir(savedir)
    _, scf_energy = _extract_dataset(pyscf_data["energy_scf"])
    assert isinstance(scf_energy, float)
    _, mo_coeffs = _extract_dataset(pyscf_data["orbcoeff_fb"])
    assert isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2
    mo_coeffs = cast(np.ndarray[tuple[int, int]], mo_coeffs)
    _, fock_ao = _extract_dataset(pyscf_data["fock_bb"])
    assert isinstance(fock_ao, np.ndarray) and fock_ao.ndim == 2
    fock_ao = cast(np.ndarray[tuple[int, int]], fock_ao)
    _, orb_energies = _extract_dataset(pyscf_data["orben_f"])
    assert isinstance(orb_energies, np.ndarray) and orb_energies.ndim == 1
    orb_energies = cast(np.ndarray[tuple[int]], orb_energies)
    # compute the density in the ao basis
    ao_density_aa, ao_density_bb = compute_ao_density(pyscf_data)
    eri_blocks = build_antisym_eri(
        pyscf_data, core_orbitals=core_orbitals, frozen_core=frozen_core,
        frozen_virtual=frozen_virtual
    )
    ao_integrals = collect_ao_integrals(pyscf_data)
    purecart = determine_purecart(pyscf_data)
    savedir_writer.write(
        scf_energy=scf_energy, mo_coeffs=mo_coeffs, fock_ao=fock_ao,
        orb_energies=orb_energies, ao_density_aa=ao_density_aa,
        ao_density_bb=ao_density_bb, eri_blocks=eri_blocks,
        ao_integrals=ao_integrals, purecart=purecart
    )


def build_antisym_eri(pyscf_data: h5py.File, core_orbitals: int | None = None,
                      frozen_core: int | None = None,
                      frozen_virtual: int | None = None
                      ) -> dict[str, np.ndarray[tuple[int, int, int, int]]]:
    """
    Builds all anti-symmetric ERI blocks (MO basis) from the pyscf data.
    Returned using keys like "ooov" or "ococ".
    """
    ret: dict[str, np.ndarray[tuple[int, int, int, int]]] = {}
    refstate = ReferenceState(
        pyscf_data, core_orbitals=core_orbitals, frozen_core=frozen_core,
        frozen_virtual=frozen_virtual
    )
    for block in itertools.product("ocv", repeat=4):
        block = "".join(block)
        if not is_canonical_eri_block(block):
            continue
        try:
            eri = getattr(refstate, block).to_ndarray()
            assert isinstance(eri, np.ndarray) and eri.ndim == 4
            ret[block] = cast(np.ndarray[tuple[int, int, int, int]], eri)
        except ValueError as e:  # CVS block not available
            if refstate.has_core_occupied_space:
                raise e
    return ret


def is_canonical_eri_block(block: Sequence[str]) -> bool:
    """Checks if a given ERI block is canonical assuming physicist notation."""
    assert len(block) == 4 and all(sp in "ocv" for sp in block)
    space_ordering = {"o": 0, "c": 1, "v": 2}
    bra = [space_ordering[sp] for sp in block[:2]]
    ket = [space_ordering[sp] for sp in block[2:]]
    bra_canonical = sorted(bra)
    ket_canonical = sorted(ket)
    if ket_canonical < bra_canonical:  # swap bra and ket
        bra_canonical, ket_canonical = ket_canonical, bra_canonical
    return bra == bra_canonical and ket == ket_canonical


def collect_ao_integrals(pyscf_data: h5py.File) -> dict:
    """
    Collects integral matrices in the AO basis from the pyscf data. Supported are
    - dipole x, y and z components. Returned as dx, dy and dz.
    """
    ret = {}
    # extract the dipole operator matrices in the AO basis.
    _, dipole = _extract_dataset(pyscf_data["multipoles/elec_1"])
    for tensor, comp in zip(dipole, ["x", "y", "z"]):
        ret[f"d{comp}"] = tensor
    return ret


def compute_ao_density(pyscf_data: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the SCF density matrix in the AO basis from the pyscf MO
    coefficients.
    """
    _, n_orbs_alpha = _extract_dataset(pyscf_data["n_orbs_alpha"])
    assert isinstance(n_orbs_alpha, int)
    _, mo_coeffs = _extract_dataset(pyscf_data["orbcoeff_fb"])
    assert isinstance(mo_coeffs, np.ndarray) and mo_coeffs.ndim == 2
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


def execute_qchem(infile: str, outfile: str, workdir: str,
                  savedir: str | None = None) -> None:
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
    savedir: str, optional
        Name of the directory that contains the SCF result to read in.
    workdir: str
        Directory in which infile, oufile and savedir are/will be located.
    """
    # modify the current working directory to workdir
    cwd = Path.cwd().resolve()
    os.chdir(workdir)
    # set the QCSCRATCH environment variable for the command
    env = os.environ.copy()
    env["QCSCRATCH"] = workdir
    # Call the executable (with or without savedir)
    if savedir is None:
        code = subprocess.run(["qchem", infile, outfile], env=env).returncode
    else:
        code = subprocess.run(
            ["qchem", infile, outfile, savedir], env=env
        ).returncode
    # qchem did not terminate normal -> copy the outputfile to the working directory
    if code:
        if Path(outfile).exists():
            shutil.copy(outfile, cwd)
        raise RuntimeError("Qchem calculation did finish with return code != 0. "
                           "Tried to copy the outputfile to the working directory.")
    # revert back to the original workdir
    os.chdir(cwd)


def clean_xyz(xyz: str):
    return "\n".join(line.strip() for line in xyz.splitlines())


def generate_qchem_input_file(infile: str | Path, adc_method: AdcMethod, basis: str,
                              xyz: str, charge: int, multiplicity: int,
                              basis_definition: str | None = None,
                              purecart: int | None = None,
                              pe_potfile: str | None = None,
                              memory: int = 10000,  # in mb
                              bohr: bool = True, any_states: int = 0,
                              singlet_states: int = 0, triplet_states: int = 0,
                              sf_states: int = 0,
                              maxiter: int = 160, conv_tol: int = 10,
                              n_core_orbitals: int | None = None,
                              n_frozen_core: int | None = None,
                              n_frozen_virtual: int | None = None,
                              max_ss: int | None = None,
                              gs_density_order: int | str | None = None,
                              isr_maxorder: int | None = None,
                              run_qchem_scf: bool = False) -> None:
    """
    Generates a qchem input file for the given test case and method.
    """
    nguess_singles = 2 * max(singlet_states, triplet_states, any_states, sf_states)
    if max_ss is None:
        max_ss = 7 * nguess_singles

    n_core_orbitals = 0 if n_core_orbitals is None else n_core_orbitals
    n_frozen_core = 0 if n_frozen_core is None else n_frozen_core
    n_frozen_virtual = 0 if n_frozen_virtual is None else n_frozen_virtual

    method = _method_dict[adc_method.name]
    if isinstance(gs_density_order, str):
        # sigma4+ -> sigma_4_plus
        gs_density_order = _gs_density_order_dict[gs_density_order]
    if gs_density_order is not None:  # append density order to the method
        method += f"\nadc_gs_density_order     {gs_density_order}"
    if isr_maxorder is not None:  # and isr_max_order
        method += f"\nadc_isr_maxorder         {isr_maxorder}"

    qsys_mem = "{:d}gb".format(max(memory // 1000, 1) + 5)
    qsys_vmem = qsys_mem

    pe = pe_potfile is not None
    custom_basis = basis_definition is not None

    if custom_basis:
        assert purecart is not None  # has to be provided for a custom basis
        basis = "gen\npurecart                 {:d}".format(purecart)

    # Adjust the input depending on whether we want to perform an qchem SCF calc
    scf_options = ["use_libqints             true",
                   "gen_scfman               true"]
    if run_qchem_scf:
        export_test_data = 2
    else:
        export_test_data = 42
        scf_options.extend(["scf_guess                read",
                            "max_scf_cycles           0"])

    input = _qchem_template.format(
        method=method,
        basis=basis,
        memory=memory,
        any_states=any_states,
        singlet_states=singlet_states,
        triplet_states=triplet_states,
        sf_states=sf_states,
        n_guesses=nguess_singles,
        bohr=bohr,
        maxiter=maxiter,
        conv_tol=conv_tol,
        max_ss=max_ss,
        charge=charge,
        multiplicity=multiplicity,
        cc_rest_occ=n_core_orbitals,
        n_frozen_core=n_frozen_core,
        n_frozen_virtual=n_frozen_virtual,
        export_test_data=export_test_data,
        scf_options="\n".join(scf_options),
        xyz=clean_xyz(xyz),
        pe=pe,
        qsys_mem=qsys_mem,
        qsys_vmem=qsys_vmem
    )
    if pe:
        input += _pe_template.format(potfile=pe_potfile)
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
ee_states                {any_states}
ee_singlets              {singlet_states}
ee_triplets              {triplet_states}
sf_states                {sf_states}
input_bohr               {bohr}
sym_ignore               true
adc_davidson_maxiter     {maxiter}
adc_davidson_conv        {conv_tol}
adc_nguess_singles       {n_guesses}
adc_davidson_maxsubspace {max_ss}
adc_prop_es              true
adc_prop_es2es           true
cc_rest_occ              {cc_rest_occ}
cc_frzn_core             {n_frozen_core}
cc_frzn_virt             {n_frozen_virtual}
adc_print                6
! 2: exports the full context to context.hdf5
! 42: additionally reads integrals from integrals.hdf5
adc_export_test_data     {export_test_data}

! integral thresholds
thresh                   15
s2thresh                 15

! scf stuff
{scf_options}

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
    "adc3": "adc(3)",
    "cvs-adc0": "cvs-adc(0)",
    "cvs-adc1": "cvs-adc(1)",
    "cvs-adc2": "cvs-adc(2)",
    "cvs-adc2x": "cvs-adc(2)-x",
    "cvs-adc3": "cvs-adc(3)"
}

_gs_density_order_dict = {
    "sigma4+": "sigma_4_plus"
}
