from collections import Counter
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional


# NOTE: Can't use a dict, because TestCase has to be hashable.
@dataclass(frozen=True)
class Kinds:
    pp: tuple[str, ...] = tuple()
    ip: tuple[str, ...] = tuple()
    ea: tuple[str, ...] = tuple()

    def __getitem__(self, key: str):
        return getattr(self, key)


@dataclass(frozen=True)
class TestCase:
    name: str
    xyz: str
    unit: str  # unit of the xyz coordinates (bohr/angstrom)
    charge: int
    multiplicity: int  # = 2S+1
    basis: str
    restricted: bool
    only_full_mode: bool  # whether to run the test case only in full mode
    pe_potfile: Optional[str] = None
    core_orbitals: Optional[int] = None
    frozen_core: Optional[int] = None
    frozen_virtual: Optional[int] = None
    # the different cases for which to generate mp/adc reference data
    # generic, cvs, frozen core (fc), frozen virtual (fv), ...
    cases: tuple[str, ...] = ("gen",)
    # the available state kinds for the test system per adc type
    kinds: Kinds = Kinds()
    # the ground state density orders to generate data for, e.g.,
    # (None, 2, 3, "sigma4+")
    # where None invokes a calculation without a specific gs_density_order
    gs_density_orders: tuple = (None,)

    @property
    def file_name(self) -> str:
        """Builds a file name based on name and basis."""
        basis = self.basis.replace("-", "").replace("*", "").replace("+", "")
        return f"{self.name}_{basis}"

    @property
    def hfdata_file_name(self) -> str:
        """Builds the file name for the hfdata."""
        return f"{self.file_name}_hfdata.hdf5"

    @property
    def hfimport_file_name(self) -> str:
        """Builds the file name for the hfimport data."""
        return f"{self.file_name}_hfimport.hdf5"

    def mpdata_file_name(self, source: str) -> str:
        """Builds the file name for the mpdata."""
        assert source in ["adcc", "adcman"]
        return f"{self.file_name}_{source}_mpdata.hdf5"

    def adcdata_file_name(self, source: str, adc_method: str) -> str:
        """Builds the file name for the adc data for the given method."""
        assert source in ["adcc", "adcman"]
        return f"{self.file_name}_{source}_{adc_method}.hdf5"

    def asdict(self, *args: str, **kwargs: str) -> dict:
        """
        Exports the data of the system in a dict.
        If no arguments are provided, all fields are added to the dictionary.
        If fields are given as arguments, only the specified fields are exported
        using the name of the field as key.
        Fields can also be provided as keyword arguments of the form: field=key.
        In this case, the entry of the fields are returned using the provided
        key.
        """
        if not args and not kwargs:
            return asdict(self)
        ret = {}
        for field in args:
            ret[field] = getattr(self, field)
        for field, key in kwargs.items():
            ret[key] = getattr(self, field)
        return ret

    def filter_cases(self, adc_type: str) -> tuple[str, ...]:
        """
        Filter the available cases only returning cases that are relevant
        for the given adc_type.
        """
        # since cvs is not (yet) implemented for IP.
        if adc_type == "pp":
            return self.cases
        raise NotImplementedError(f"Filtering for adc type {adc_type} not "
                                  "implemented.")

    def validate(self):
        """
        Validates the set cases by checking that the cases are valid and ensuring
        that required data is set, e.g., that core_orbitals is defined for cvs.
        """
        # validate the reference cases
        requirements = {"cvs": "core_orbitals",
                        "fc": "frozen_core",
                        "fv": "frozen_virtual"}
        for case in self.cases:
            for component in case.split("-"):
                if component == "gen":
                    continue
                assert component in requirements
                assert getattr(self, requirements[component], None) is not None
        # validate the PP-ADC kinds
        assert len(fields(self.kinds)) == 3
        assert not self.kinds.ip
        assert not self.kinds.ea
        if self.restricted:
            assert all(kind in ["singlet", "triplet"] for kind in self.kinds.pp)
        else:
            assert all(kind in ["any", "spin_flip"] for kind in self.kinds.pp)


def kinds_to_nstates(kinds: tuple[str, ...]) -> list[str]:
    """
    Transforms the given kinds to a list of keywords to request states of the
    corresponding kind in an adc calculation.
    """
    # singlet, triplet -> n_singlets, n_triplets
    # any              -> n_states
    # spin_flip        -> n_spin_flip
    ret = []
    for kind in kinds:
        if kind == "any":
            ret.append("n_states")
        elif kind == "spin_flip":
            ret.append("n_spin_flip")
        else:
            ret.append(f"n_{kind}s")
    return ret


_xyz = {
    "h2o": ("""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    """, "Bohr"),
    #
    "cn": ("""
    C 0 0 0
    N 0 0 2.2143810738114829
    """, "Bohr"),
    #
    "hf": ("""
    H 0 0 0
    F 0 0 2.5
    """, "Bohr"),
    #
    "ch2nh2": ("""
    C -1.043771327642266  0.9031379094521343 -0.0433881118200138
    N  1.356218645077853 -0.0415928720016770  0.9214682528604154
    H -1.624635343811075  2.6013402912925274  1.0436579440747924
    H -2.522633198204392 -0.5697335292951204  0.1723619198215792
    H  2.681464678974086  1.3903093043650074  0.6074335654801934
    H  1.838098806841944 -1.5878801706882844 -0.2108367437177239
    """, "Bohr"),
    #
    "r2methyloxirane": ("""
    O        0.0000000000      0.0000000000      0.0000000000
    C        2.7197505315      0.0000000000      0.0000000000
    H        3.5183865867      0.0000000000      1.8891781049
    C        1.3172146985     -2.2531204594     -0.7737861566
    H        1.1322105108     -3.8348590209      0.5099803895
    H        1.1680586154     -2.6812379131     -2.7700739383
    C        3.9569164314      1.7018306568     -1.8913890844
    H        3.0053826354      1.5478557721     -3.7088898774
    H        3.8622600497      3.6616656215     -1.2676471822
    H        5.9395792877      1.1934754318     -2.1292489119
    """, "Bohr"),  # (R)-2-Methyloxirane
    #
    "formaldehyde": ("""
    C 2.0092420208996 3.8300915804899 0.8199294419789
    O 2.1078857690998 2.0406638776593 2.1812021228452
    H 2.0682421748693 5.7438044586615 1.5798996515014
    H 1.8588483602149 3.6361694243085 -1.2192956060942
    """, "Bohr"),
    #
    "nh3": ("""
    N     -0.0000000001    -0.1040380466      0.0000000000
    H     -0.9015844116     0.4818470201     -1.5615900098
    H     -0.9015844116     0.4818470201      1.5615900098
    H      1.8031688251     0.4818470204      0.0000000000
    """, "Bohr")
}


def _init_test_cases() -> tuple[TestCase, ...]:
    test_cases: list[TestCase] = []
    # some shared data
    restricted_kinds = Kinds(pp=("singlet", "triplet"))
    unrestricted_kinds = Kinds(pp=("any",))
    spin_flip_kinds = Kinds(pp=("spin_flip",))
    # CH2NH2
    ref_cases = ("gen", "cvs")
    xyz, unit = _xyz["ch2nh2"]
    test_cases.append(TestCase(
        name="ch2nh2", xyz=xyz, unit=unit, charge=0, multiplicity=2,
        basis="sto-3g", core_orbitals=2, cases=ref_cases, only_full_mode=False,
        restricted=False, kinds=unrestricted_kinds
    ))
    test_cases.append(TestCase(
        name="ch2nh2", xyz=xyz, unit=unit, charge=0, multiplicity=2,
        basis="cc-pvdz", only_full_mode=True, restricted=False,
        kinds=unrestricted_kinds
    ))
    # CN
    xyz, unit = _xyz["cn"]
    ref_cases = ("gen", "cvs", "fc", "fv", "fv-cvs", "fc-fv", "fc-cvs", "fc-fv-cvs")
    test_cases.append(TestCase(
        name="cn", xyz=xyz, unit=unit, charge=0, multiplicity=2,
        basis="sto-3g", core_orbitals=1, frozen_core=1, frozen_virtual=1,
        cases=ref_cases, only_full_mode=False, restricted=False,
        kinds=unrestricted_kinds
    ))
    ref_cases = ("gen", "cvs")
    test_cases.append(TestCase(
        name="cn", xyz=xyz, unit=unit, charge=0, multiplicity=2,
        basis="cc-pvdz", core_orbitals=1, frozen_core=1, frozen_virtual=3,
        cases=ref_cases, only_full_mode=True, restricted=False,
        kinds=unrestricted_kinds
    ))
    # H2O
    xyz, unit = _xyz["h2o"]
    ref_cases = ("gen", "cvs", "fc", "fv", "fv-cvs", "fc-fv", "fc-cvs", "fc-fv-cvs")
    test_cases.append(TestCase(
        name="h2o", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="sto-3g", core_orbitals=1, frozen_core=1, frozen_virtual=1,
        cases=ref_cases, only_full_mode=False, restricted=True,
        kinds=restricted_kinds
    ))
    ref_cases = ("gen", "cvs")
    test_cases.append(TestCase(
        name="h2o", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="def2-tzvp", core_orbitals=1, cases=ref_cases, restricted=True,
        only_full_mode=True, kinds=restricted_kinds
    ))
    test_cases.append(TestCase(
        name="h2o", xyz=xyz, unit=unit, charge=0, multiplicity=1, restricted=True,
        basis="cc-pvdz", only_full_mode=True
    ))
    # HF
    ref_cases = ("gen", "fc", "fv", "fc-fv")
    xyz, unit = _xyz["hf"]
    test_cases.append(TestCase(
        name="hf", xyz=xyz, unit=unit, charge=0, multiplicity=3,
        basis="6-31g", frozen_core=1, frozen_virtual=3, cases=ref_cases,
        only_full_mode=False, restricted=False, kinds=spin_flip_kinds
    ))
    # (R)-2-Methyloxirane
    ref_cases = ("gen", "cvs")
    xyz, unit = _xyz["r2methyloxirane"]
    test_cases.append(TestCase(
        name="r2methyloxirane", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="sto-3g", core_orbitals=1, cases=ref_cases, only_full_mode=False,
        restricted=True
    ))
    test_cases.append(TestCase(
        name="r2methyloxirane", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="cc-pvdz", only_full_mode=True, restricted=True
    ))
    # Formaledhyde
    xyz, unit = _xyz["formaldehyde"]
    pe_potfile = Path(__file__).resolve().parent
    pe_potfile = pe_potfile / "generators" / "potentials" / "fa_6w.pot"
    test_cases.append(TestCase(
        name="formaldehyde", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="sto-3g", pe_potfile=str(pe_potfile), only_full_mode=False,
        restricted=True, kinds=Kinds(pp=("singlet",))
    ))
    test_cases.append(TestCase(
        name="formaldehyde", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="cc-pvdz", pe_potfile=str(pe_potfile), only_full_mode=True,
        restricted=True, kinds=Kinds(pp=("singlet",))
    ))
    # NH3
    xyz, unit = _xyz["nh3"]
    test_cases.append(TestCase(
        name="nh3", xyz=xyz, unit=unit, charge=0, multiplicity=1, basis="3-21g",
        only_full_mode=False, restricted=True
    ))
    # validate the testcases: cvs -> core_orbitals needs to be set
    # and similar for the other cases
    for case in test_cases:
        case.validate()
    # Furthermore ensure that all test cases have a unique file_name
    assert all(
        v == 1 for v in Counter([case.file_name for case in test_cases]).values()
    )
    return tuple(test_cases)


available = _init_test_cases()


def get(n_expected_cases: int = None, **kwargs: str) -> list[TestCase]:
    """
    Filter test cases according to the fields of the test cases, e.g.,
    name="h2o"
    to obtain all test cases for the H2O molecule or
    name="h2o", basis="sto-3g"
    to obtain the H2O sto-3g test case.
    If n_expected_cases is set, the returned list will be checked to contain
    the desired amount of test cases. An error is raised if this is not the case.
    """
    ret = []
    for case in available:
        if all(getattr(case, field) == val
                for field, val in kwargs.items()):
            ret.append(case)
    if not ret:
        raise ValueError(f"Could not find a test case with fields {kwargs}.")
    elif n_expected_cases is not None and n_expected_cases != len(ret):
        raise ValueError(f"Could not find {n_expected_cases} cases with fields "
                         f"{kwargs}. Found {len(ret)} cases. Maybe a case was "
                         "added or removed.")
    return ret


def get_by_filename(*args: str) -> list[TestCase]:
    """
    Obtain test cases according to their filename. At the moment the
    file name for a test case is build from the molecule name and the basis set,
    e.g., "h2o_sto3g". The function assumes to find 1 test case per file name.
    """
    ret = []
    for case in available:
        if case.file_name in args:
            ret.append(case)
    if len(args) != len(ret):
        raise ValueError(f"Found {len(ret)} test cases for {len(args)} filenames. "
                         "Expected to find 1 test case per file name.")
    return ret
