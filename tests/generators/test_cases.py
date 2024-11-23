from dataclasses import dataclass, asdict
from pathlib import Path


_basis_to_fname = {
    "sto-3g": "sto3g",
    "cc-pvdz": "ccpvdz",
    "def2-tzvp": "def2tzvp",
    "6-311+g**": "6311g",
    "6-31g": "631g"
}

_molname_to_fname = {
    "hf": "hf3",
    "r2methyloxirane": "methox"
}


@dataclass(frozen=True, slots=True)
class TestCase:
    name: str
    xyz: str
    unit: str
    charge: int
    multiplicity: int  # = 2S+1
    basis: str
    pe_pot_file: str = None

    @property
    def file_name(self) -> str:
        """Builds a file name based on name and basis."""
        name = self.name
        if name in _molname_to_fname:
            name = _molname_to_fname[self.name]
        return f"{name}_{_basis_to_fname[self.basis]}"

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


_xyz = {
    "h2o": ("""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    """, "Bohr"),
    #
    "h2s": ("""
    S  -0.38539679062   0 -0.27282082253
    H  -0.0074283962687 0  2.2149138578
    H   2.0860198029    0 -0.74589639249
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
    """, "Bohr")
}


def _init_test_cases() -> tuple[TestCase]:
    cases = []
    # CH2NH2
    xyz, unit = _xyz["ch2nh2"]
    cases.append(TestCase(
        name="ch2nh2", xyz=xyz, unit=unit, charge=0, multiplicity=2,
        basis="sto-3g"
    ))
    # CN
    xyz, unit = _xyz["cn"]
    cases.append(TestCase(
        name="cn", xyz=xyz, unit=unit, charge=0, multiplicity=2,
        basis="cc-pvdz"
    ))
    cases.append(TestCase(
        name="cn", xyz=xyz, unit=unit, charge=0, multiplicity=2,
        basis="sto-3g"
    ))
    # H2O
    xyz, unit = _xyz["h2o"]
    cases.append(TestCase(
        name="h2o", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="def2-tzvp"
    ))
    cases.append(TestCase(
        name="h2o", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="sto-3g"
    ))
    # H2S
    xyz, unit = _xyz["h2s"]
    cases.append(TestCase(
        name="h2s", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="6-311+g**"
    ))
    cases.append(TestCase(
        name="h2s", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="sto-3g"
    ))
    # HF
    xyz, unit = _xyz["hf"]
    cases.append(TestCase(
        name="hf", xyz=xyz, unit=unit, charge=0, multiplicity=3,
        basis="6-31g"
    ))
    # (R)-2-Methyloxirane
    xyz, unit = _xyz["r2methyloxirane"]
    cases.append(TestCase(
        name="r2methyloxirane", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="sto-3g"
    ))
    # Formaledhyde
    xyz, unit = _xyz["formaldehyde"]
    pe_pot_file = Path(__file__).parent / "potentials/fa_6w.pot"
    cases.append(TestCase(
        name="formaldehyde", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="sto-3g", pe_pot_file=str(pe_pot_file)
    ))
    cases.append(TestCase(
        name="formaldehyde", xyz=xyz, unit=unit, charge=0, multiplicity=1,
        basis="cc-pvdz", pe_pot_file=str(pe_pot_file)
    ))
    return tuple(cases)


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
