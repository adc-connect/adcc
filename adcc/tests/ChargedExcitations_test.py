#!/usr/bin/env python3
import pytest
from numpy.testing import assert_allclose

from adcc.AdcMethod import AdcMethod
from adcc.ChargedExcitations import DetachedStates, AttachedStates

from .testdata_cache import testdata_cache


# ---------------------------------------------------------------------
# Shared parametrization (reuse your existing case table if available)
# ---------------------------------------------------------------------

cases_ip_ea = [
    ("h2o_sto3g", "ip-adc2", "gen", "doublet"),
    ("h2o_sto3g", "ea-adc2", "gen", "doublet"),
]


# ---------------------------------------------------------------------
# 1. Basic construction + size consistency
# ---------------------------------------------------------------------

@pytest.mark.parametrize("system,method,case,kind", cases_ip_ea)
def test_ip_ea_basic_interface(system, method, case, kind):
    adc_type = AdcMethod(method).adc_type

    if adc_type == "ip":
        state = testdata_cache.adcc_states(
            system=system,
            method=method,
            case=case,
            kind=kind,
            is_alpha=True,
        )
        assert isinstance(state, DetachedStates)
    elif adc_type == "ea":
        state = testdata_cache.adcc_states(
            system=system,
            method=method,
            case=case,
            kind=kind,
            is_alpha=True,
        )
        assert isinstance(state, AttachedStates)
    else:
        raise AssertionError("Unexpected ADC type")

    # size matches number of excitation vectors
    assert state.size == len(state.excitation_vector)
    assert state.size == len(state.excitation_energy)


# ---------------------------------------------------------------------
# 2. Pole strength is well-defined and matches state count
# ---------------------------------------------------------------------

@pytest.mark.parametrize("system,method,case,kind", cases_ip_ea)
def test_ip_ea_pole_strength(system, method, case, kind):
    state = testdata_cache.adcc_states(
        system=system,
        method=method,
        case=case,
        kind=kind,
        is_alpha=True,
    )

    ps = state.pole_strength

    assert len(ps) == state.size
    assert (ps >= 0).all()  # positive pole_strenghts


# ---------------------------------------------------------------------
# 3. QC variable export consistency
# ---------------------------------------------------------------------

@pytest.mark.parametrize("system,method,case,kind", cases_ip_ea)
def test_ip_ea_qcvars_export(system, method, case, kind):
    state = testdata_cache.adcc_states(
        system=system,
        method=method,
        case=case,
        kind=kind,
        is_alpha=True,
    )

    qcvars = state.to_qcvars(properties=False)

    adc_type = AdcMethod(method).adc_type

    if adc_type == "ip":
        assert any("IONIZATION POTENTIALS" in key for key in qcvars)
    elif adc_type == "ea":
        assert any("ELECTRON AFFINITIES" in key for key in qcvars)

    assert any("NUMBER" in key for key in qcvars)


# ---------------------------------------------------------------------
# 4. describe() runs without error and contains expected wording
# ---------------------------------------------------------------------

@pytest.mark.parametrize("system,method,case,kind", cases_ip_ea)
def test_ip_ea_describe(system, method, case, kind):
    state = testdata_cache.adcc_states(
        system=system,
        method=method,
        case=case,
        kind=kind,
        is_alpha=True,
    )

    adc_type = AdcMethod(method).adc_type
    # Only test restricted alpha
    if adc_type == "ip":
        state.spin_change = -0.5
    elif adc_type == "ea":
        state.spin_change = 0.5

    desc = state.describe()

    assert state.kind in desc.lower()

    if adc_type == "ip":
        assert "ionization" in desc.lower()
        assert "detachment" in desc.lower()
    elif adc_type == "ea":
        assert "affinity" in desc.lower()
        assert "attachment" in desc.lower()


# ---------------------------------------------------------------------
# 5. IP/EA hermiticity-style sanity check
# ---------------------------------------------------------------------
#
# Hermiticity of the ADC matrix itself belongs in:
#     tests/test_adc_matrix.py
#
# NOT here.
#
# However, what *is* appropriate here:
#   Energies must be real-valued.
#

@pytest.mark.parametrize("system,method,case,kind", cases_ip_ea)
def test_ip_ea_energies_real(system, method, case, kind):
    state = testdata_cache.adcc_states(
        system=system,
        method=method,
        case=case,
        kind=kind,
        is_alpha=True,
    )

    assert_allclose(state.excitation_energy.imag, 0.0)
