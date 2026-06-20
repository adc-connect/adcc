#!/usr/bin/env python3
import pytest

from adcc.AdcMethod import AdcMethod, AdcType
from adcc.ChargedExcitations import DetachedStates, AttachedStates

from .testdata_cache import testdata_cache


cases_ip_ea = [
    ("h2o_sto3g", "ip-adc2", "gen", "doublet"),
    ("h2o_sto3g", "ea-adc2", "gen", "doublet"),
]


@pytest.mark.parametrize("system,method,case,kind", cases_ip_ea)
def test_ip_ea_basic_interface(system, method, case, kind):
    adc_type = AdcMethod(method).adc_type

    if adc_type is AdcType.IP:
        state = testdata_cache.adcc_states(
            system=system,
            method=method,
            case=case,
            kind=kind,
            is_alpha=True,
        )
        assert isinstance(state, DetachedStates)
    elif adc_type is AdcType.EA:
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
    if adc_type is AdcType.IP:
        state.spin_change = -0.5
    elif adc_type is AdcType.EA:
        state.spin_change = 0.5

    desc = state.describe()

    assert state.kind in desc.lower()

    if adc_type is AdcType.IP:
        assert "ionization" in desc.lower()
        assert "detachment" in desc.lower()
    elif adc_type is AdcType.EA:
        assert "affinity" in desc.lower()
        assert "attachment" in desc.lower()
