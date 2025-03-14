#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import pytest
from numpy.testing import assert_allclose

from adcc.OneParticleOperator import OneParticleOperator
from adcc.ExcitedStates import EnergyCorrection

from .testdata_cache import testdata_cache


class TestExcitedStates:
    def test_excitation_view(self):
        state = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", case="gen", kind="singlet"
        )

        n_ref = len(state.excitation_vector)
        assert n_ref == state.size

        for i, exci in enumerate(state.excitations):
            for key in dir(exci):
                if key.startswith("_"):
                    continue
                blacklist = ["__", "index", "_ao", "excitation_vector",
                             "method", "parent_state"]
                if any(b in key for b in blacklist):
                    continue
                try:
                    ref = getattr(state, key)[i]
                    res = getattr(exci, key)
                except NotImplementedError:
                    continue
                except TypeError:
                    try:
                        ref = getattr(state, key)("origin")[i]
                        res = getattr(exci, key)("origin")
                    except NotImplementedError:
                        # electric dipole velocity integrals are not implemented in
                        # the reference backend
                        continue
                if isinstance(ref, OneParticleOperator):
                    assert ref.blocks == res.blocks
                    for b in ref.blocks:
                        assert_allclose(ref[b].to_ndarray(),
                                        res[b].to_ndarray())
                else:
                    assert_allclose(ref, res)

    def test_custom_excitation_energy_corrections(self):
        state = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", kind="singlet", case="gen"
        )

        cc1 = EnergyCorrection("custom_correction1",
                               lambda exci: exci.excitation_energy ** 2)
        cc2 = EnergyCorrection("custom_correction2",
                               lambda exci: 2.0)
        cc3 = EnergyCorrection("custom_correction3",
                               lambda exci: -42.0)
        state_corrected = state + [cc1, cc2]
        for i in range(state.size):
            assert hasattr(state_corrected, "custom_correction1")
            assert hasattr(state_corrected, "custom_correction2")
            assert_allclose(state.excitation_energy[i],
                            state_corrected.excitation_energy_uncorrected[i])
            corr = state.excitation_energy[i] ** 2 + 2.0
            assert_allclose(state.excitation_energy[i] + corr,
                            state_corrected.excitation_energy[i])

        with pytest.raises(ValueError):
            state_corrected += cc2
        with pytest.raises(TypeError):
            state_corrected += 1

        state_corrected2 = state_corrected + cc3
        for i in range(state.size):
            assert hasattr(state_corrected2, "custom_correction1")
            assert hasattr(state_corrected2, "custom_correction2")
            assert hasattr(state_corrected2, "custom_correction3")
            assert_allclose(state.excitation_energy[i],
                            state_corrected2.excitation_energy_uncorrected[i])
            corr = state.excitation_energy[i] ** 2 + 2.0 - 42.0
            assert_allclose(state.excitation_energy[i] + corr,
                            state_corrected2.excitation_energy[i])
        state_corrected2.describe()

    def test_dataframe_export(self, gauge_origin="origin"):
        state = testdata_cache.adcc_states(
            system="h2o_sto3g", method="adc2", kind="singlet", case="gen"
        )

        df = state.to_dataframe()
        df.drop(["excitation", "kind"], inplace=True, axis=1)
        components = ["x", "y", "z"]
        assert len(df.columns)
        for key in df.columns:
            if hasattr(state, key):
                try:
                    assert_allclose(df[key], getattr(state, key))
                except TypeError:
                    assert_allclose(df[key], getattr(state, key)(gauge_origin))
            elif hasattr(state, key[:-2]):
                newkey = key[:-2]
                i = components.index(key[-1])
                try:
                    assert_allclose(df[key], getattr(state, newkey)[:, i])
                except TypeError:
                    assert_allclose(df[key],
                                    getattr(state, newkey)(gauge_origin)[:, i])
            elif hasattr(state, key[:-3]):
                newkey = key[:-3]
                i = components.index(key[-1])
                j = components.index(key[-2])
                try:
                    assert_allclose(df[key], getattr(state, newkey)[:, i, j])
                except TypeError:
                    assert_allclose(df[key],
                                    getattr(state, newkey)(gauge_origin)[:, i, j])
            else:
                raise KeyError(f"Key {key} not found in ExcitedStates object.")
