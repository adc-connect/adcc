#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
from pytest import approx

from adcc import block as b
from adcc import LazyMp

from .testdata_cache import testdata_cache
from . import testcases


test_cases = testcases.get_by_filename(
    "h2o_sto3g", "cn_sto3g", "h2o_def2tzvp", "cn_ccpvdz"
)
cases = [(case.file_name, c) for case in test_cases for c in ["gen", "cvs"]]
generators = ["adcman", "adcc"]


@pytest.fixture(scope="class")
def instances():
    mp: dict[tuple, LazyMp] = {}
    for system, case in cases:
        mp[(system, case)] = LazyMp(testdata_cache.refstate(system, case=case))
    return mp


class TestLazyMp:
    def test_exceptions(self, instances: dict[tuple, LazyMp]):
        mp = instances[("h2o_sto3g", "gen")]
        assert mp.energy_correction(0) == 0.0
        assert mp.energy_correction(1) == 0.0
        with pytest.raises(AssertionError):
            mp.t2(b.vvoo)
        with pytest.raises(NotImplementedError):
            mp.t2eri(b.oooo, b.oo)
        with pytest.raises(NotImplementedError):
            mp.td2(b.ccvv)
        with pytest.raises(NotImplementedError):
            mp.energy_correction(4)

    #
    # Generic
    #
    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_mp2_energy(self, system: str, case: str, generator: str,
                        instances: dict[tuple, LazyMp]):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert (
            instances[(system, case)].energy_correction(2)
            == approx(refmp["mp2"]["energy"])
        )
        if "cvs" not in case:
            return
        # CVS MP(2) energy should be equal to the non-CVS one
        non_cvs_case = "-".join(c for c in case.split("-") if c != "cvs")
        if not non_cvs_case:
            non_cvs_case = "gen"

        refmp = testdata_cache._load_data(
            system=system, method="mp", case=non_cvs_case, source=generator
        )
        assert (
            instances[(system, case)].energy_correction(2)
            == approx(refmp["mp2"]["energy"])
        )

    # MP3 energy not implemented for CVS
    @pytest.mark.parametrize("system,case", [(s, c) for s, c in cases
                                             if "cvs" not in c])
    @pytest.mark.parametrize("generator", generators)
    def test_mp3_energy(self, system: str, case: str, generator: str,
                        instances: dict[tuple, LazyMp]):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert (
            instances[(system, case)].energy_correction(3)
            == approx(refmp["mp3"]["energy"])
        )

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_df(self, system: str, case: str, generator: str,
                instances: dict[tuple, LazyMp]):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert_allclose(instances[(system, case)].df("o1v1").to_ndarray(),
                        refmp["mp1"]["df_o1v1"], atol=1e-12)
        if "cvs" in case:
            assert_allclose(instances[(system, case)].df("o2v1").to_ndarray(),
                            refmp["mp1"]["df_o2v1"], atol=1e-12)

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_t2(self, system: str, case: str, generator: str,
                instances: dict[tuple, LazyMp]):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        blocks = ["o1o1v1v1"]
        if "cvs" in case:
            blocks.extend(["o1o2v1v1", "o2o2v1v1"])
        for label in blocks:
            assert_allclose(instances[(system, case)].t2(label).to_ndarray(),
                            refmp["mp1"][f"t_{label}"], atol=1e-12)
            assert f"t2/{label}" in instances[(system, case)].timer.tasks

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_td2(self, system: str, case: str, generator: str,
                 instances: dict[tuple, LazyMp]):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert_allclose(instances[(system, case)].td2("o1o1v1v1").to_ndarray(),
                        refmp["mp2"]["td_o1o1v1v1"], atol=1e-12)
        assert "td2/o1o1v1v1" in instances[(system, case)].timer.tasks

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_mp2_diffdm_mo(self, system: str, case: str, generator: str,
                           instances: dict[tuple, LazyMp]):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        mp2diff = instances[(system, case)].mp2_diffdm

        assert mp2diff.is_symmetric
        blocks = ["o1o1", "o1v1", "v1v1"]
        if "cvs" in case:
            blocks.extend(["o2o1", "o2o2", "o2v1"])
        for label in blocks:
            assert_allclose(mp2diff[label].to_ndarray(),
                            refmp["mp2"]["dm_" + label], atol=1e-12)
        assert "mp2_diffdm" in instances[((system, case))].timer.tasks

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_mp2_diffdm_ao(self, system: str, case: str, generator: str,
                           instances: dict[tuple, LazyMp]):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        mp2diff = instances[(system, case)].mp2_diffdm
        reference_state = instances[(system, case)].reference_state

        dm_α, dm_β = mp2diff.to_ao_basis(reference_state)
        assert_allclose(dm_α.to_ndarray(), refmp["mp2"]["dm_bb_a"], atol=1e-12)
        assert_allclose(dm_β.to_ndarray(), refmp["mp2"]["dm_bb_b"], atol=1e-12)

        if "cvs" not in case:
            return
        # CVS MP(2) densities in AOs should be equal to the non-CVS one
        non_cvs_case = "-".join(c for c in case.split("-") if c != "cvs")
        if not non_cvs_case:
            non_cvs_case = "gen"
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=non_cvs_case, source=generator
        )
        assert_allclose(dm_α.to_ndarray(), refmp["mp2"]["dm_bb_a"], atol=1e-12)
        assert_allclose(dm_β.to_ndarray(), refmp["mp2"]["dm_bb_b"], atol=1e-12)

    #
    # Cache
    #
    def test_cache(self, instances: dict[tuple, LazyMp]):
        # call some stuff twice
        instances[("h2o_sto3g", "gen")].energy_correction(2)
        instances[("h2o_sto3g", "gen")].energy_correction(2)
        instances[("h2o_sto3g", "gen")].energy_correction(3)
        instances[("h2o_sto3g", "gen")].energy_correction(3)
        timer = instances[("h2o_sto3g", "gen")].timer
        assert "energy_correction/2" in timer.tasks
        assert "energy_correction/3" in timer.tasks
        assert len(timer.intervals("t2/o1o1v1v1")) == 1
        assert len(timer.intervals("td2/o1o1v1v1")) == 1
        assert len(timer.intervals("energy_correction/2")) == 1
        assert len(timer.intervals("energy_correction/3")) == 1
