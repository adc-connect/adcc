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
from adcc import OperatorSymmetry

from .testdata_cache import testdata_cache
from . import testcases


test_cases = testcases.get_by_filename(
    "h2o_sto3g", "cn_sto3g", "h2o_def2tzvp", "cn_ccpvdz"
)
small_cases = [
    (case.file_name, c) for case in test_cases if not case.only_full_mode
    for c in ["gen", "cvs"]
]
large_cases = [
    (case.file_name, c) for case in test_cases if case.only_full_mode
    for c in ["gen", "cvs"]
]
assert not any(c in small_cases for c in large_cases)
cases = small_cases + large_cases
generators = ["adcman", "adcc"]
# we don't have tt2 reference data from adcman for cvs, since there
# is nothing implemented currently for cvs that requires the tt2 amplitudes.
tt2_cases = [(system, case, "adcc") for system, case in small_cases]
tt2_cases.extend(
    (system, case, "adcman") for system, case in small_cases if "cvs" not in case
)


# Helper class to lazily cache the LazyMp instances so we can utilize their cache
# for the LazyMp tests below (avoid recomputing LazyMp.t2 for energy, td2, ...)
# Since the instance cache is lazy, we don't have to know whether we are running
# in fast or full mode!
# By using a scoped fixture, the instances (and their cache) will be dropped
# after the test.
class LazyMpCache:
    def __init__(self):
        self.instances = {}

    def get(self, system: str, case: str) -> LazyMp:
        instance = self.instances.get((system, case), None)
        if instance is None:  # init and cache the LazyMp instance
            instance = LazyMp(testdata_cache.refstate(system, case=case))
            self.instances[(system, case)] = instance
        return instance


@pytest.fixture(scope="class")
def instances():
    return LazyMpCache()


class TestLazyMp:
    def test_exceptions(self, instances: LazyMpCache):
        mp = instances.get("h2o_sto3g", "gen")
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
                        instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert (
            instances.get(system, case).energy_correction(2)
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
            instances.get(system, case).energy_correction(2)
            == approx(refmp["mp2"]["energy"])
        )

    # MP3 energy not implemented for CVS
    @pytest.mark.parametrize("system,case", [(s, c) for s, c in cases
                                             if "cvs" not in c])
    @pytest.mark.parametrize("generator", generators)
    def test_mp3_energy(self, system: str, case: str, generator: str,
                        instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert (
            instances.get(system, case).energy_correction(3)
            == approx(refmp["mp3"]["energy"])
        )

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_df(self, system: str, case: str, generator: str,
                instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert_allclose(instances.get(system, case).df("o1v1").to_ndarray(),
                        refmp["mp1"]["df_o1v1"], atol=1e-12)
        if "cvs" in case:
            assert_allclose(instances.get(system, case).df("o2v1").to_ndarray(),
                            refmp["mp1"]["df_o2v1"], atol=1e-12)

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_t2(self, system: str, case: str, generator: str,
                instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        blocks = ["o1o1v1v1"]
        if "cvs" in case:
            blocks.extend(["o1o2v1v1", "o2o2v1v1"])
        for label in blocks:
            assert_allclose(instances.get(system, case).t2(label).to_ndarray(),
                            refmp["mp1"][f"t_{label}"], atol=1e-12)
            assert f"t2/{label}" in instances.get(system, case).timer.tasks

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_td2(self, system: str, case: str, generator: str,
                 instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert_allclose(instances.get(system, case).td2("o1o1v1v1").to_ndarray(),
                        refmp["mp2"]["td_o1o1v1v1"], atol=1e-12)
        assert "td2/o1o1v1v1" in instances.get(system, case).timer.tasks

    @pytest.mark.parametrize("system,case,generator", tt2_cases)
    def test_tt2(self, system: str, case: str, generator: str,
                 instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        assert_allclose(
            instances.get(system, case).tt2("o1o1o1v1v1v1").to_ndarray(),
            refmp["mp2"]["tt_o1o1o1v1v1v1"],
            atol=1e-12
        )
        assert "tt2/o1o1o1v1v1v1" in instances.get(system, case).timer.tasks

    def test_triples_symmetry(self, instances: LazyMpCache):
        import libadcc
        # Ensure that the triples have the correct symmetry that matches
        # the symmetry of make_symmetry_triples
        mp = instances.get("h2o_sto3g", "gen")
        tt2 = mp.tt2("o1o1o1v1v1v1")
        sym = libadcc.make_symmetry_triples(mp.mospaces, "o1o1o1v1v1v1")
        reimport = libadcc.Tensor(sym)
        reimport.set_from_ndarray(tt2.to_ndarray(), 1e-14)
        # and again for unrestricted (no spin block mapping)
        mp = instances.get("cn_sto3g", "gen")
        tt2 = mp.tt2("o1o1o1v1v1v1")
        sym = libadcc.make_symmetry_triples(mp.mospaces, "o1o1o1v1v1v1")
        reimport = libadcc.Tensor(sym)
        reimport.set_from_ndarray(tt2.to_ndarray(), 1e-14)

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_mp2_diffdm_mo(self, system: str, case: str, generator: str,
                           instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        mp2diff = instances.get(system, case).mp2_diffdm

        assert mp2diff.symmetry is OperatorSymmetry.HERMITIAN
        blocks = ["o1o1", "o1v1", "v1v1"]
        if "cvs" in case:
            blocks.extend(["o2o1", "o2o2", "o2v1"])
        for label in blocks:
            assert_allclose(mp2diff[label].to_ndarray(),
                            refmp["mp2"]["dm_" + label], atol=1e-12)
        assert "mp2_diffdm" in instances.get(system, case).timer.tasks

    @pytest.mark.parametrize("system,case", cases)
    @pytest.mark.parametrize("generator", generators)
    def test_mp2_diffdm_ao(self, system: str, case: str, generator: str,
                           instances: LazyMpCache):
        refmp = testdata_cache._load_data(
            system=system, method="mp", case=case, source=generator
        )
        mp2diff = instances.get(system, case).mp2_diffdm
        reference_state = instances.get(system, case).reference_state

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
    def test_cache(self, instances: LazyMpCache):
        # call some stuff twice
        instances.get("h2o_sto3g", "gen").energy_correction(2)
        instances.get("h2o_sto3g", "gen").energy_correction(2)
        instances.get("h2o_sto3g", "gen").energy_correction(3)
        instances.get("h2o_sto3g", "gen").energy_correction(3)
        timer = instances.get("h2o_sto3g", "gen").timer
        assert "energy_correction/2" in timer.tasks
        assert "energy_correction/3" in timer.tasks
        assert len(timer.intervals("t2/o1o1v1v1")) == 1
        assert len(timer.intervals("td2/o1o1v1v1")) == 1
        assert len(timer.intervals("energy_correction/2")) == 1
        assert len(timer.intervals("energy_correction/3")) == 1
