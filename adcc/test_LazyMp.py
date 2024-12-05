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
import unittest

from .misc import expand_test_templates
from . import block as b
from numpy.testing import assert_allclose

from adcc import LazyMp
from adcc.testdata.cache import cache

import pytest

from pytest import approx

# All test cases to deal with here
testcases = ["h2o_sto3g", "cn_sto3g"]
if cache.mode_full:
    testcases += ["h2o_def2tzvp", "cn_ccpvdz"]


@expand_test_templates(testcases)
class TestLazyMp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mp = {}
        cls.mp_cvs = {}
        for case in testcases:
            cls.mp[case] = LazyMp(cache.refstate[case])
            cls.mp_cvs[case] = LazyMp(cache.refstate_cvs[case])

    def test_exceptions(self):
        mp = self.mp["h2o_sto3g"]
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
    def template_mp2_energy(self, case):
        refmp = cache.reference_data[case]["mp"]
        assert self.mp[case].energy_correction(2) == \
            approx(refmp["mp2"]["energy"])

    def template_mp3_energy(self, case):
        refmp = cache.reference_data[case]["mp"]
        assert self.mp[case].energy_correction(3) == \
            approx(refmp["mp3"]["energy"])

    def template_df(self, case):
        refmp = cache.reference_data[case]["mp"]
        assert_allclose(self.mp[case].df("o1v1").to_ndarray(),
                        refmp["mp1"]["df_o1v1"], atol=1e-12)

    def template_t2(self, case):
        refmp = cache.reference_data[case]["mp"]
        assert_allclose(self.mp[case].t2("o1o1v1v1").to_ndarray(),
                        refmp["mp1"]["t_o1o1v1v1"], atol=1e-12)
        assert "t2/o1o1v1v1" in self.mp[case].timer.tasks

    def template_td2(self, case):
        refmp = cache.reference_data[case]["mp"]
        assert_allclose(self.mp[case].td2("o1o1v1v1").to_ndarray(),
                        refmp["mp2"]["td_o1o1v1v1"], atol=1e-12)
        assert "td2/o1o1v1v1" in self.mp[case].timer.tasks

    def template_mp2_density_mo(self, case):
        refmp = cache.reference_data[case]["mp"]
        mp2diff = self.mp[case].mp2_diffdm

        assert mp2diff.is_symmetric
        for label in ["o1o1", "o1v1", "v1v1"]:
            assert_allclose(mp2diff[label].to_ndarray(),
                            refmp["mp2"]["dm_" + label], atol=1e-12)
        assert "mp2_diffdm" in self.mp[case].timer.tasks

    def template_mp2_density_ao(self, case):
        refmp = cache.reference_data[case]["mp"]
        mp2diff = self.mp[case].mp2_diffdm
        reference_state = self.mp[case].reference_state

        dm_α, dm_β = mp2diff.to_ao_basis(reference_state)
        assert_allclose(dm_α.to_ndarray(), refmp["mp2"]["dm_bb_a"], atol=1e-12)
        assert_allclose(dm_β.to_ndarray(), refmp["mp2"]["dm_bb_b"], atol=1e-12)

    #
    # CVS
    #
    def template_cvs_mp2_energy(self, case):
        refmpcvs = cache.reference_data[case]["cvs-mp"]
        assert self.mp_cvs[case].energy_correction(2) == \
            approx(refmpcvs["mp2"]["energy"])

        # CVS MP(2) energy should be equal to the non-CVS one
        refmp = cache.reference_data[case]["mp"]
        assert self.mp_cvs[case].energy_correction(2) == \
            approx(refmp["mp2"]["energy"])

    def template_cvs_mp3_energy(self, case):
        # Note: CVS-MP(3) energies in adcman are wrong, so they
        #       are not used for testing here.
        # CVS MP(3) energy should be equal to the non-CVS one
        pytest.xfail("MP3 energies not yet available for CVS")
        refmp = cache.reference_data[case]["mp"]
        assert self.mp_cvs[case].energy_correction(3) == \
            approx(refmp["mp3"]["energy"])

    def template_cvs_df(self, case):
        refmpcvs = cache.reference_data[case]["cvs-mp"]
        assert_allclose(self.mp_cvs[case].df("o1v1").to_ndarray(),
                        refmpcvs["mp1"]["df_o1v1"], atol=1e-12)
        assert_allclose(self.mp_cvs[case].df("o2v1").to_ndarray(),
                        refmpcvs["mp1"]["df_o2v1"], atol=1e-12)

    def template_cvs_t2(self, case):
        refmpcvs = cache.reference_data[case]["cvs-mp"]
        for label in ["o1o1v1v1", "o1o2v1v1", "o2o2v1v1"]:
            assert_allclose(self.mp_cvs[case].t2(label).to_ndarray(),
                            refmpcvs["mp1"]["t_" + label], atol=1e-12)

    def template_cvs_td2(self, case):
        refmpcvs = cache.reference_data[case]["cvs-mp"]
        for label in ["o1o1v1v1", "o1o2v1v1", "o2o2v1v1"]:
            if "td_" + label in refmpcvs["mp2"]:
                assert_allclose(self.mp_cvs[case].td2(label).to_ndarray(),
                                refmpcvs["mp2"]["td_" + label], atol=1e-12)

    def template_cvs_mp2_density_mo(self, case):
        refmpcvs = cache.reference_data[case]["cvs-mp"]
        mp2diff = self.mp_cvs[case].mp2_diffdm

        assert mp2diff.is_symmetric
        for label in ["o1o1", "o2o1", "o1v1", "o2o2", "o2v1", "v1v1"]:
            assert_allclose(mp2diff[label].to_ndarray(),
                            refmpcvs["mp2"]["dm_" + label], atol=1e-12)

    def template_cvs_mp2_density_ao(self, case):
        refmpcvs = cache.reference_data[case]["cvs-mp"]
        refmp = cache.reference_data[case]["mp"]
        mp2diff = self.mp_cvs[case].mp2_diffdm
        reference_state = self.mp_cvs[case].reference_state

        dm_α, dm_β = mp2diff.to_ao_basis(reference_state)
        assert_allclose(dm_α.to_ndarray(),
                        refmpcvs["mp2"]["dm_bb_a"], atol=1e-12)
        assert_allclose(dm_β.to_ndarray(),
                        refmpcvs["mp2"]["dm_bb_b"], atol=1e-12)

        # CVS MP(2) densities in AOs should be equal to the non-CVS one
        assert_allclose(dm_α.to_ndarray(), refmp["mp2"]["dm_bb_a"], atol=1e-12)
        assert_allclose(dm_β.to_ndarray(), refmp["mp2"]["dm_bb_b"], atol=1e-12)

    #
    # Cache
    #
    def template_cache(self, case):
        # call some stuff twice
        self.mp[case].energy_correction(2)
        self.mp[case].energy_correction(2)
        self.mp[case].energy_correction(3)
        self.mp[case].energy_correction(3)
        timer = self.mp[case].timer
        assert "energy_correction/2" in timer.tasks
        assert "energy_correction/3" in timer.tasks
        assert len(timer.intervals("t2/o1o1v1v1")) == 1
        assert len(timer.intervals("td2/o1o1v1v1")) == 1
        assert len(timer.intervals("energy_correction/2")) == 1
        assert len(timer.intervals("energy_correction/3")) == 1

    def test_apply_density_order(self):
        # no density_order
        test = LazyMp(cache.refstate["h2o_sto3g"])
        for level in [1, 2, 3, "sigma4+", 4]:
            assert test._apply_density_order(level) == level
        # number density order
        test = LazyMp(cache.refstate["h2o_sto3g"], density_order=2)
        assert test._apply_density_order(1) == 2
        assert test._apply_density_order(2) == 2
        assert test._apply_density_order(3) == 3
        assert test._apply_density_order("sigma4+") == "sigma4+"
        # sigma4+ density order
        test = LazyMp(cache.refstate["h2o_sto3g"], density_order="sigma4+")
        assert test._apply_density_order(2) == "sigma4+"
        assert test._apply_density_order(3) == "sigma4+"
        assert test._apply_density_order("sigma4+") == "sigma4+"
        assert test._apply_density_order(4) == 4
        # invalid level
        with pytest.raises(ValueError):
            test._apply_density_order("sdf")
        # invalid density order
        with pytest.raises(ValueError):
            LazyMp(cache.refstate["h2o_sto3g"], density_order="sdf")
