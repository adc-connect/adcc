#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import unittest

from numpy.testing import assert_allclose

from adcc import LazyMp
from adcc.testdata.cache import cache

import pytest

from .misc import expand_test_templates
from pytest import approx

# All test cases to deal with here
testcases = ["h2o_sto3g", "cn_sto3g"]
if pytest.config.option.mode == "full":
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

    def template_td2(self, case):
        refmp = cache.reference_data[case]["mp"]
        assert_allclose(self.mp[case].td2("o1o1v1v1").to_ndarray(),
                        refmp["mp2"]["td_o1o1v1v1"], atol=1e-12)

    def template_mp2_density_mo(self, case):
        refmp = cache.reference_data[case]["mp"]
        mp2diff = self.mp[case].mp2_diffdm

        assert mp2diff.is_symmetric
        for label in ["o1o1", "o1v1", "v1v1"]:
            assert_allclose(mp2diff[label].to_ndarray(),
                            refmp["mp2"]["dm_" + label], atol=1e-12)

    def template_mp2_density_ao(self, case):
        refmp = cache.reference_data[case]["mp"]
        mp2diff = self.mp[case].mp2_diffdm
        reference_state = self.mp[case].reference_state

        dm_α, dm_β = mp2diff.to_ao_basis(reference_state)
        assert_allclose(dm_α.to_ndarray(), refmp["mp2"]["dm_bb_a"], atol=1e-12)
        assert_allclose(dm_β.to_ndarray(), refmp["mp2"]["dm_bb_b"], atol=1e-12)

    def template_set_t2(self, case):
        mp = LazyMp(cache.refstate[case])
        t2orig = mp.t2("o1o1v1v1")
        mp2orig = mp.energy_correction(2)

        # Scale T2 amplitudes by factor 2, which should scale the energy by 2
        fac = 2
        t2scaled = fac * t2orig
        mp.set_t2("o1o1v1v1", t2scaled)
        mp2scaled = mp.energy_correction(2)

        assert mp2scaled == approx(fac * mp2orig)

    #
    # CVS
    #
    def template_cvs_mp2_energy(self, case):
        refmpcvs = cache.reference_data[case]["mp_cvs"]
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
        refmpcvs = cache.reference_data[case]["mp_cvs"]
        assert_allclose(self.mp_cvs[case].df("o1v1").to_ndarray(),
                        refmpcvs["mp1"]["df_o1v1"], atol=1e-12)
        assert_allclose(self.mp_cvs[case].df("o2v1").to_ndarray(),
                        refmpcvs["mp1"]["df_o2v1"], atol=1e-12)

    def template_cvs_t2(self, case):
        refmpcvs = cache.reference_data[case]["mp_cvs"]
        for label in ["o1o1v1v1", "o1o2v1v1", "o2o2v1v1"]:
            assert_allclose(self.mp_cvs[case].t2(label).to_ndarray(),
                            refmpcvs["mp1"]["t_" + label], atol=1e-12)

    def template_cvs_td2(self, case):
        refmpcvs = cache.reference_data[case]["mp_cvs"]
        for label in ["o1o1v1v1", "o1o2v1v1", "o2o2v1v1"]:
            if "td_" + label in refmpcvs["mp2"]:
                assert_allclose(self.mp_cvs[case].td2(label).to_ndarray(),
                                refmpcvs["mp2"]["td_" + label], atol=1e-12)

    def template_cvs_mp2_density_mo(self, case):
        refmpcvs = cache.reference_data[case]["mp_cvs"]
        mp2diff = self.mp_cvs[case].mp2_diffdm

        assert mp2diff.is_symmetric
        for label in ["o1o1", "o2o1", "o1v1", "o2o2", "o2v1", "v1v1"]:
            assert_allclose(mp2diff[label].to_ndarray(),
                            refmpcvs["mp2"]["dm_" + label], atol=1e-12)

    def template_cvs_mp2_density_ao(self, case):
        refmpcvs = cache.reference_data[case]["mp_cvs"]
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
