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
import adcc
import unittest
import numpy as np
import itertools

from .misc import assert_allclose_signfix, expand_test_templates
from numpy.testing import assert_allclose

from adcc import ExcitedStates
from adcc.testdata.cache import cache

import pytest

from pytest import approx

# The methods to test
methods = ["adc0", "adc1", "adc2", "adc2x", "adc3"]

# The different reference data that are tested against
generators = ["atd", "adcc"]
kind_keyword = {"atd": "state", "adcc": "any"}


class TestFunctionalityBase(unittest.TestCase):
    def base_test(self, system, method, kind, generator="atd", prefix="",
                  test_mp=True, **args):
        if prefix:
            prefix += "-"
        hf = cache.hfdata[system]
        if generator == "atd":
            refdata = cache.reference_data[system]
            args["conv_tol"] = 3e-9
        else:
            refdata = cache.adcc_reference_data[system]
        ref = refdata[prefix.replace("_", "-") + method][kind]
        n_ref = len(ref["eigenvalues"])
        smallsystem = ("sto3g" in system or "631g" in system)

        # Run ADC and properties
        res = getattr(adcc, method.replace("-", "_"))(hf, **args)

        # Checks
        assert isinstance(res, ExcitedStates)
        assert res.converged
        assert_allclose(res.excitation_energy[:n_ref],
                        ref["eigenvalues"], atol=1e-7)

        if test_mp:
            refmp = refdata[prefix + "mp"]
            if res.method.level >= 2:
                assert res.ground_state.energy_correction(2) == \
                    approx(refmp["mp2"]["energy"])
            if res.method.level >= 3:
                if not res.method.is_core_valence_separated:
                    # TODO The latter check can be removed once CVS-MP3 energies
                    #      are implemented
                    assert res.ground_state.energy_correction(3) == \
                        approx(refmp["mp3"]["energy"])

        if generator == "atd":
            if "cvs" in method and res.method.level >= 2:
                pytest.xfail("From second order on, the CVS transition dipole "
                             "moments in the reference data are erroneous.")

        if method == "adc0" and "cn" in system:
            # TODO Investigate this
            pytest.xfail("CN adc0 transition properties currently fail for "
                         "unknown reasons and are thus not explicitly tested.")

        for i in range(n_ref):
            # Computing the dipole moment implies a lot of cancelling in the
            # contraction, which has quite an impact on the accuracy.
            res_tdm = res.transition_dipole_moment[i]
            ref_tdm = ref["transition_dipole_moments"][i]

            # Test norm and actual values
            res_tdm_norm = np.sum(res_tdm * res_tdm)
            ref_tdm_norm = np.sum(ref_tdm * ref_tdm)
            assert res_tdm_norm == approx(ref_tdm_norm, abs=1e-5)

            # If the eigenpair is degenerate, then some rotation
            # in the eigenspace is possible, which reflects as a
            # rotation inside the dipole moments. This is the case
            # for example for the CN test system. For simplicity,
            # we only compare the norm of the transition dipole moment
            # in such cases and skip the test for the exact values.
            if "cn" not in system:
                assert_allclose_signfix(res_tdm, ref_tdm, atol=1e-5)

        # Computing the dipole moment implies a lot of cancelling in the
        # contraction, which has quite an impact on the accuracy.
        assert_allclose(res.state_dipole_moment[:n_ref],
                        ref["state_dipole_moments"], atol=1e-4)

        # Test we do not use too many iterations
        if smallsystem:
            n_iter_bound = {
                "adc0": 1, "adc1": 4, "adc2": 10, "adc2x": 14, "adc3": 13,
                "cvs-adc0": 1, "cvs-adc1": 4, "cvs-adc2": 5, "cvs-adc2x": 12,
                "cvs-adc3": 13,
            }[method]
        else:
            n_iter_bound = {
                "adc0": 1, "adc1": 8, "adc2": 16, "adc2x": 17, "adc3": 17,
                "cvs-adc0": 1, "cvs-adc1": 7, "cvs-adc2": 16, "cvs-adc2x": 18,
                "cvs-adc3": 17,
            }[method]
        assert res.n_iter <= n_iter_bound


@expand_test_templates(list(itertools.product(generators, methods)))
class TestFunctionalityGeneral(TestFunctionalityBase):
    def template_h2o_sto3g_singlets(self, generator, method):
        n_singlets = 10
        if method == "adc2":
            n_singlets = 9
        self.base_test("h2o_sto3g", method, "singlet", generator,
                       n_singlets=n_singlets)

    def template_h2o_def2tzvp_singlets(self, generator, method):
        kwargs = {}
        if generator == "adcc":
            kwargs["n_guesses"] = 6
            kwargs["max_subspace"] = 24
        self.base_test("h2o_def2tzvp", method, "singlet", generator,
                       n_singlets=3, **kwargs)

    def template_h2o_sto3g_triplets(self, generator, method):
        self.base_test("h2o_sto3g", method, "triplet", generator, n_triplets=10)

    def template_h2o_def2tzvp_triplets(self, generator, method):
        kwargs = {}
        if generator == "adcc":
            kwargs["n_guesses"] = 6
            kwargs["max_subspace"] = 24
        self.base_test("h2o_def2tzvp", method, "triplet", generator,
                       n_triplets=3, **kwargs)

    def template_cn_sto3g(self, generator, method):
        kwargs = {}
        if method in ["adc0", "adc1"]:
            kwargs["max_subspace"] = 42
        self.base_test("cn_sto3g", method, kind_keyword[generator], generator,
                       n_states=8, **kwargs)

    def template_cn_ccpvdz(self, generator, method):
        kwargs = {}
        n_states = 5
        if method == "adc1":
            kwargs["max_subspace"] = 42
            n_states = 4
        self.base_test("cn_ccpvdz", method, kind_keyword[generator], generator,
                       n_states=n_states, **kwargs)


@expand_test_templates(list(itertools.product(generators, methods)))
class TestFunctionalityCvs(TestFunctionalityBase):
    def template_cvs_h2o_sto3g_singlets(self, generator, method):
        n_singlets = 3
        if method in ["adc0", "adc1"]:
            n_singlets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, "singlet", generator,
                       n_singlets=n_singlets, core_orbitals=1)

    def template_cvs_h2o_def2tzvp_singlets(self, generator, method):
        kwargs = {}
        if generator == "adcc":
            kwargs["n_guesses"] = 6
            kwargs["max_subspace"] = 24
        self.base_test("h2o_def2tzvp", "cvs-" + method, "singlet", generator,
                       n_singlets=3, core_orbitals=1, **kwargs)

    def template_cvs_h2o_sto3g_triplets(self, generator, method):
        n_triplets = 3
        if method in ["adc0", "adc1"]:
            n_triplets = 2
        self.base_test("h2o_sto3g", "cvs-" + method, "triplet", generator,
                       n_triplets=n_triplets, core_orbitals=1)

    def template_cvs_h2o_def2tzvp_triplets(self, generator, method):
        kwargs = {}
        if generator == "adcc":
            kwargs["n_guesses"] = 6
            kwargs["max_subspace"] = 24
        self.base_test("h2o_def2tzvp", "cvs-" + method, "triplet", generator,
                       n_triplets=3, core_orbitals=1, **kwargs)

    def template_cvs_cn_sto3g(self, generator, method):
        self.base_test("cn_sto3g", "cvs-" + method, kind_keyword[generator],
                       generator, n_states=6, core_orbitals=1)

    def template_cvs_cn_ccpvdz(self, generator, method):
        kwargs = {}
        if generator == "atd" and method in ["adc0", "adc1"]:
            kwargs["max_subspace"] = 28
        self.base_test("cn_ccpvdz", "cvs-" + method, kind_keyword[generator],
                       generator, n_states=5, core_orbitals=1, **kwargs)


@expand_test_templates(list(itertools.product(generators, methods)))
class TestFunctionalitySpinFlip(TestFunctionalityBase):
    def template_hf3_spin_flip(self, generator, method):
        self.base_test("hf3_631g", method, "spin_flip", generator, n_spin_flip=9)


@expand_test_templates(generators)
class TestFunctionalitySpaces(TestFunctionalityBase):
    #
    # H2O STO-3G
    #
    def base_test_h2o_sto3g(self, prefix, method, kind, generator):
        kw_prefix = {"fc": {"frozen_core": 1},
                     "fv": {"frozen_virtual": 1}, }
        kw_kind = {"singlet": {"n_singlets": 3},
                   "triplet": {"n_triplets": 3}, }

        kw_extra = kw_kind[kind]
        for pfx in prefix.split("-"):
            kw_extra.update(kw_prefix[pfx])
        if "cvs" in method:
            kw_extra["core_orbitals"] = 1
        self.base_test("h2o_sto3g", method, kind, generator, prefix=prefix,
                       **kw_extra)

    def template_h2o_sto3g_fc_adc2_singlets(self, generator):
        self.base_test_h2o_sto3g("fc", "adc2", "singlet", generator)

    def template_h2o_sto3g_fc_adc2_triplets(self, generator):
        self.base_test_h2o_sto3g("fc", "adc2", "triplet", generator)

    def template_h2o_sto3g_fc_fv_adc2_singlets(self, generator):
        self.base_test_h2o_sto3g("fc-fv", "adc2", "singlet", generator)

    def template_h2o_sto3g_fc_fv_adc2_triplets(self, generator):
        self.base_test_h2o_sto3g("fc-fv", "adc2", "triplet", generator)

    def template_h2o_sto3g_fv_adc2x_singlets(self, generator):
        self.base_test_h2o_sto3g("fv", "adc2x", "singlet", generator)

    def template_h2o_sto3g_fv_adc2x_triplets(self, generator):
        self.base_test_h2o_sto3g("fv", "adc2x", "triplet", generator)

    def template_h2o_sto3g_fv_cvs_adc2x_singlets(self, generator):
        self.base_test_h2o_sto3g("fv", "cvs-adc2x", "singlet", generator)

    def template_h2o_sto3g_fv_cvs_adc2x_triplets(self, generator):
        self.base_test_h2o_sto3g("fv", "cvs-adc2x", "triplet", generator)

    #
    # CN STO-3G
    #
    def base_test_cn_sto3g(self, prefix, method, generator):
        kw_prefix = {"fc": {"frozen_core": 1},
                     "fv": {"frozen_virtual": 1}, }
        kw_extra = {"n_states": 4, }
        if generator == "adcc" and prefix in ["fc", "fc-fv"]:
            kw_extra["max_subspace"] = 30
        for pfx in prefix.split("-"):
            kw_extra.update(kw_prefix[pfx])
        if "cvs" in method:
            kw_extra["core_orbitals"] = 1
        self.base_test("cn_sto3g", method, kind_keyword[generator], generator,
                       prefix=prefix, **kw_extra)

    def template_cn_sto3g_fc_adc2(self, generator):
        self.base_test_cn_sto3g("fc", "adc2", generator)

    def template_cn_sto3g_fc_fv_adc2(self, generator):
        self.base_test_cn_sto3g("fc-fv", "adc2", generator)

    def template_cn_sto3g_fv_adc2x(self, generator):
        self.base_test_cn_sto3g("fv", "adc2x", generator)

    def template_cn_sto3g_fv_cvs_adc2x(self, generator):
        self.base_test_cn_sto3g("fv", "cvs-adc2x", generator)

    #
    # H2S STO-3G
    #
    def base_test_h2s_sto3g(self, prefix, method, kind, generator):
        kw_prefix = {"fc": {"frozen_core": 1},
                     "fv": {"frozen_virtual": 1}, }
        kw_kind = {"singlet": {"n_singlets": 3},
                   "triplet": {"n_triplets": 3}, }
        kw_extra = kw_kind[kind]
        for pfx in prefix.split("-"):
            kw_extra.update(kw_prefix[pfx])
        if "cvs" in method:
            kw_extra["core_orbitals"] = 1
        self.base_test("h2s_sto3g", method, kind, generator, prefix=prefix,
                       test_mp=False, **kw_extra)

    def template_h2s_sto3g_fc_cvs_adc2_singlets(self, generator):
        self.base_test_h2s_sto3g("fc", "cvs-adc2", "singlet", generator)

    def template_h2s_sto3g_fc_cvs_adc2_triplets(self, generator):
        self.base_test_h2s_sto3g("fc", "cvs-adc2", "triplet", generator)

    def template_h2s_sto3g_fc_fv_cvs_adc2x_singlets(self, generator):
        self.base_test_h2s_sto3g("fc-fv", "cvs-adc2x", "singlet", generator)

    def template_h2s_sto3g_fc_fv_cvs_adc2x_triplets(self, generator):
        self.base_test_h2s_sto3g("fc-fv", "cvs-adc2x", "triplet", generator)

    #
    # H2S 6311+G**
    #
    def base_test_h2s_6311g(self, prefix, method, kind, generator):
        kw_prefix = {"fc": {"frozen_core": 1},
                     "fv": {"frozen_virtual": 3}, }
        kw_kind = {"singlet": {"n_singlets": 3},
                   "triplet": {"n_triplets": 3}, }

        kw_extra = kw_kind[kind]
        kw_extra["max_subspace"] = 60
        for pfx in prefix.split("-"):
            kw_extra.update(kw_prefix[pfx])
        if "cvs" in method:
            kw_extra["core_orbitals"] = 1
        self.base_test("h2s_6311g", method, kind, generator, prefix=prefix,
                       **kw_extra)

    def template_h2s_6311g_fc_adc2_singlets(self, generator):
        self.base_test_h2s_6311g("fc", "adc2", "singlet", generator)

    def template_h2s_6311g_fc_adc2_triplets(self, generator):
        self.base_test_h2s_6311g("fc", "adc2", "triplet", generator)

    def template_h2s_6311g_fv_adc2_singlets(self, generator):
        self.base_test_h2s_6311g("fv", "adc2", "singlet", generator)

    def template_h2s_6311g_fv_adc2_triplets(self, generator):
        self.base_test_h2s_6311g("fv", "adc2", "triplet", generator)

    def template_h2s_6311g_fc_fv_adc2_singlets(self, generator):
        self.base_test_h2s_6311g("fc-fv", "adc2", "singlet", generator)

    def template_h2s_6311g_fc_fv_adc2_triplets(self, generator):
        self.base_test_h2s_6311g("fc-fv", "adc2", "triplet", generator)

    def template_h2s_6311g_fc_cvs_adc2x_singlets(self, generator):
        self.base_test_h2s_6311g("fc", "cvs-adc2x", "singlet", generator)

    def template_h2s_6311g_fc_cvs_adc2x_triplets(self, generator):
        self.base_test_h2s_6311g("fc", "cvs-adc2x", "triplet", generator)

    def template_h2s_6311g_fv_cvs_adc2x_singlets(self, generator):
        self.base_test_h2s_6311g("fv", "cvs-adc2x", "singlet", generator)

    def template_h2s_6311g_fv_cvs_adc2x_triplets(self, generator):
        self.base_test_h2s_6311g("fv", "cvs-adc2x", "triplet", generator)

    def template_h2s_6311g_fc_fv_cvs_adc2x_singlets(self, generator):
        self.base_test_h2s_6311g("fc-fv", "cvs-adc2x", "singlet", generator)

    def template_h2s_6311g_fc_fv_cvs_adc2x_triplets(self, generator):
        self.base_test_h2s_6311g("fc-fv", "cvs-adc2x", "triplet", generator)
