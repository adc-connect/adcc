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
import os

import adcc

from adcc import AmplitudeVector, empty_like, hdf5io
from adcc.misc import cached_property


def compute_prelim(data, core_valence_separation=False):
    method = "adc2"
    n_core_orbitals = None
    if core_valence_separation:
        method = "cvs-adc2"
        n_core_orbitals = data["n_core_orbitals"]

    # Run preliminary ADC(2) or CVS-ADC(2) calculation
    return adcc.tmp_run_prelim(data, method, n_guess_singles=10,
                               n_core_orbitals=n_core_orbitals)


class AdcMockState():
    pass


def make_mock_adc_state(prelim, method, kind, reference):
    matrix = adcc.AdcMatrix(method, prelim.ground_state)
    guesses = getattr(prelim, "guesses_" + kind)

    # Number of full state results
    n_full = len(reference[method][kind]["eigenvectors_singles"])

    state = AdcMockState()
    state.method = matrix.method
    state.ground_state = prelim.ground_state
    state.reference_state = prelim.reference
    state.kind = kind
    state.eigenvalues = reference[method][kind]["eigenvalues"][:n_full]

    # prelim.guesses is computed via ADC(2), hence this has
    # a singles and a doubles part. For ADC(1), however, we only
    # need a singles part
    has_doubles = "eigenvectors_doubles" in reference[method][kind]

    if has_doubles:
        state.eigenvectors = [empty_like(gv) for gv in guesses[:n_full]]
    else:
        state.eigenvectors = [AmplitudeVector(empty_like(gv["s"]))
                              for gv in guesses[:n_full]]

    vec_singles = reference[method][kind]["eigenvectors_singles"]
    vec_doubles = reference[method][kind].get("eigenvectors_doubles", None)
    for i, evec in enumerate(state.eigenvectors):
        evec["s"].set_from_ndarray(vec_singles[i])
        if has_doubles:
            evec["d"].set_from_ndarray(vec_doubles[i])
    return state


def fullfile(fn):
    thisdir = os.path.dirname(__file__)
    if os.path.isfile(os.path.join(thisdir, fn)):
        return os.path.join(thisdir, fn)
    elif os.path.isfile(fn):
        return fn
    else:
        return None


class TestdataCache():
    @property
    def testcases(self):
        """
        The definition of the test cases: Data generator and reference file
        """
        cases = ["h2o_sto3g", "cn_sto3g"]
        return [k for k in cases
                if os.path.isfile(fullfile(k + "_hfdata.hdf5"))]

    @cached_property
    def hfdata(self):
        """
        The HF data a testcase is based upon
        """
        ret = {}
        for k in self.testcases:
            datafile = fullfile(k + "_hfdata.hdf5")
            ret[k] = hdf5io.load(datafile)
        return ret

    @cached_property
    def prelim(self):
        return {k: compute_prelim(self.hfdata[k]) for k in self.testcases}

    @cached_property
    def prelim_cvs(self):
        return {k: compute_prelim(self.hfdata[k], core_valence_separation=True)
                for k in self.testcases}

    @cached_property
    def reference_data(self):
        methods = ["cvs_adc0", "cvs_adc1", "cvs_adc2", "cvs_adc2x", "cvs_adc3",
                   "adc0", "adc1", "adc2", "adc2x", "adc3"]

        ret = {}
        for k in self.testcases:
            fulldict = {}
            for m in methods:
                datafile = fullfile(k + "_reference_" + m + ".hdf5")
                if datafile is None or not os.path.isfile(datafile):
                    continue
                fulldict.update(hdf5io.load(datafile))
            ret[k] = fulldict
        return ret

    @cached_property
    def adc_states(self):
        """
        Construct a hierachy of dicts, which contains a mock adc state
        for all test cases, all methods and all kinds (singlet, triplet)
        """
        res = {}
        for case in self.testcases:
            available_kinds = self.reference_data[case]["available_kinds"]
            res_case = {}
            for method in ["adc0", "adc1", "adc2", "adc2x", "adc3"]:
                res_case[method] = {
                    kind: make_mock_adc_state(self.prelim[case], method, kind,
                                              self.reference_data[case])
                    for kind in available_kinds
                }

            for cvs_method in ["cvs-adc0", "cvs-adc1", "cvs-adc2", "cvs-adc2x"]:
                res_case[cvs_method] = {
                    kind: make_mock_adc_state(self.prelim_cvs[case], cvs_method,
                                              kind, self.reference_data[case])
                    for kind in available_kinds
                }

            res[case] = res_case
        return res


# Setup memory and cache object
adcc.memory_pool.initialise(max_memory=512 * 1024 * 1024)  # 512 MiB
cache = TestdataCache()
