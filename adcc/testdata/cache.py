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
import ast
import adcc

from .geometry import xyz

from adcc import AdcMatrix, ExcitedStates, LazyMp, guess_zero, hdf5io
from adcc.misc import cached_property
from adcc.solver import EigenSolverStateBase
from adcc.caching_policy import CacheAllPolicy

import pytest


class AdcMockState(EigenSolverStateBase):
    def __init__(self, matrix):
        super().__init__(matrix)


def make_mock_adc_state(refstate, matmethod, kind, reference):
    ground_state = LazyMp(refstate, CacheAllPolicy())
    matrix = AdcMatrix(matmethod, ground_state)

    # Number of full state results
    n_full = len(reference[kind]["eigenvectors_singles"])

    state = AdcMockState(matrix)
    state.method = matrix.method
    state.ground_state = ground_state
    state.reference_state = refstate
    state.kind = kind
    state.eigenvalues = reference[kind]["eigenvalues"][:n_full]

    spin_change = 0
    if refstate.restricted and kind == "singlet":
        symm = "symmetric"
    elif refstate.restricted and kind == "triplet":
        symm = "antisymmetric"
    elif kind in ["state", "spin_flip"]:
        symm = "none"
    else:
        raise ValueError("Unknown kind: {}".format(kind))

    state.eigenvectors = [
        guess_zero(matrix, irrep="A", spin_change=spin_change,
                   spin_block_symmetrisation=symm)
        for i in range(n_full)
    ]

    has_doubles = "eigenvectors_doubles" in reference[kind]
    vec_singles = reference[kind]["eigenvectors_singles"]
    vec_doubles = reference[kind].get("eigenvectors_doubles", None)
    for i, evec in enumerate(state.eigenvectors):
        evec["s"].set_from_ndarray(vec_singles[i])
        if has_doubles:
            evec["d"].set_from_ndarray(vec_doubles[i], 1e-14)
    return ExcitedStates(state)


def fullfile(fn):
    thisdir = os.path.dirname(__file__)
    if os.path.isfile(os.path.join(thisdir, fn)):
        return os.path.join(thisdir, fn)
    elif os.path.isfile(fn):
        return fn
    else:
        return ""


class TestdataCache():
    @property
    def testcases(self):
        """
        The definition of the test cases: Data generator and reference file
        """
        cases = ["h2o_sto3g", "cn_sto3g", "hf3_631g", "h2s_sto3g"]
        if not hasattr(pytest, "config") or pytest.config.option.mode == "full":
            cases += ["cn_ccpvdz", "h2o_def2tzvp", "h2s_6311g"]
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
            # TODO This could be made a plain HDF5.File
            ret[k] = hdf5io.load(datafile)
        return ret

    @cached_property
    def refstate(self):
        def cache_eri(refstate):
            refstate.import_all()
            return refstate
        return {k: cache_eri(adcc.ReferenceState(self.hfdata[k]))
                for k in self.testcases}

    @cached_property
    def refstate_cvs(self):
        ret = {}
        for case in self.testcases:
            # TODO once hfdata is an HDF5 file
            # refcases = ast.literal_eval(self.hfdata[case]["reference_cases"][()])
            refcases = ast.literal_eval(self.hfdata[case]["reference_cases"])
            if "cvs" not in refcases:
                continue
            ret[case] = adcc.ReferenceState(self.hfdata[case], **refcases["cvs"])
            ret[case].import_all()
        return ret

    def refstate_nocache(self, case, spec):
        # TODO once hfdata is an HDF5 file
        # refcases = ast.literal_eval(self.hfdata[case]["reference_cases"][()])
        refcases = ast.literal_eval(self.hfdata[case]["reference_cases"])
        return adcc.ReferenceState(self.hfdata[case], **refcases[spec])

    @cached_property
    def hfimport(self):
        ret = {}
        for k in self.testcases:
            datafile = fullfile(k + "_hfimport.hdf5")
            if os.path.isfile(datafile):
                ret[k] = hdf5io.load(datafile)
        return ret

    @cached_property
    def reference_data(self):
        prefixes = ["", "cvs", "fc", "fv", "fc_cvs",
                    "fv_cvs", "fc_fv", "fc_fv_cvs"]
        raws = ["adc0", "adc1", "adc2", "adc2x", "adc3"]
        methods = raws + ["_".join([p, r]) for p in prefixes
                          for r in raws if p != ""]

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
            if case not in self.reference_data:
                continue
            available_kinds = self.reference_data[case]["available_kinds"]
            res_case = {}
            for method in ["adc0", "adc1", "adc2", "adc2x", "adc3"]:
                if method not in self.reference_data[case]:
                    continue
                res_case[method] = {
                    kind: make_mock_adc_state(self.refstate[case], method, kind,
                                              self.reference_data[case][method])
                    for kind in available_kinds
                }

            for method in ["cvs-adc0", "cvs-adc1", "cvs-adc2",
                           "cvs-adc2x", "cvs-adc3"]:
                if method not in self.reference_data[case]:
                    continue
                res_case[method] = {
                    kind: make_mock_adc_state(self.refstate_cvs[case],
                                              method, kind,
                                              self.reference_data[case][method])
                    for kind in available_kinds
                }

            other_methods = [(spec, cvs, basemethod) for spec in ["fc", "fv"]
                             for cvs in ["", "cvs"]
                             for basemethod in ["adc2", "adc2x"]]
            for spec, cvs, basemethod in other_methods:
                # Find the method to put into the ADC matrix class
                if cvs:
                    matmethod = cvs + "-" + basemethod
                    fspec = spec + "-" + cvs
                else:
                    matmethod = basemethod
                    fspec = spec
                # The full method (including "spec" like "fc")
                method = spec + "-" + matmethod
                if method not in self.reference_data[case]:
                    continue
                res_case[method] = {
                    kind: make_mock_adc_state(
                        self.refstate_nocache(case, fspec), matmethod, kind,
                        self.reference_data[case][method])
                    for kind in available_kinds
                }

            res[case] = res_case
        return res


# Setup cache object
cache = TestdataCache()
