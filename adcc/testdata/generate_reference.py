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
import os
import ast
import sys
import adcc

from dump_reference_adcc import dump_reference_adcc
from os.path import dirname, join
from adcc.MoSpaces import expand_spaceargs

import h5py

sys.path.insert(0, join(dirname(__file__), "adcc-testdata"))

import adcctestdata as atd  # noqa: E402


def dump_all(case, kwargs, kwargs_overwrite={}, spec="gen", generator="adcc"):
    assert spec in ["gen", "cvs"]
    for method in ["adc0", "adc1", "adc2", "adc2x", "adc3"]:
        kw = kwargs_overwrite.get(method, kwargs)
        dump_method(case, method, kw, spec, generator=generator)


def dump_method(case, method, kwargs, spec, generator="adcc"):
    h5file = case + "_hfdata.hdf5"
    if not os.path.isfile(h5file):
        raise ValueError("HfData not found: " + h5file)

    if generator == "atd":
        dumpfunction = atd.dump_reference
        hfdata = atd.HdfProvider(h5file)
    else:
        dumpfunction = dump_reference_adcc
        hfdata = adcc.DataHfProvider(h5py.File(h5file, "r"))

    # Get dictionary of parameters for the reference cases.
    refcases = ast.literal_eval(hfdata.data["reference_cases"][()].decode())
    kwargs = dict(kwargs)
    if generator == "atd":
        kwargs.update(expand_spaceargs(hfdata, **refcases[spec]))
    else:
        kwargs.update(refcases[spec])

    fullmethod = method
    if "cvs" in spec:
        fullmethod = "cvs-" + method

    prefix = ""
    if spec != "gen":
        prefix = spec.replace("-", "_") + "_"
    adc_tree = prefix.replace("_", "-") + method
    mp_tree = prefix.replace("_", "-") + "mp"

    if generator == "atd":
        dumpfile = "{}_reference_{}{}.hdf5".format(case, prefix, method)
    else:
        dumpfile = "{}_adcc_reference_{}{}.hdf5".format(case, prefix, method)
    if not os.path.isfile(dumpfile):
        dumpfunction(hfdata, fullmethod, dumpfile, mp_tree=mp_tree,
                     adc_tree=adc_tree, n_states_full=2, **kwargs)


#
# =============================================================================
#


def dump_h2o_sto3g():  # H2O restricted
    # All methods for general and CVS
    kwargs = {"n_singlets": 10, "n_triplets": 10}
    overwrite = {"adc2": {"n_singlets": 9, "n_triplets": 10}, }
    dump_all("h2o_sto3g", kwargs, overwrite, spec="gen")

    kwargs = {"n_singlets": 3, "n_triplets": 3}
    overwrite = {
        "adc0": {"n_singlets": 2, "n_triplets": 2},
        "adc1": {"n_singlets": 2, "n_triplets": 2},
    }
    dump_all("h2o_sto3g", kwargs, overwrite, spec="cvs")

    case = "h2o_sto3g"  # Just ADC(2) and ADC(2)-x
    kwargs = {"n_singlets": 3, "n_triplets": 3}
    dump_method(case, "adc2", kwargs, spec="fc")
    dump_method(case, "adc2", kwargs, spec="fc-fv")
    dump_method(case, "adc2x", kwargs, spec="fv")
    dump_method(case, "adc2x", kwargs, spec="fv-cvs")


def dump_h2o_def2tzvp():  # H2O restricted
    kwargs = {"n_singlets": 3, "n_triplets": 3, "n_guess_singles": 6,
              "max_subspace": 24}
    dump_all("h2o_def2tzvp", kwargs, spec="gen")
    dump_all("h2o_def2tzvp", kwargs, spec="cvs")


def dump_cn_sto3g():  # CN unrestricted
    dump_all("cn_sto3g", {"n_states": 8, "n_guess_singles": 10}, spec="gen")
    dump_all("cn_sto3g", {"n_states": 6, "n_guess_singles": 7}, spec="cvs")

    # Just ADC(2) and ADC(2)-x for the other methods
    case = "cn_sto3g"
    dump_method(case, "adc2", {"n_states": 4, "n_guess_singles": 12,
                               "max_subspace": 30}, spec="fc")
    dump_method(case, "adc2", {"n_states": 4, "n_guess_singles": 14,
                               "max_subspace": 30}, spec="fc-fv")
    dump_method(case, "adc2x", {"n_states": 4, "n_guess_singles": 8}, spec="fv")
    dump_method(case, "adc2x", {"n_states": 4}, spec="fv-cvs")


def dump_cn_ccpvdz():  # CN unrestricted
    kwargs = {"n_states": 5, "n_guess_singles": 7}
    overwrite = {"adc1": {"n_states": 4, "n_guess_singles": 8}, }
    dump_all("cn_ccpvdz", kwargs, overwrite, spec="gen")
    dump_all("cn_ccpvdz", kwargs, spec="cvs")


def dump_hf3_631g():  # HF triplet unrestricted (spin-flip)
    dump_all("hf3_631g", {"n_spin_flip": 9}, spec="gen")


def dump_h2s_sto3g():
    case = "h2s_sto3g"
    kwargs = {"n_singlets": 3, "n_triplets": 3}

    dump_method(case, "adc2", kwargs, spec="fc-cvs")
    dump_method(case, "adc2x", kwargs, spec="fc-fv-cvs")


def dump_h2s_6311g():
    case = "h2s_6311g"
    kwargs = {"n_singlets": 3, "n_triplets": 3}
    for spec in ["gen", "fc", "fv", "fc-fv"]:
        dump_method(case, "adc2", kwargs, spec=spec)

    kwargs = {"n_singlets": 3, "n_triplets": 3, "n_guess_singles": 6,
              "max_subspace": 60}
    for spec in ["fv-cvs", "fc-cvs", "fc-fv-cvs"]:
        dump_method(case, "adc2x", kwargs, spec=spec)

    kwargs["n_guess_singles"] = 8
    dump_method(case, "adc2x", kwargs, spec="cvs")


def dump_methox_sto3g():  # (R)-2-methyloxirane
    kwargs = {"n_singlets": 2}
    dump_all("methox_sto3g", kwargs, spec="gen", generator="adcc")
    dump_all("methox_sto3g", kwargs, spec="cvs", generator="adcc")


def main():
    dump_h2o_sto3g()
    dump_h2o_def2tzvp()
    dump_cn_sto3g()
    dump_cn_ccpvdz()
    dump_hf3_631g()
    dump_h2s_sto3g()
    dump_h2s_6311g()
    dump_methox_sto3g()


if __name__ == "__main__":
    main()
