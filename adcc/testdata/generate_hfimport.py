#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import adcc
import numpy as np

from adcc import hdf5io
from adcc.testdata.cache import cache


def build_dict(refstate):
    subspaces = refstate.mospaces.subspaces

    ret = {"subspaces": np.array(subspaces, dtype="S")}
    for ss in subspaces:
        ret["orbital_energies/" + ss] = \
            refstate.orbital_energies(ss).to_ndarray()
        ret["orbital_coefficients/" + ss + "b"] = \
            refstate.orbital_coefficients(ss + "b").to_ndarray()

    ss_pairs = []
    for i in range(len(subspaces)):
        for j in range(i, len(subspaces)):
            ss1, ss2 = subspaces[i], subspaces[j]
            ss_pairs.append(ss1 + ss2)
            ret["fock/{}{}".format(ss1, ss2)] = \
                refstate.fock(ss1 + ss2).to_ndarray()

    for i in range(len(ss_pairs)):
        for j in range(i, len(ss_pairs)):
            p1, p2 = ss_pairs[i], ss_pairs[j]
            ret["eri/{}{}".format(p1, p2)] = \
                refstate.eri(p1 + p2).to_ndarray()

    return ret


def dump_imported(key, dump_cvs=True):
    dumpfile = "{}_hfimport.hdf5".format(key)
    if os.path.isfile(dumpfile):
        return  # Done already

    print("Caching data for {} ...".format(key))
    data = cache.hfdata[key]
    dictionary = {}
    # TODO once hfdata is an HDF5 file
    # refcases = ast.literal_eval(data["reference_cases"][()])
    refcases = ast.literal_eval(data["reference_cases"])
    for name, args in refcases.items():
        print("Working on {} {} ...".format(key, name))
        refstate = adcc.ReferenceState(data, **args)
        for k, v in build_dict(refstate).items():
            dictionary[name + "/" + k] = v
    print("Writing data for {} ...".format(key))
    hdf5io.save(dumpfile, dictionary)


def main():
    # H2O restricted
    dump_imported("h2o_sto3g")
    dump_imported("h2o_def2tzvp")

    # CN unrestricted
    dump_imported("cn_sto3g")
    dump_imported("cn_ccpvdz")

    # CH2NH2 unrestricted (no symmetries)
    dump_imported("ch2nh2_sto3g")


if __name__ == "__main__":
    main()
