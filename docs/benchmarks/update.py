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
import json
import itertools
import numpy as np

import requests


def write_commit_file(commit):
    with open("commit.rst", "w") as fp:
        fp.write("This summary only shows a few key results, which have been "
                 f"generated using commit **{commit}** from the "
                 "`adcc repository <https://code.adc-connect.org>`_. "
                 "The full results in interactive form are accessible on "
                 "https://adc-connect.github.io/adcc-bench.")


def fetch_json(commit, machine):
    baseurl = ("https://raw.githubusercontent.com/adc-connect/"
               "adcc-bench/results")
    resultsfile = (baseurl + "/results/" + machine + "/" + commit
                   + "-virtualenv-py3.6-numpy1.15-pybind11-""pyscf1.6.3"
                   "-scipy1.2.json")
    res = requests.get(resultsfile)
    assert res.ok
    return json.loads(res.text)


def write_details(data, testcase, reference, basis, n_ao=None):
    lines = []

    memkey = [k for k in data["results"].keys()
              if f"Full{testcase}" in k and "peakmem_" in k]
    timekey = [k for k in data["results"].keys()
               if f"Full{testcase}" in k and "time_" in k]
    if len(memkey) == 0:
        return
    assert len(memkey) == 1
    assert len(timekey) == 1
    memkey = memkey[0]
    timekey = timekey[0]

    mbasis, mmethod, mstates, mtol, mthreads = data["results"][memkey]["params"]
    tbasis, tmethod, tstates, ttol, tthreads = data["results"][timekey]["params"]
    tol = max(float(e) for e in set(mtol).intersection(ttol))
    states = max(int(e) for e in set(mstates).intersection(tstates))
    mproduct = list(itertools.product(*data["results"][memkey]["params"]))
    tproduct = list(itertools.product(*data["results"][timekey]["params"]))
    assert len(mthreads) == len(tthreads) == 1
    assert mthreads[0] == tthreads[0]
    threads = mthreads[0]

    kind = " (energies only)"
    if "oscillator_strength" in memkey:
        kind = " (energies and oscillator strength)"
    nfunc = ""
    if n_ao is not None:
        nfunc = f" ({n_ao} functions)"
    lines += [f"- **Basis set:** {basis}{nfunc}"]
    lines += [f"- **Reference:** {reference}"]
    lines += [f"- **Convergence tolerance:** {tol}"]
    lines += [f"- **Number of states:** {states} {kind}"]
    lines += [f"- **Threads:** {threads}"]
    lines += [""]

    lines += ["=========  =============  ===================="]
    lines += ["method          time (s)     peak memory (MiB)"]
    lines += ["=========  =============  ===================="]
    for method in sorted(set(mmethod).intersection(tmethod)):
        method = method[1:-1]
        for bas in [basis.upper(), basis.lower()]:
            try:
                midx = mproduct.index(("'" + bas + "'", "'" + method + "'",
                                       str(states), str(tol), str(threads)))
                tidx = tproduct.index(("'" + bas + "'", "'" + method + "'",
                                       str(states), str(tol), str(threads)))
            except ValueError:
                continue

        time = data["results"][timekey]["result"][tidx]
        memory = data["results"][memkey]["result"][midx]
        if time is None:
            time = np.nan
        if memory is not None:
            memory = memory / 1024**2
        else:
            memory = np.nan
        lines += [f"{method:<9s}  {time:>13.0f}  {memory:>20.0f}"]
    lines += ["=========  =============  ===================="]

    with open(testcase + ".rst", "w") as fp:
        fp.write("\n".join(lines))


def main():
    commit = "9c7bba83"
    write_commit_file(commit)
    json_clustern08 = fetch_json(commit, "mlv-clustern08")
    write_details(json_clustern08, "PhosphineCvs", "RHF", "6-311++G**", n_ao=51)
    write_details(json_clustern08, "MethylammoniumRadical", "UHF",
                  "cc-pVTZ", n_ao=116)
    write_details(json_clustern08, "ParaNitroAniline", "RHF", "cc-pVDZ", n_ao=170)
    write_details(json_clustern08, "WaterExpensive", "RHF", "cc-pVQZ", n_ao=115)
    write_details(json_clustern08, "Noradrenaline", "RHF", "6-311++G**", n_ao=341)


if __name__ == "__main__":
    main()
