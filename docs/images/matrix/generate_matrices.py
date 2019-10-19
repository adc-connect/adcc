#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import matplotlib
import numpy as np

from pyscf import gto, scf

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import h5py


def dump_matrix(basis, method):
    key = basis.replace("*", "s").replace("-", "").lower()
    fn = "water_" + key + "_" + method + ".hdf5"
    if os.path.isfile(fn):
        with h5py.File(fn, "r") as fp:
            return np.asarray(fp["adc_matrix"])

    mol = gto.M(
        atom="""O 0 0 0
                H 0 0 1.795239827225189
                H 1.693194615993441 0 -0.599043184453037""",
        basis=basis,
        unit="Bohr"
    )
    scfres = scf.RHF(mol)
    scfres.conv_tol = 1e-13
    scfres.kernel()

    mat = adcc.AdcMatrix(method, adcc.ReferenceState(scfres))
    key = basis.replace("*", "s").replace("-", "").lower()
    mat = mat.to_dense_matrix()

    fn = "water_" + key + "_" + method + ".hdf5"
    if not os.path.isfile(fn):
        with h5py.File(fn, "w") as fp:
            fp.create_dataset("adc_matrix", data=mat, compression="gzip")
    with h5py.File(fn, "r") as fp:
        return np.asarray(fp["adc_matrix"])


def plot_matrix(mtx, vrange=(-6, 2), sdline=None):
    plt.close()
    cmap = matplotlib.cm.YlOrRd
    cmap.set_bad("white", 1.)
    cmap.set_over(cmap(cmap.N), 1.)
    cmap.set_under("white", 1.)
    norm = LogNorm(vmin=10**min(vrange), vmax=10**max(vrange))

    ticks = []
    for i in range(min(vrange), max(vrange)):
        ticks += list(np.linspace(10**i, 10 * 10**i, 5, endpoint=False))
    ticks += [10**max(vrange)]

    img = plt.matshow(np.abs(mtx), cmap=cmap, norm=norm)
    plt.colorbar(img, ticks=ticks)
    plt.draw()

    if sdline:
        plt.axhline(y=40, xmin=0, xmax=mtx.shape[1],
                    color="grey", linewidth=0.7)
        plt.axvline(x=40, ymin=0, ymax=mtx.shape[1],
                    color="grey", linewidth=0.7)


def make_save_plot(basis, method, **kwargs):
    mtx = dump_matrix(basis, method)
    plot_matrix(mtx, **kwargs)

    key = basis.replace("*", "s").replace("-", "").lower()
    plt.savefig("matrix_water_" + method + "_" + key + ".png",
                bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    # make_save_plot("sto-3g", "adc1", sdline=40)
    make_save_plot("sto-3g", "adc2", sdline=40)
    make_save_plot("sto-3g", "adc3", sdline=40)
