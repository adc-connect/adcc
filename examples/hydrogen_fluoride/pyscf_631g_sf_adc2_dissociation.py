#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import os
import adcc
import numpy as np

from pyscf import gto, scf
from matplotlib import pyplot as plt


def run_spin_flip(distance):
    # Run SCF in pyscf
    mol = gto.M(
        atom='H 0 0 0;'
             'F 0 0 {}'.format(distance),
        basis='6-31G',
        unit="Bohr",
        spin=2  # =2S, ergo triplet
    )
    scfres = scf.UHF(mol)
    scfres.conv_tol = 1e-12
    scfres.conv_tol_grad = 1e-8
    scfres.kernel()

    # Run ADC and compute total energy
    states = adcc.adc2(scfres, n_spin_flip=1)

    ene = scfres.energy_tot() + states.ground_state.energy_correction(2)
    return ene + states.excitation_energy[0]


def run_progression(outfile="631g_adc2_dissociation.nptxt"):
    if os.path.isfile(outfile):
        return np.loadtxt(outfile)

    dists = np.linspace(1.0, 5.0, 30)
    enes = [run_spin_flip(d) for d in dists]

    result = np.vstack((dists, enes)).T
    np.savetxt(outfile, result)
    return result


def main():
    result = run_progression()
    plt.plot(result[:, 0], result[:, 1])
    plt.show()


if __name__ == "__main__":
    main()
