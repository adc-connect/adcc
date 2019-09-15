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
import numpy as np

from libadcc import HartreeFockProvider


class DictOperatorIntegralProvider:
    def __init__(self, data={}):
        self.data = data
        self.backend = data.get("backend", "dict")

        if "multipoles" in data:
            mmp = data["multipoles"]
            if "elec_1" in mmp:
                assert mmp["elec_1"].shape[0] == 3
                self.electric_dipole = list(mmp["elec_1"])


class DictHfProvider(HartreeFockProvider):
    def __init__(self, data):
        """
        Initialise the DictHfProvider class with the `data` containing the
        dictionary of data. Let `nf` denote the number of Fock spin orbitals
        (i.e. the sum of both the alpha and the beta orbitals) and `nb`
        the number of basis functions. The following keys are required in the
        dictionary:

        1. `restricted` (`bool`): `True` for a restricted SCF calculation,
           `False` otherwise
        2. `conv_tol` (`float`): Tolerance value used for SCF convergence,
           should be roughly equivalent to l2 norm of the Pulay error.
        3. orbcoeff_fb` (`np.array` with dtype `float`, size `(nf, nb)`):
           SCF orbital coefficients, i.e. the uniform transform from the basis
           to the molecular orbitals.
        4. `occupation_f` (`np.array` with dtype `float`, size `(nf, )`:
           Occupation number for each SCF orbitals (i.e. diagonal of the HF
           density matrix in the SCF orbital basis).
        5. `orben_f` (`np.array` with dtype `float`, size `(nf, )`:
           SCF orbital energies
        6. `fock_ff` (`np.array` with dtype `float`, size `(nf, nf)`:
           Fock matrix in SCF orbital basis. Notice, the full matrix is expected
           also for restricted calculations.
        7. `eri_phys_asym_ffff` (`np.array` with dtype `float`,
           size `(nf, nf, nf, nf)`: Antisymmetrised electron-repulsion integral
           tensor in the SCF orbital basis, using the Physicists' indexing
           covention, i.e. that the index tuple `(i,j,k,l)` refers to
           the integral
           `<ij||kl>`. TODO Equation for integral
           The full tensor (including zero blocks) is expected.

        As an alternative to `eri_phys_asym_ffff`, the user may provide

        8. `eri_ffff` (`np.array` with dtype `float`, size `(nf, nf, nf, nf)`:
           Electron-repulsion integral tensor in chemists' notation.
           The index tuple `(i,j,k,l)` thus refers to the integral `(ij|kl)`.
           Notice, that no antisymmetrisation has been applied in this tensor.

        The above keys define the least set of quantities to start a calculation
        in `adcc`. In order to have access to properties such as dipole moments
        or to get the correct state energies, further keys are highly
        recommended to be provided as well.

        9.  `backend` (`str`): Descriptive string for the backend of which data
            is contained in here (default: `dict`).
        10. `energy_scf` (`float`): Final total SCF energy of both electronic
            and nuclear energy terms. (default: `0.0`)
        11. `multipoles` (`dict`): Dictionary containing electric and nuclear
            multipole moments:

              - `elec_1` (`np.array`, size `(nb, nb)`): Electric dipole moment
                integrals in the atomic orbital basis (i.e. the discretisation
                basis with `nb` elements).
              - `nuc_0` (`float`): Total nuclear charge
              - `nuc_1` (`np.array` size `(3, )`: Nuclear dipole moment

            The defaults for all entries are all-zero multipoles.
        12. `spin_multiplicity` (`int`): The spin mulitplicity of the HF
            ground state described by the data. A value of `0` (for unknown)
            should be supplied for unrestricted calculations.
            (default: 1 for restricted and 0 for unrestricted calculations)

        Parameters
        ----------
        data : dict
            Dictionary containing the HartreeFock data to use. For the required
            keys see details above.
        """
        # Do not forget the next line, otherwise weird errors result
        super().__init__()
        self.data = data
        self.operator_integral_provider = DictOperatorIntegralProvider(data)

        if data["orbcoeff_fb"].shape[0] % 2 != 0:
            raise ValueError("orbcoeff_fb first axis should have even length")
        nb = self.get_n_bas()
        nf = 2 * self.get_n_orbs_alpha()

        checks = [("orbcoeff_fb", (nf, nb)), ("occupation_f", (nf, )),
                  ("orben_f", (nf, )), ("fock_ff", (nf, nf)),
                  ("eri_ffff", (nf, nf, nf, nf)),
                  ("eri_phys_asym_ffff", (nf, nf, nf, nf)), ]
        for key, exshape in checks:
            if key not in data:
                continue
            if data[key].shape != exshape:
                raise ValueError("Shape mismatch for key {}: Expected {}, but "
                                 "got {}.".format(key, exshape,
                                                  data[key].shape))

    #
    # Required keys
    #
    def get_restricted(self):
        return self.data["restricted"]

    def get_conv_tol(self):
        if "conv_tol" in self.data:
            return self.data["conv_tol"]
        return self.data["threshold"]  # The old name was "threshold"

    def fill_occupation_f(self, out):
        out[:] = self.data["occupation_f"]

    def fill_orbcoeff_fb(self, out):
        out[:] = self.data["orbcoeff_fb"]

    def fill_orben_f(self, out):
        out[:] = self.data["orben_f"]

    def fill_fock_ff(self, slices, out):
        out[:] = self.data["fock_ff"][slices]

    def fill_eri_ffff(self, slices, out):
        out[:] = self.data["eri_ffff"][slices]

    def fill_eri_phys_asym_ffff(self, slices, out):
        out[:] = self.data["eri_phys_asym_ffff"][slices]

    def has_eri_phys_asym_ffff_inner(self):
        return "eri_phys_asym_ffff" in self.data

    #
    # Recommended keys
    #
    def get_backend(self):
        return self.data.get("backend", "dict")

    def get_energy_scf(self):
        return self.data.get("energy_scf", 0.0)

    def get_nuclear_multipole(self, order):
        if order == 0:
            # The function interface needs to be a np.array on return
            nuc_0 = self.data.get("multipoles", {}).get("nuclear_0", 0.0)
            return np.array([nuc_0])
        elif order == 1:
            nuc_1 = self.data.get("multipoles", {}).get("nuclear_1", [0., 0, 0])
            return np.array(nuc_1)
        else:
            raise NotImplementedError("get_nuclear_multipole with order > 1")

    def get_spin_multiplicity(self):
        if "spin_multiplicity" in self.data:
            return self.data["spin_multiplicity"]
        elif not self.get_restricted():
            return 0
        else:
            return self.get_n_alpha() - self.get_n_beta() + 1

    #
    # Deduced keys
    #
    def get_n_alpha(self):
        na = self.get_n_orbs_alpha()
        return int(np.sum(self.data["occupation_f"][:na]))

    def get_n_beta(self):
        na = self.get_n_orbs_alpha()
        return int(np.sum(self.data["occupation_f"][na:]))

    def get_n_orbs_alpha(self):
        return self.data["orbcoeff_fb"].shape[0] // 2

    def get_n_bas(self):
        return self.data["orbcoeff_fb"].shape[1]
