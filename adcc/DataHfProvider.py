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
import numpy as np

import h5py

from libadcc import HartreeFockProvider


def get_scalar_value(data, key, default=None):
    """
    Function to allow to retrieve array data both from dict and from
    HDF5 objects
    """
    if "/" in key:
        key, subkey = key.split("/", 1)
        return get_scalar_value(data.get(key, {}), subkey, default=default)
    if default is not None and key not in data:
        return default

    value = data[key]
    if not hasattr(value, "shape"):
        return value  # Just a scalar
    elif value.shape == ():
        return value[()]
    elif value.shape == (1, ):
        return value[0]
    else:
        raise ValueError("Unrecognised scalar value shape ", value.shape,
                         " should be () or (1, )")


def get_array_value(data, key, default=None):
    """
    Function to allow to retrieve scalar data both from dict and from
    HDF5 objects
    """
    if "/" in key:
        key, subkey = key.split("/", 1)
        return get_array_value(data.get(key, {}), subkey, default=default)
    if default is not None and key not in data:
        return default
    return np.asarray(data[key])


class DataOperatorIntegralProvider:
    def __init__(self, backend="data"):
        self.backend = backend


class DataHfProvider(HartreeFockProvider):
    def __init__(self, data):
        """
        Initialise the DataHfProvider class with the `data` being a supported
        data container (currently python dictionary or HDF5 file).
        Let `nf` denote the number of Fock spin orbitals (i.e. the sum of both
        the alpha and the beta orbitals) and `nb` the number of basis functions.
        With `array` we indicate either a `np.array` or an HDF5 dataset.
        The following keys are required in the container:

        1. **restricted** (`bool`): `True` for a restricted SCF calculation,
           `False` otherwise
        2. **conv_tol** (`float`): Tolerance value used for SCF convergence,
           should be roughly equivalent to l2 norm of the Pulay error.
        3. **orbcoeff_fb** (`.array` with dtype `float`, size `(nf, nb)`):
           SCF orbital coefficients, i.e. the uniform transform from the basis
           to the molecular orbitals.
        4. **occupation_f** (`array` with dtype `float`, size `(nf, )`:
           Occupation number for each SCF orbitals (i.e. diagonal of the HF
           density matrix in the SCF orbital basis).
        5. **orben_f** (`array` with dtype `float`, size `(nf, )`:
           SCF orbital energies
        6. **fock_ff** (`array` with dtype `float`, size `(nf, nf)`:
           Fock matrix in SCF orbital basis. Notice, the full matrix is expected
           also for restricted calculations.
        7. **eri_phys_asym_ffff** (`array` with dtype `float`,
           size `(nf, nf, nf, nf)`: Antisymmetrised electron-repulsion integral
           tensor in the SCF orbital basis, using the Physicists' indexing
           convention, i.e. that the index tuple `(i,j,k,l)` refers to
           the integral :math:`\\langle ij || kl \\rangle`, i.e.

           .. math::
              \\int_\\Omega \\int_\\Omega d r_1 d r_2 \\frac{
              \\phi_i(r_1) \\phi_j(r_2)
              \\phi_k(r_1) \\phi_l(r_2)}{|r_1 - r_2|}
              - \\int_\\Omega \\int_\\Omega d r_1 d r_2 \\frac{
              \\phi_i(r_1) \\phi_j(r_2)
              \\phi_l(r_1) \\phi_k(r_2)}{|r_1 - r_2|}

           The full tensor (including zero blocks) is expected.

        As an alternative to `eri_phys_asym_ffff`, the user may provide

        8. **eri_ffff** (`array` with dtype `float`, size `(nf, nf, nf, nf)`:
           Electron-repulsion integral tensor in chemists' notation.
           The index tuple `(i,j,k,l)` thus refers to the integral
           :math:`(ij|kl)`, which is

           .. math::
              \\int_\\Omega \\int_\\Omega d r_1 d r_2
              \\frac{\\phi_i(r_1) \\phi_j(r_1)
              \\phi_k(r_2) \\phi_l(r_2)}{|r_1 - r_2|}

           Notice, that no antisymmetrisation has been applied in this tensor.

        The above keys define the least set of quantities to start a calculation
        in `adcc`. In order to have access to properties such as dipole moments
        or to get the correct state energies, further keys are highly
        recommended to be provided as well.

        9. **energy_scf** (`float`): Final total SCF energy of both electronic
           and nuclear energy terms. (default: `0.0`)
        10. **multipoles**: Container with electric and nuclear
            multipole moments. Can be another dictionary or simply an HDF5
            group.

              - **elec_1** (`array`, size `(3, nb, nb)`):
                Electric dipole moment integrals in the atomic orbital basis
                (i.e. the discretisation basis with `nb` elements). First axis
                indicates cartesian component (x, y, z).
              - **nuc_0** (`float`): Total nuclear charge
              - **nuc_1** (`array` size `(3, )`: Nuclear dipole moment

            The defaults for all entries are all-zero multipoles.
        11. **spin_multiplicity** (`int`): The spin mulitplicity of the HF
            ground state described by the data. A value of `0` (for unknown)
            should be supplied for unrestricted calculations.
            (default: 1 for restricted and 0 for unrestricted calculations)

        A descriptive string for the backend can be supplied optionally as well.
        In case of using a python `dict` as the data container, this should be
        done using the key `backend`. For an HDF5 file, this should be done
        using the attribute `backend`. Defaults based on the filename are
        generated.

        Parameters
        ----------
        data : dict or h5py.File
            Dictionary containing the HartreeFock data to use. For the required
            keys see details above.
        """

        # Do not forget the next line, otherwise weird errors result
        super().__init__()
        self.data = data

        if isinstance(data, dict):
            self.__backend = data.get("backend", "dict")
        elif isinstance(data, h5py.File):
            if "r" not in data.mode:
                raise ValueError("Passed h5py.File stream (filename: {}) not "
                                 "readable.".format(data.filename))
            self.__backend = data.attrs.get(
                "backend", '<HDF5 file "{}">'.format(data.filename)
            )
        else:
            raise TypeError("Can only deal with data objects of type dict "
                            "or h5py.File.")

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

        # Setup integral data
        opprov = DataOperatorIntegralProvider(self.__backend)
        mmp = data.get("multipoles", {})
        if "elec_1" in mmp:
            if mmp["elec_1"].shape != (3, nb, nb):
                raise ValueError("multipoles/elec_1 is expected to have shape "
                                 + str((3, nb, nb)) + " not "
                                 + str(mmp["elec_1"].shape))
            opprov.electric_dipole = np.asarray(mmp["elec_1"])
        magm = data.get("magnetic_moments", {})
        if "mag_1" in magm:
            if magm["mag_1"].shape != (3, nb, nb):
                raise ValueError("magnetic_moments/mag_1 is expected to have "
                                 "shape " + str((3, nb, nb)) + " not "
                                 + str(magm["mag_1"].shape))
            opprov.magnetic_dipole = np.asarray(magm["mag_1"])
        derivs = data.get("derivatives", {})
        if "nabla" in derivs:
            if derivs["nabla"].shape != (3, nb, nb):
                raise ValueError("derivatives/nabla is expected to "
                                 "have shape "
                                 + str((3, nb, nb)) + " not "
                                 + str(derivs["nabla"].shape))
            opprov.nabla = np.asarray(derivs["nabla"])
        self.operator_integral_provider = opprov

    #
    # Required keys
    #
    def get_restricted(self):
        return get_scalar_value(self.data, "restricted")

    def get_conv_tol(self):
        if "conv_tol" in self.data:
            return get_scalar_value(self.data, "conv_tol")
        # The old name was "threshold"
        return get_scalar_value(self.data, "threshold")

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
        # Only required if eri_ffff not provided
        out[:] = self.data["eri_phys_asym_ffff"][slices]

    #
    # Recommended keys
    #
    def get_backend(self):
        return self.__backend

    def get_energy_scf(self):
        return get_scalar_value(self.data, "energy_scf", 0.0)

    def get_nuclear_multipole(self, order):
        if order == 0:  # The function interface needs an np.array on return
            nuc_0 = get_scalar_value(self.data, "multipoles/nuclear_0", 0.0)
            return np.array([nuc_0])
        elif order == 1:
            return get_array_value(self.data, "multipoles/nuclear_1",
                                   [0., 0, 0])
        else:
            raise NotImplementedError("get_nuclear_multipole with order > 1")

    def get_spin_multiplicity(self):
        if "spin_multiplicity" in self.data:
            return get_scalar_value(self.data, "spin_multiplicity")
        elif not self.get_restricted():
            return 0
        else:
            noa = self.get_n_orbs_alpha()
            na = int(np.sum(self.data["occupation_f"][:noa]))
            nb = int(np.sum(self.data["occupation_f"][noa:]))
            return na - nb + 1

    #
    # Deduced keys
    #
    def get_n_orbs_alpha(self):
        return self.data["orbcoeff_fb"].shape[0] // 2

    def get_n_bas(self):
        return self.data["orbcoeff_fb"].shape[1]

    def has_eri_phys_asym_ffff_inner(self):
        return "eri_phys_asym_ffff" in self.data


class DictHfProvider(DataHfProvider):
    def __init__(self, *args, **kwargs):
        from warnings import warn

        super().__init__(*args, **kwargs)
        warn(DeprecationWarning("DictHfProvider is deprecated, "
                                "use DataHfProvider"))
