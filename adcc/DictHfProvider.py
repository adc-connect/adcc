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

from warnings import warn

from libadcc import HartreeFockProvider


class DummyOperatorIntegralProvider:
    pass


class DictHfProvider(HartreeFockProvider):
    """
    Very simple implementation of the HartreeFockProvider
    interface, which extracts the relevant data from
    a dictionary, which contains scalars and numpy
    arrays. See the implemntation for details about
    the expected keys.
    """

    def __init__(self, data):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.data = data
        self.operator_integral_provider = DummyOperatorIntegralProvider()

    def get_backend(self):
        return self.data.get("backend", "dict")

    def get_n_alpha(self):
        return self.data["n_alpha"]

    def get_n_beta(self):
        return self.data["n_beta"]

    def get_conv_tol(self):
        if "conv_tol" in self.data:
            return self.data["conv_tol"]
        return self.data["threshold"]

    def get_restricted(self):
        return self.data["restricted"]

    def get_energy_scf(self):
        return self.data["energy_scf"]

    def get_spin_multiplicity(self):
        return self.data["spin_multiplicity"]

    def get_n_orbs_alpha(self):
        return self.data["n_orbs_alpha"]

    def get_n_orbs_beta(self):
        return self.data["n_orbs_beta"]

    def get_n_bas(self):
        return self.data["n_bas"]

    def fill_occupation_f(self, out):
        if "occupation_f" in self.data:
            out[:] = self.data["occupation_f"]
        else:
            warn("Using dummy occupation wrapper in DictProvider")
            n_oa = self.get_n_orbs_alpha()
            n_ob = self.get_n_orbs_beta()
            out[:] = np.zeros(n_oa + n_ob)
            out[:self.get_n_alpha()] = 1.
            out[n_oa:n_oa + self.get_n_beta()] = 1.

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
