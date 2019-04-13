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
import unittest

import adcc

from pytest import approx
from libadcc import HartreeFockProvider
from adcc.testdata.cache import cache


class DummyData(HartreeFockProvider):
    def __init__(self, data):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.data = data

    def get_n_alpha(self):
        return self.data["n_alpha"]

    def get_n_beta(self):
        return self.data["n_beta"]

    def get_threshold(self):
        return self.data["threshold"]

    def get_restricted(self):
        return self.data["restricted"]

    def get_energy_term(self, term):
        return self.data["energy_" + term]

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

    def fill_orbcoeff_fb(self, out):
        out[:] = self.data["orbcoeff_fb"]

    def fill_orben_f(self, out):
        out[:] = self.data["orben_f"]

    def fill_fock_ff(self, slices, out):
        out[:] = self.data["fock_ff"][slices]

    def fill_eri_ffff(self, slices, out):
        out[:] = self.data["eri_ffff"][slices]

    def get_energy_term_keys(self):
        return ["nuclear_repulsion"]


class TestHartreeFockProvider(unittest.TestCase):
    def base_test(self, system, **args):
        hf = DummyData(cache.hfdata[system])
        refdata = cache.reference_data[system]

        res = adcc.adc2(hf, n_singlets=10, **args)
        assert type(res) == list
        assert len(res) == 1
        res = res[0]  # Extract the first (and only state)

        ref = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref)

    def test_h2o(self):
        self.base_test("h2o_sto3g")
