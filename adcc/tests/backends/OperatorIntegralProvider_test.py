## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
import unittest

import numpy as np

from adcc.backends import OperatorIntegralProvider


class MinimalProvider(OperatorIntegralProvider):
    @property
    def backend(self) -> str:
        return "minimal"


class PartialProvider(MinimalProvider):
    @property
    def overlap(self):
        return np.eye(2)

    def magnetic_dipole(self, gauge_origin="origin"):
        if gauge_origin != "mass_center":
            raise NotImplementedError("only mass_center")
        return (np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)))

    def some_helper(self):
        """Public helper that is not defined on the base class: no operator."""


class TestOperatorIntegralProvider(unittest.TestCase):
    def test_available(self):
        # test the implementation on the base class
        assert not MinimalProvider().available
        assert PartialProvider().available == ("magnetic_dipole", "overlap")
