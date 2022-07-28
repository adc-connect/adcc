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
import unittest

from adcc.misc import expand_test_templates
#from . import block as b
#from numpy.testing import assert_allclose

#from adcc import LazyMp
from adcc.testdata.cache import cache

#import pytest

#from pytest import approx

#import os


# This test is different from the others, since the "approx" method to
# the full matrix dimension has to equal the "standard" method. Since
# they further require different routines apart from those, which are
# already tested with absolute values, we can just test them against
# each other.

# In order to keep this test as small as possible, we also require a
# new test case, HF sto-3g, since it contains only one virtual orbital,
# which keeps the matrix dimension low. This is important since we
# build most of the matrix in the approx method from properties, so this
# is very slow for a lot of states, compared to the standard method.

# One should add more testcases, but they should be added to
# cache.mode_full
testcases = ["h2o_sto3g"]

@expand_test_templates(testcases)
class qed_test(unittest.TestCase):
    def __init__(self, case):
        self.refstate = cache.reference_data(case)
        self.refstate.coupling = [0.0, 0.0, 0.05] # set some coupling
        self.refstate.frequency = [0.0, 0.0, 0.5] # set some frequency
        print(self.refstate)