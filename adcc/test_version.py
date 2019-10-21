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
import re
import ast
import adcc
import unittest

import pytest
import libadcc


def extract_adccore_version():
    """Extracts the desired adccore version from the setup.py script"""
    setupfile = os.path.join(adcc.__path__[-1], "..", "setup.py")
    assert os.path.isfile(setupfile)

    with open(setupfile, "r") as fp:
        for line in fp:
            match = re.match(r"^ *adccore_version *= *(\([^()]*\))", line)
            if match:
                return ast.literal_eval(match.group(1))[0]
        else:
            pytest.fail("Could not extract adccore version from " + setupfile)


class VersionTest(unittest.TestCase):
    def test_versions(self):
        """Test that versions of libadcc and adcc agree"""
        assert extract_adccore_version() == libadcc.__version__
