#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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

import pytest

from adcc import hdf5io
from adcc.misc import cached_property

from .cache import fullfile


# TODO: remove once we have this from adcman
class TmpTestdataPropertyCache():
    @cached_property
    def qchem_results(self):
        """
        The definition of the test cases: Data generator and reference file
        """
        cases = ["h2o_sto3g"]
        methods = ["cvs_adc2", "adc2"]
        if not hasattr(pytest, "config") or pytest.config.option.mode == "full":
            cases += ["h2o_ccpvdz"]
        ret = {}
        for k in cases:
            for m in methods:
                datafile = fullfile(k + "_" + m + "_qc.hdf5")
                if datafile is None or not os.path.isfile(datafile):
                    continue
                ret[k + "_" + m] = hdf5io.load(datafile)
        return ret


property_cache = TmpTestdataPropertyCache()
