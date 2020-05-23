#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import urllib.request
import hashlib
from adcc import hdf5io

_sha256sums = {
    "qchem_dump.hdf5":
        "942be5e3066574e2023dbada9eaa76249e4a123cebf31c5f8b2c3eb21bae859b"
}


class QChemData:
    @property
    def data(self):
        thisdir = os.path.dirname(__file__)
        qc_datafile = os.path.join(thisdir, "qchem_dump.hdf5")
        if not os.path.isfile(qc_datafile):
            # TODO: move somewhere else, just for travis to run the tests
            url = 'https://maxscheurer.com/files/qchem_dump.hdf5'
            response = urllib.request.urlopen(url)
            data = response.read()
            with open(qc_datafile, 'wb') as f:
                f.write(data)

        with open(qc_datafile, 'rb') as f:
            content = f.read()
            sha = hashlib.sha256(content)
            assert sha.hexdigest() == _sha256sums["qchem_dump.hdf5"]
        return hdf5io.load(qc_datafile)


qchem_data = QChemData().data
