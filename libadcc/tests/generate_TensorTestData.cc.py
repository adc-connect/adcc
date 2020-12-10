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
import subprocess
import numpy as np

N = 3
a = np.random.randn(N, N, N, N)

a_sym_01 = 0.5 * (a + a.transpose((1, 0, 2, 3)))
a_asym_01 = 0.5 * (a - a.transpose((1, 0, 2, 3)))
a_sym_01_23 = 0.5 * (a + a.transpose((1, 0, 3, 2)))
a_asym_01_23 = 0.5 * (a - a.transpose((1, 0, 3, 2)))

#
#
#

HEADER = """
//
// Copyright (C) 2018 by the adcc authors
//
// This file is part of adcc.
//
// adcc is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// adcc is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with adcc. If not, see <http://www.gnu.org/licenses/>.
//

#include "TensorTestData.hh"

"""


def npprint(arr):
    fmt = '{:.18e}'
    str_arr = [fmt.format(x) for x in arr]
    return ",".join(str_arr)


with open("./TensorTestData.cc", "w") as f:
    f.write(HEADER)
    f.write("namespace libadcc {\nnamespace tests {\n\n")
    f.write("size_t TensorTestData::N = " + str(N) + ";")
    f.write("std::vector<double> TensorTestData::a = {"
            + npprint(a.ravel()) + "};")
    f.write("std::vector<double> TensorTestData::a_sym_01 = {"
            + npprint(a_sym_01.ravel()) + "};")
    f.write("std::vector<double> TensorTestData::a_asym_01 = {"
            + npprint(a_asym_01.ravel()) + "};")
    f.write("std::vector<double> TensorTestData::a_sym_01_23 = {"
            + npprint(a_sym_01_23.ravel()) + "};")
    f.write("std::vector<double> TensorTestData::a_asym_01_23 = {"
            + npprint(a_asym_01_23.ravel()) + "};")
    f.write("\n}\n}")


subprocess.check_call(["clang-format", "-i", "./TensorTestData.cc"])
