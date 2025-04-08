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
import unittest

from adcc.MoSpaces import expand_spaceargs
from adcc.HfCounterData import HfCounterData


class TestReferenceState(unittest.TestCase):
    def counter_data(self, n_bas, n_alpha=11, n_beta=12):
        return HfCounterData(n_alpha, n_beta, n_bas, n_bas, False)

    #
    # core_orbitals
    #
    def test_tuple_list_core_orbitals(self):
        res = expand_spaceargs(self.counter_data(n_bas=80),
                               core_orbitals=([0, 1, 2], [0, 1, 2]))
        assert res["core_orbitals"] == [0, 1, 2, 80, 81, 82]

        res = expand_spaceargs(self.counter_data(n_bas=70),
                               core_orbitals=([1, 3, 7], [2, 3, 6]))
        assert res["core_orbitals"] == [1, 3, 7, 72, 73, 76]

    def test_list_core_orbitals(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               core_orbitals=[1, 3, 7])
        assert res["core_orbitals"] == [1, 3, 7, 71, 73, 77]

    def test_range_core_orbitals(self):
        res = expand_spaceargs(self.counter_data(n_bas=80),
                               core_orbitals=range(3))
        assert res["core_orbitals"] == [0, 1, 2, 80, 81, 82]

        res = expand_spaceargs(self.counter_data(n_bas=70),
                               core_orbitals=range(1, 4))
        assert res["core_orbitals"] == [1, 2, 3, 71, 72, 73]

    def test_int_core_orbitals(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               core_orbitals=5)
        assert res["core_orbitals"] == [0, 1, 2, 3, 4, 70, 71, 72, 73, 74]

    #
    # frozen_core
    #
    def test_tuple_list_frozen_core(self):
        res = expand_spaceargs(self.counter_data(n_bas=80),
                               frozen_core=([0, 1, 2], [0, 1, 2]))
        assert res["frozen_core"] == [0, 1, 2, 80, 81, 82]

        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=([1, 3, 7], [2, 3, 6]))
        assert res["frozen_core"] == [1, 3, 7, 72, 73, 76]

    def test_list_frozen_core(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=[1, 3, 7])
        assert res["frozen_core"] == [1, 3, 7, 71, 73, 77]

    def test_range_frozen_core(self):
        res = expand_spaceargs(self.counter_data(n_bas=80),
                               frozen_core=range(3))
        assert res["frozen_core"] == [0, 1, 2, 80, 81, 82]

        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=range(1, 4))
        assert res["frozen_core"] == [1, 2, 3, 71, 72, 73]

    def test_int_frozen_core(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=5)
        assert res["frozen_core"] == [0, 1, 2, 3, 4, 70, 71, 72, 73, 74]

    #
    # frozen_virtual
    #
    def test_tuple_list_frozen_virtual(self):
        res = expand_spaceargs(self.counter_data(n_bas=80),
                               frozen_virtual=([77, 78, 79], [77, 78, 79]))
        assert res["frozen_virtual"] == [77, 78, 79, 157, 158, 159]

        res = expand_spaceargs(self.counter_data(n_bas=80),
                               frozen_virtual=([74, 76, 79], [73, 75, 77]))
        assert res["frozen_virtual"] == [74, 76, 79, 153, 155, 157]

    def test_list_frozen_virtual(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_virtual=[64, 66, 69])
        assert res["frozen_virtual"] == [64, 66, 69, 134, 136, 139]

    def test_range_frozen_virtual(self):
        res = expand_spaceargs(self.counter_data(n_bas=80),
                               frozen_virtual=range(77, 80))
        assert res["frozen_virtual"] == [77, 78, 79, 157, 158, 159]

    def test_int_frozen_virtual(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_virtual=4)
        assert res["frozen_virtual"] == [66, 67, 68, 69, 136, 137, 138, 139]

    #
    # fc and cvs
    #
    def test_int_fc_cvs(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=4, core_orbitals=2)
        assert res["frozen_core"] == [0, 1, 2, 3, 70, 71, 72, 73]
        assert res["core_orbitals"] == [4, 5, 74, 75]

    def test_list_fc_cvs(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=[0, 1, 3, 5], core_orbitals=[2, 6])
        assert res["frozen_core"] == [0, 1, 3, 5, 70, 71, 73, 75]
        assert res["core_orbitals"] == [2, 6, 72, 76]

    # TODO Not yet supported
    # def test_mixed_fc_cvs(self):
    #     res = expand_spaceargs(self.counter_data(n_bas=70),
    #                            frozen_core=[0, 1, 3, 5], core_orbitals=2)
    #     assert res["frozen_core"] == [0, 1, 3, 5, 70, 71, 73, 75]
    #     assert res["core_orbitals"] == [2, 4, 72, 74]

    #
    # fc and fv
    #
    def test_int_fc_fv(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=4, frozen_virtual=2)
        assert res["frozen_core"] == [0, 1, 2, 3, 70, 71, 72, 73]
        assert res["frozen_virtual"] == [68, 69, 138, 139]

    def test_list_fc_fv(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=[0, 1, 3, 5],
                               frozen_virtual=[66, 69])
        assert res["frozen_core"] == [0, 1, 3, 5, 70, 71, 73, 75]
        assert res["frozen_virtual"] == [66, 69, 136, 139]

    #
    # fc, cvs and fv
    #
    def test_int_fc_fv_cvs(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=4, core_orbitals=2, frozen_virtual=5)
        assert res["frozen_core"] == [0, 1, 2, 3, 70, 71, 72, 73]
        assert res["core_orbitals"] == [4, 5, 74, 75]
        assert res["frozen_virtual"] == \
            [65, 66, 67, 68, 69, 135, 136, 137, 138, 139]

    def test_list_fc_fv_cvs(self):
        res = expand_spaceargs(self.counter_data(n_bas=70),
                               frozen_core=[0, 1, 3, 5], core_orbitals=[2, 6],
                               frozen_virtual=[67, 69])
        assert res["frozen_core"] == [0, 1, 3, 5, 70, 71, 73, 75]
        assert res["core_orbitals"] == [2, 6, 72, 76]
        assert res["frozen_virtual"] == [67, 69, 137, 139]
