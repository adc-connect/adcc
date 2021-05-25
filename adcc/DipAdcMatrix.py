#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
ffrom .AdcMatrix import AdcMatrix
from .adc_dip import matrix as dipmatrix

class DipAdcMatrix(AdcMatrix):
    default_block_orders = {
        "adc0":  dict(hh_hh=0, phhh_hh=None, hh_phhh=None, phhh_phhh=None),  # noqa: E501
        "adc1":  dict(hh_hh=1, phhh_hh=None, hh_phhh=None, phhh_phhh=None),  # noqa: E501
        "adc2":  dict(hh_hh=2, phhh_hh=1,    hh_phhh=1,    phhh_phhh=0),     # noqa: E501
    }

    def __generate_matrix_block(self, *args, **kwargs):
        return dipmatrix.block(*args, **kwargs)

