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
from .AdcMatrix import AdcMatrix
from .adc_pp import matrix as ppmatrix

class PpAdcMatrix(AdcMatrix):
    default_block_orders = {
        #             ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),
        "adc0":  dict(ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc1":  dict(ph_ph=1, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc2":  dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=0),     # noqa: E501
        "adc2x": dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=1),     # noqa: E501
        "adc3":  dict(ph_ph=3, ph_pphh=2,    pphh_ph=2,    pphh_pphh=1),     # noqa: E501
    }


    def __generate_matrix_block(self, *args, **kwargs):
        return ppmatrix.block(*args, **kwargs)

