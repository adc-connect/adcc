#!/usr/bin/env python3
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
import pytest

from adcc.AdcMethod import AdcMethod, AdcType, GroundStateType, IsrMethod


adc_methods = [("adc1", None), ("adc2x", None), ("cvs-adc3", None),
               ("adc", ValueError), ("cvs_adc2", ValueError),
               ("xyz-adc2", ValueError), ("adc5", NotImplementedError),
               ("isr2", ValueError), ("mp-adc2", None), ("cvs-mp-adc2", None),
               ("cvs-cvs-adc2", ValueError), ("adcc", ValueError),
               ("pp-adc2", None), ("ee-adc2", ValueError),
               ("mp-pp-adc2", None), ("pp-mp-adc2", ValueError)]

isr_methods = [("isr1", None), ("cvs-isr2", None),
               ("adc", ValueError), ("cvs_isr2", ValueError),
               ("xyz-isr2", ValueError), ("isr5", NotImplementedError),
               ("adc2", ValueError)]


class TestAdcMethod:
    @pytest.mark.parametrize("method, expected_exception", adc_methods)
    def test_validate_adcmethod(self, method, expected_exception):
        if expected_exception:
            with pytest.raises(expected_exception):
                AdcMethod(method)
        else:
            adc_method = AdcMethod(method)
            assert adc_method.name == method.replace("mp-", "").replace("pp-", "")
            assert adc_method.adc_type is AdcType.PP
            assert adc_method.gs_type is GroundStateType.MP

    def test_adcmethod(self):
        method = AdcMethod("adc2")
        cvs_method = AdcMethod("cvs-adc2")

        assert method.name == cvs_method.base_method.name
        assert method.adc_type is AdcType.PP
        assert method.gs_type is GroundStateType.MP

        method_new_level = method.at_level(1)
        assert method_new_level._method_base_name == method._method_base_name
        assert method_new_level.level.to_int() == 1
        assert method_new_level.adc_type is AdcType.PP
        assert method_new_level.gs_type is GroundStateType.MP

        as_isr_method = method.as_method(IsrMethod)
        assert isinstance(as_isr_method, IsrMethod)
        assert as_isr_method.name == "isr2"
        assert as_isr_method.adc_type is AdcType.PP
        assert as_isr_method.gs_type is GroundStateType.MP


class TestIsrMethod:
    @pytest.mark.parametrize("method, expected_exception", isr_methods)
    def test_validate_isrmethod(self, method, expected_exception):
        if expected_exception:
            with pytest.raises(expected_exception):
                IsrMethod(method)
        else:
            isr_method = IsrMethod(method)
            assert isr_method.name == method
            assert isr_method.adc_type is AdcType.PP
            assert isr_method.gs_type is GroundStateType.MP

    def test_isrmethod(self):
        method = IsrMethod("isr2")
        cvs_method = IsrMethod("cvs-isr2")

        assert method.name == cvs_method.base_method.name

        method_new_level = method.at_level(1)
        assert method_new_level._method_base_name == method._method_base_name
        assert method_new_level.level.to_int() == 1
        assert method.adc_type is AdcType.PP
        assert method.gs_type is GroundStateType.MP

        as_adc_method = method.as_method(AdcMethod)
        assert isinstance(as_adc_method, AdcMethod)
        assert as_adc_method.adc_type is AdcType.PP
        assert as_adc_method.gs_type is GroundStateType.MP
