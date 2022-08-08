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


def get_valid_methods():
    valid_prefixes = ["cvs"]
    valid_bases = ["adc0", "adc1", "adc2", "adc2x", "adc3"]

    ret = valid_bases + [p + "-" + m for p in valid_prefixes
                         for m in valid_bases]
    return ret


class AdcMethod:
    available_methods = get_valid_methods()

    def __init__(self, method):
        if method not in self.available_methods:
            raise ValueError("Invalid method " + str(method) + ". Only "
                             + ",".join(self.available_methods) + " are known.")

        split = method.split("-")
        self.__base_method = split[-1]
        split = split[:-1]
        self.is_core_valence_separated = "cvs" in split

        try:
            if self.__base_method == "adc2x":
                self.level = 2
            else:
                self.level = int(self.__base_method[-1])
        except ValueError:
            raise ValueError("Not a valid base method: " + self.__base_method)

    def at_level(self, newlevel):
        """
        Return an equivalent method, where only the level is changed
        (e.g. calling this on a CVS method returns a CVS method)
        """
        if self.is_core_valence_separated:
            return AdcMethod("cvs-adc" + str(newlevel))
        else:
            return AdcMethod("adc" + str(newlevel))

    @property
    def name(self):
        if self.is_core_valence_separated:
            return "cvs-" + self.__base_method
        else:
            return self.__base_method

    @property
    def property_method(self):
        """
        The name of the canonical method to use for computing properties
        for this ADC method. This only differs from the name property
        for the ADC(2)-x family of methods.
        """
        if self.__base_method == "adc2x":
            return AdcMethod(self.name.replace("adc2x", "adc2")).name
        else:
            return self.name

    @property
    def base_method(self):
        """
        The base (full) method, i.e. with all approximations such as
        CVS stripped off.
        """
        return AdcMethod(self.__base_method)

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def __repr__(self):
        return "Method(name={})".format(self.name)
