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


class Method:
    def __init__(self, method, method_type):
        # validate base method
        split = method.split("-")
        if not split[-1].startswith(method_type):
            raise ValueError(f"{split[-1]} is not a valid method type")

        level = split[-1][3:]
        if level == "2x":
            self.level = 2
        elif not level.isnumeric:
            raise ValueError(f"{level} is not a valid method level")
        else:
            self.level = int(level)

        self.__base_method = split[-1]

        # validate prefix
        split = split[:-1]
        if split and "cvs" not in split:
            raise ValueError(f"{split[0]} is not a valid method prefix")

        self.is_core_valence_separated = "cvs" in split
        # NOTE: added this to make the testdata generation ready for IP/EA
        self.adc_type = "pp"

    @property
    def name(self):
        if self.is_core_valence_separated:
            return "cvs-" + self.__base_method
        else:
            return self.__base_method

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


class AdcMethod(Method):
    def __init__(self, method):
        super().__init__(method, "adc")
        if self.level > 3:
            raise NotImplementedError("Only ADC(0), ADC(1), ADC(2), ADC(2)-x, "
                                      "and ADC(3) are available.")

    def at_level(self, newlevel):
        """
        Return an equivalent method, where only the level is changed
        (e.g. calling this on a CVS method returns a CVS method)
        """
        if self.is_core_valence_separated:
            return AdcMethod("cvs-adc" + str(newlevel))
        else:
            return AdcMethod("adc" + str(newlevel))


class IsrMethod(Method):
    def __init__(self, method, validate_level=True):
        super().__init__(method, "isr")

        # Temporary workaround for tests
        # TODO: remove once ISR(3) is fully implemented
        if validate_level:
            if self.level > 2:
                raise NotImplementedError("Only ISR(0), ISR(1), and ISR(2) "
                                          "are available.")

    def at_level(self, newlevel):
        """
        Return an equivalent method, where only the level is changed
        (e.g. calling this on a CVS method returns a CVS method)
        """
        if self.is_core_valence_separated:
            return IsrMethod("cvs-isr" + str(newlevel))
        else:
            return IsrMethod("isr" + str(newlevel))
