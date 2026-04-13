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
    def __init__(self, method: str, method_base_name: str):
        assert len(method_base_name) == 3
        self._method_base_name = method_base_name

        # validate base method type
        split = method.split("-")
        if not split[-1].startswith(self._method_base_name):
            raise ValueError(f"{split[-1]} is not a valid method type")

        # validate method level
        level = split[-1][3:]
        if level == "2x":
            self.level = 2
        elif not level.isnumeric:
            raise ValueError(f"{level} is not a valid method level")
        else:
            self.level = int(level)

        self._base_method = self._method_base_name + level
        assert self._base_method == split[-1]

        # validate prefix
        split = split[:-1]
        if split and split[0] not in ["cvs"]:
            raise ValueError(f"{split[0]} is not a valid method prefix")

        self.is_core_valence_separated = "cvs" in split
        # NOTE: added this to make the testdata generation ready for IP/EA
        self.adc_type = "pp"

    @property
    def name(self):
        if self.is_core_valence_separated:
            return "cvs-" + self._base_method
        else:
            return self._base_method

    @property
    def base_method(self):
        """
        The base (full) method, i.e. with all approximations such as
        CVS stripped off.
        """
        return self.__class__(self._base_method)

    def at_level(self, newlevel: int):
        """
        Return an equivalent method, where only the level is changed
        (e.g. calling this on a CVS method returns a CVS method)
        """
        if self.is_core_valence_separated:
            return self.__class__("cvs-" + self._method_base_name + str(newlevel))
        else:
            return self.__class__(self._method_base_name + str(newlevel))

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def __repr__(self):
        return "Method(name={})".format(self.name)


class AdcMethod(Method):
    def __init__(self, method: str):
        super().__init__(method, "adc")
        if self.level > 3:
            raise NotImplementedError(f"{method} not available, only ADC(0), "
                                      "ADC(1), ADC(2), ADC(2)-x, and ADC(3).")


class IsrMethod(Method):
    def __init__(self, method: str):
        super().__init__(method, "isr")
        if self.level > 2:
            raise NotImplementedError(f"{method} not available, "
                                      "only ISR(0), ISR(1), and ISR(2).")
