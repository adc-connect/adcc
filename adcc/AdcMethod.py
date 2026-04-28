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
from typing import Optional, TypeVar
from enum import Enum

T = TypeVar("T", bound="Method")


class MethodLevel(Enum):
    # numeric levels
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5

    # special levels
    TWO_X = "2x"    # extended 2nd-order ADC: 2p2h-2p2h in 1st order
    # 1st-order ISR: in singles excitation space only (starting from 1-particle
    # operators, doubles are required for a consistent first-order description)
    ONE_S = "1s"
    # 2nd-order ISR: in doubles excitation space only (starting from 2-particle
    # operators, triples are required for a consistent second-order description)
    TWO_D = "2d"
    # 3nd-order ISR: in doubles excitation space only (starting from 1-particle
    # operators, triples are required for a consistent third-order description)
    THREE_D = "3d"

    def to_str(self) -> str:
        return str(self.value)

    def to_int(self) -> int:
        # numerical methods
        if isinstance(self.value, int):
            return self.value
        # return base int for special methods
        elif isinstance(self.value, str):
            return int(self.value[0])
        else:
            raise ValueError


class Method:
    # this has to be set on the child classes
    _method_base_name: Optional[str] = None
    max_level: int = 0
    special_levels: tuple[MethodLevel, ...] = tuple()

    def __init__(self, method: str):
        assert self._method_base_name is not None

        # validate base method type
        split = method.split("-")
        if not split[-1].startswith(self._method_base_name):
            raise ValueError(f"{split[-1]} is not a valid method type")

        # validate method level
        level = split[-1][len(self._method_base_name):]
        if level.isnumeric():
            self.level: MethodLevel = MethodLevel(int(level))
        else:
            self.level: MethodLevel = MethodLevel(level)
        self._validate_level(self.level)

        assert self._base_method == split[-1]

        # validate prefix
        split = split[:-1]
        valid_prefixes: tuple[str, ...] = ("cvs",)
        if len(split) > len(valid_prefixes):
            raise ValueError("Invalid number of method prefixes provided "
                             f"in {split}.")
        if any(pref not in valid_prefixes for pref in split):
            raise ValueError(f"Invalid method prefix in {split}.")

        self.is_core_valence_separated: bool = "cvs" in split
        # NOTE: added this to make the testdata generation ready for IP/EA
        self.adc_type: str = "pp"

    def _validate_level(self, level: MethodLevel) -> None:
        if isinstance(level.value, int) and level.value <= self.max_level:
            return

        # special cases
        if level in self.special_levels:
            return

        raise NotImplementedError(f"{self._base_method} is not implemented.")

    @property
    def name(self) -> str:
        """The name of the Method as string."""
        if self.is_core_valence_separated:
            return "cvs-" + self._base_method
        else:
            return self._base_method

    @property
    def _base_method(self) -> str:
        assert self._method_base_name is not None
        return self._method_base_name + self.level.to_str()

    @property
    def base_method(self: T) -> T:
        """
        The base (full) method, i.e. with all approximations such as
        CVS stripped off.
        """
        return self.__class__(self._base_method)

    def at_level(self: T, newlevel: int) -> T:
        """
        Return an equivalent method, where only the level is changed
        (e.g. calling this on a CVS method returns a CVS method)
        """
        assert self._method_base_name is not None
        if self.is_core_valence_separated:
            return self.__class__("cvs-" + self._method_base_name + str(newlevel))
        else:
            return self.__class__(self._method_base_name + str(newlevel))

    def as_method(self, method_cls: type[T]) -> T:
        """
        Return a equivalent Method with the method base name replaced
        by the provided name.
        """
        assert self._method_base_name is not None
        assert method_cls._method_base_name is not None
        return method_cls(
            self.name.replace(self._method_base_name, method_cls._method_base_name)
        )

    def __eq__(self, other):
        return self.name == other.name

    def __ne__(self, other):
        return self.name != other.name

    def __repr__(self):
        return "Method(name={})".format(self.name)


class AdcMethod(Method):
    _method_base_name = "adc"
    max_level = 3
    special_levels = (MethodLevel.TWO_X,)


class IsrMethod(Method):
    _method_base_name = "isr"
    max_level = 2
    special_levels = (MethodLevel.ONE_S, MethodLevel.TWO_D)
