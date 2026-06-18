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
from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Union, TypeVar
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
    # 3rd-order ISR: in doubles excitation space only (starting from 1-particle
    # operators, triples are required for a consistent third-order description)
    THREE_D = "3d"

    def to_str(self) -> str:
        return str(self.value)

    def to_int(self) -> int:
        """
        Converts the level to an integer. This also resolves special
        levels. For instance, 'TWO_X' resolves as 2.
        """
        # numerical methods
        if isinstance(self.value, int):
            return self.value
        # return base int for special methods
        elif isinstance(self.value, str):
            return int(self.value[0])
        else:
            raise ValueError(f"Unknown value type {type(self.value)}.")


class AdcType(Enum):
    PP = "pp"

    def to_str(self) -> str:
        return self.value

    @classmethod
    def is_valid_adc_type(cls, adc_type: str) -> bool:
        try:
            cls(adc_type)
            return True
        except ValueError:
            return False


class GroundStateType(Enum):
    MP = "mp"

    def to_str(self) -> str:
        return self.value

    @classmethod
    def is_valid_gs_type(cls, gs_type: str) -> bool:
        try:
            cls(gs_type)
            return True
        except ValueError:
            return False


@dataclass(frozen=True)
class LevelSpec:
    max_level: Optional[int] = None
    special_levels: tuple[MethodLevel, ...] = tuple()

    def supports(self, level: MethodLevel) -> bool:
        if isinstance(level.value, int):
            return self.max_level is not None and level.value <= self.max_level
        return level in self.special_levels


@dataclass(frozen=True)
class LevelKey:
    adc_type: AdcType
    gs_type: GroundStateType
    cvs: bool


class Method:
    # this has to be set on the child classes
    _method_base_name: Optional[str] = None
    _supported_levels: dict[LevelKey, LevelSpec] = {}

    def __init__(self, method: str):
        """
        Constructs a Method validating and decomposing the given method string.
        Valid method strings are of the form
        [prefixes]-[adc_type]-[method_base_name][level]
        with the individual components separated by '-'.
        """
        assert self._method_base_name is not None

        # validate base method type, which has to be the last part of the
        # method string
        split = method.split("-")
        if not split[-1].startswith(self._method_base_name):
            raise ValueError(f"{split[-1]} is not a valid method type")

        # validate method level
        level = split[-1][len(self._method_base_name):]
        if level.isnumeric():
            self.level: MethodLevel = MethodLevel(int(level))
        else:
            self.level: MethodLevel = MethodLevel(level)

        split = split[:-1]
        # validate and set the adc_type, which has to be the last entry of split,
        # i.e., the method string has to end with e.g. ip-adc2
        self.adc_type: AdcType = AdcType.PP
        if split and AdcType.is_valid_adc_type(split[-1]):
            self.adc_type: AdcType = AdcType(split[-1])
            split = split[:-1]
        # Therefore, all remaining prefixes at this point can not be
        # a valid adc_type
        if any(AdcType.is_valid_adc_type(pref) for pref in split):
            raise ValueError(
                f"Invalid method string {method}. ADC type (e.g. 'pp') either "
                "provided twice or in wrong position. Valid method strings "
                f"have to end with e.g. 'pp-adc2'."
            )
        # validate prefixes
        valid_prefixes: tuple[str, ...] = ("cvs", "mp")
        if any(pref not in valid_prefixes for pref in split):
            raise ValueError(f"Invalid method prefix in {split}. Valid prefixes "
                             f"are {valid_prefixes}.")
        if any(count != 1 for count in Counter(split).values()):
            raise ValueError(f"Invalid method string {method}. Duplicate "
                             f"prefix detected in {split}.")
        # set and remove cvs
        self.is_core_valence_separated: bool = "cvs" in split
        if self.is_core_valence_separated:
            split.remove("cvs")
        # deal with the gs_type
        self.gs_type: GroundStateType = GroundStateType.MP
        for pref in split:
            if GroundStateType.is_valid_gs_type(pref):
                self.gs_type: GroundStateType = GroundStateType(pref)
                split.remove(pref)
                break
        # The remaining prefixes can not be a ground state type anymore
        if any(GroundStateType.is_valid_gs_type(pref) for pref in split):
            raise ValueError(f"Invalid method string {method}. Ground state "
                             "type (e.g. 'mp') provided twice.")
        # at this point all prefixes should have been handled and removed
        if split:
            raise ValueError(f"Invalid method prefixes detected: {split}."
                             f"Parsed from method string '{method}'.")
        # ensure that the level is valid for the given method
        # this also depends on the adc type, gs_type and possibly other prefixes
        self._validate_level(self.level)

    def _validate_level(self, level: MethodLevel) -> None:
        key = LevelKey(
            adc_type=self.adc_type, gs_type=self.gs_type,
            cvs=self.is_core_valence_separated
        )
        spec = self._supported_levels.get(key, None)
        if spec is not None and spec.supports(level):
            return
        raise NotImplementedError(f"{self.name} method is not implemented.")

    @property
    def name(self) -> str:
        """The name of the Method as string."""
        if self.prefixes:
            return f"{self.prefixes}-{self._base_method}"
        else:
            return self._base_method

    @property
    def _base_method(self) -> str:
        assert self._method_base_name is not None
        if self.adc_type is AdcType.PP:
            return f"{self._method_base_name}{self.level.to_str()}"
        else:
            return (
                f"{self.adc_type.to_str()}-"
                f"{self._method_base_name}{self.level.to_str()}"
            )

    @property
    def prefixes(self) -> str:
        """String containing all prefixes of the method separated by '-'."""
        ret = []
        if self.is_core_valence_separated:
            ret.append("cvs")
        if self.gs_type is not GroundStateType.MP:
            ret.append(self.gs_type.to_str())
        return "-".join(ret)

    @property
    def base_method(self: T) -> T:
        """
        The base (full) method, i.e. with all approximations such as
        CVS stripped off.
        """
        return self.__class__(self._base_method)

    def at_level(self: T, newlevel: Union[int, str]) -> T:
        """
        Return an equivalent method, where only the level is changed
        (e.g. calling this on a CVS method returns a CVS method)
        """
        assert self._method_base_name is not None
        if self.prefixes:
            return self.__class__(
                f"{self.prefixes}-{self._method_base_name}{newlevel}"
            )
        else:
            return self.__class__(
                f"{self._method_base_name}{newlevel}"
            )

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

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Method):
            return NotImplemented
        return self.name == other.name

    def __ne__(self, other: Any):
        if not isinstance(other, Method):
            return NotImplemented
        return self.name != other.name

    def __repr__(self) -> str:
        return f"Method(name={self.name})"


class AdcMethod(Method):
    _method_base_name = "adc"
    _supported_levels = {
        LevelKey(
            adc_type=AdcType.PP,
            gs_type=GroundStateType.MP,
            cvs=False
        ): LevelSpec(
            max_level=4,
            special_levels=(MethodLevel.TWO_X,)
        ),
        LevelKey(
            adc_type=AdcType.PP,
            gs_type=GroundStateType.MP,
            cvs=True
        ): LevelSpec(
            max_level=3,
            special_levels=(MethodLevel.TWO_X,)
        )
    }


class IsrMethod(Method):
    _method_base_name = "isr"
    _supported_levels = {
        LevelKey(
            adc_type=AdcType.PP,
            gs_type=GroundStateType.MP,
            cvs=False
        ): LevelSpec(
            max_level=2,
            special_levels=(MethodLevel.ONE_S, MethodLevel.TWO_D)
        ),
        LevelKey(
            adc_type=AdcType.PP,
            gs_type=GroundStateType.MP,
            cvs=True
        ): LevelSpec(
            max_level=2,
            special_levels=(MethodLevel.ONE_S, MethodLevel.TWO_D)
        )
    }
