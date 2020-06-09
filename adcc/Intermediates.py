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
from .timings import Timer
from .functions import evaluate


class Intermediates(dict):
    """
    TODO docme
    """
    def __init__(self):
        super().__init__()
        self.timer = Timer()
        # TODO Make caching configurable ??

    def push(self, key, expression_or_function):
        """
        Register an intermediate using a particular key.
        The second argument can either be an expression to evaluate or
        a function to call for obtaining the evaluated intermediate tensor.
        This will be done instantly in case the intermediate is not yet known
        and the result returned.
        """
        if not self.__contains__(key):
            with self.timer.record(key):
                if callable(expression_or_function):
                    tensor = evaluate(expression_or_function())
                else:
                    tensor = evaluate(expression_or_function)
            self.__setitem__(key, tensor)
        return self.__getitem__(key)

    def __getattr__(self, key):
        if self.__contains__(key):
            return self.__getitem__(key)
        else:
            raise AttributeError
