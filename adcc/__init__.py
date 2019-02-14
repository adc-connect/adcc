#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

from libadcc import Tensor, HartreeFockSolution_i, HartreeFockProvider
from .functions import copy, empty_like, zeros_like, ones_like, nosym_like, dot
from .functions import linear_combination, divide, add, subtract, multiply
from .functions import contract, transpose
from .state_densities import attach_state_densities
from .AmplitudeVector import AmplitudeVector
from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .thread_pool import thread_pool
from .memory_pool import memory_pool
from .tmp_run_prelim import tmp_run_prelim
from .run_adc import run_adc

__version__ = "0.6.2"
__licence__ = "LGPL v3"
__authors__ = "Michael F. Herbst and Maximilian Scheurer"
__email__ = "info@michael-herbst.com"
# feel free to add your name above if you commit something


def adc2(*args, **kwargs):
    return run_adc("adc2", *args, **kwargs)


def adc2x(*args, **kwargs):
    return run_adc("adc2x", *args, **kwargs)


def adc3(*args, **kwargs):
    return run_adc("adc3", *args, **kwargs)


def cvs_adc2(*args, **kwargs):
    return run_adc("cvs-adc2", *args, **kwargs)


def cvs_adc2x(*args, **kwargs):
    return run_adc("cvs-adc2x", *args, **kwargs)


# Attach the same doc string as run_adc to other functions
adc2.__doc__ = run_adc.__doc__
adc2x.__doc__ = run_adc.__doc__
adc3.__doc__ = run_adc.__doc__
cvs_adc2.__doc__ = run_adc.__doc__
cvs_adc2x.__doc__ = run_adc.__doc__
