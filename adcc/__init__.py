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
import sys
import warnings

from libadcc import HartreeFockProvider, get_n_threads, set_n_threads

from .LazyMp import LazyMp
from .Tensor import Tensor
from .Symmetry import Symmetry
from .MoSpaces import MoSpaces
from .AdcMatrix import AdcBlockView, AdcMatrix
from .AdcMethod import AdcMethod
from .functions import (contract, copy, direct_sum, dot, einsum, empty_like,
                        evaluate, lincomb, linear_combination, nosym_like,
                        ones_like, transpose, zeros_like)
from .memory_pool import memory_pool
from .State2States import State2States
from .ExcitedStates import ExcitedStates
from .Excitation import Excitation
from .ElectronicTransition import ElectronicTransition
from .DataHfProvider import DataHfProvider, DictHfProvider
from .ReferenceState import ReferenceState
from .AmplitudeVector import AmplitudeVector
from .OneParticleOperator import OneParticleOperator
from .opt_einsum_integration import register_with_opt_einsum

# This has to be the last set of import
from .guess import (guess_symmetries, guess_zero, guesses_any, guesses_singlet,
                    guesses_spin_flip, guesses_triplet)
from .workflow import run_adc
from .exceptions import InputError

__all__ = ["run_adc", "InputError", "AdcMatrix", "AdcBlockView",
           "AdcMethod", "Symmetry", "ReferenceState", "MoSpaces",
           "einsum", "contract", "copy", "dot", "empty_like", "evaluate",
           "lincomb", "nosym_like", "ones_like", "transpose",
           "linear_combination", "zeros_like", "direct_sum",
           "memory_pool", "set_n_threads", "get_n_threads", "AmplitudeVector",
           "HartreeFockProvider", "ExcitedStates", "State2States",
           "Excitation", "ElectronicTransition", "Tensor", "DictHfProvider",
           "DataHfProvider", "OneParticleOperator",
           "guesses_singlet", "guesses_triplet", "guesses_any",
           "guess_symmetries", "guesses_spin_flip", "guess_zero", "LazyMp",
           "adc0", "cis", "adc1", "adc2", "adc2x", "adc3",
           "cvs_adc0", "cvs_adc1", "cvs_adc2", "cvs_adc2x", "cvs_adc3",
           "banner"]

__version__ = "0.15.11"
__license__ = "GPL v3"
__url__ = "https://adc-connect.org"
__authors__ = ["Michael F. Herbst", "Maximilian Scheurer"]
__email__ = "developers@adc-connect.org"
__contributors__ = []


def with_runadc_doc(func):
    func.__doc__ = run_adc.__doc__
    return func


@with_runadc_doc
def adc0(*args, **kwargs):
    return run_adc(*args, **kwargs, method="adc0")


@with_runadc_doc
def cis(*args, **kwargs):
    warnings.warn("CIS is a hardly tested feature. Use with caution.")
    state = run_adc(*args, **kwargs, method="adc1")
    return ExcitedStates(state, property_method="adc0")


@with_runadc_doc
def adc1(*args, **kwargs):
    return run_adc(*args, **kwargs, method="adc1")


@with_runadc_doc
def adc2(*args, **kwargs):
    return run_adc(*args, **kwargs, method="adc2")


@with_runadc_doc
def adc2x(*args, **kwargs):
    return run_adc(*args, **kwargs, method="adc2x")


@with_runadc_doc
def adc3(*args, **kwargs):
    return run_adc(*args, **kwargs, method="adc3")


@with_runadc_doc
def cvs_adc0(*args, **kwargs):
    return run_adc(*args, **kwargs, method="cvs-adc0")


@with_runadc_doc
def cvs_adc1(*args, **kwargs):
    return run_adc(*args, **kwargs, method="cvs-adc1")


@with_runadc_doc
def cvs_adc2(*args, **kwargs):
    return run_adc(*args, **kwargs, method="cvs-adc2")


@with_runadc_doc
def cvs_adc2x(*args, **kwargs):
    return run_adc(*args, **kwargs, method="cvs-adc2x")


@with_runadc_doc
def cvs_adc3(*args, **kwargs):
    return run_adc(*args, **kwargs, method="cvs-adc3")


def banner(colour=sys.stdout.isatty()):
    """Return a nice banner describing adcc and its components

    The returned string contains version information, maintainer emails
    and references.

    Parameters
    ----------
    colour : bool
        Should colour be used in the print out
    """
    if colour:
        yellow = '\033[93m'
        green = '\033[92m'
        cyan = '\033[96m'
        white = '\033[0m'
    else:
        yellow = ''
        green = ''
        cyan = ''
        white = ''

    empty = "|" + 70 * " " + "|\n"
    string = "+" + 70 * "-" + "+\n"
    string += "|{0:^70s}|\n".format(
        "adcc:  Seamlessly connect your host program to ADC"
    ).replace("adcc", "adc" + yellow + "c" + white)
    string += "+" + 70 * "-" + "+\n"
    string += empty
    string += "|     version     " + green + f"{__version__:<52}" + white + " |\n"

    # Print authors as groups
    groups = []
    cbuffer = []
    for i, author in enumerate(__authors__):
        if len(", ".join(cbuffer) + author) + 2 <= 52:
            cbuffer.append(author)
        else:
            groups.append(cbuffer)
            cbuffer = [author]
    if cbuffer:
        groups.append(cbuffer)
    for i, buf in enumerate(groups):
        authors = "authors" if i == 0 else ""
        joined = ", ".join(buf)
        if i != len(groups) - 1:
            joined += ","
        string += f"|     {authors:8s}    {joined:<52} |\n"

    string += "|     citation    " + yellow + "DOI 10.1002/wcms.1462" + white
    string += 32 * " " + "|\n"
    string += f"|     website     {__url__:<52} |\n"
    string += "|     email       " + cyan + f"{__email__:<52}" + white + " |\n"
    string += empty
    string += "+" + 70 * "-" + "+"
    return string


register_with_opt_einsum()
