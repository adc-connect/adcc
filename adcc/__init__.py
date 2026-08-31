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

from libadcc import HartreeFockProvider, get_n_threads, set_n_threads

from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod, IsrMethod
from .AmplitudeVector import AmplitudeVector
from .DataHfProvider import DataHfProvider, DictHfProvider
from .ElectronicTransition import ElectronicTransition
from .exceptions import InputError
from .Excitation import Excitation
from .ExcitedStates import ExcitedStates
from .functions import (
    copy,
    direct_sum,
    dot,
    einsum,
    empty_like,
    evaluate,
    lincomb,
    linear_combination,
    nosym_like,
    ones_like,
    transpose,
    zeros_like,
)

# This has to be the last set of import
from .guess import (
    guess_symmetries,
    guess_zero,
    guesses_any,
    guesses_singlet,
    guesses_spin_flip,
    guesses_triplet,
)
from .LazyMp import LazyMp
from .memory_pool import memory_pool
from .MoSpaces import MoSpaces
from .NParticleOperator import OperatorSymmetry
from .OneParticleDensity import OneParticleDensity
from .OneParticleOperator import OneParticleOperator
from .opt_einsum_integration import register_with_opt_einsum
from .ReferenceState import ReferenceState
from .State2States import State2States
from .Symmetry import Symmetry
from .Tensor import Tensor
from .TwoParticleDensity import TwoParticleDensity
from .TwoParticleOperator import TwoParticleOperator
from .workflow import run_adc

__all__ = [
    "AdcMatrix",
    "AdcMethod",
    "AmplitudeVector",
    "DataHfProvider",
    "DictHfProvider",
    "ElectronicTransition",
    "Excitation",
    "ExcitedStates",
    "HartreeFockProvider",
    "InputError",
    "IsrMethod",
    "LazyMp",
    "MoSpaces",
    "OneParticleDensity",
    "OneParticleOperator",
    "OperatorSymmetry",
    "ReferenceState",
    "State2States",
    "Symmetry",
    "Tensor",
    "TwoParticleDensity",
    "TwoParticleOperator",
    "adc0",
    "adc1",
    "adc2",
    "adc2x",
    "adc3",
    "banner",
    "cis",
    "copy",
    "cvs_adc0",
    "cvs_adc1",
    "cvs_adc2",
    "cvs_adc2x",
    "cvs_adc3",
    "direct_sum",
    "dot",
    "einsum",
    "empty_like",
    "evaluate",
    "get_n_threads",
    "guess_symmetries",
    "guess_zero",
    "guesses_any",
    "guesses_singlet",
    "guesses_spin_flip",
    "guesses_triplet",
    "lincomb",
    "linear_combination",
    "memory_pool",
    "nosym_like",
    "ones_like",
    "run_adc",
    "set_n_threads",
    "transpose",
    "zeros_like",
]

__version__ = "0.18.0"
__license__ = "GPL v3"
__url__ = "https://adc-connect.org"
__authors__ = ["Michael F. Herbst", "Maximilian Scheurer", "Jonas Leitner",
               "Antonia Papapostolou", "Friederike Schneider",
               "Adrian L. Dempwolff", "Adrian J. Müller"]
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
    state = run_adc(*args, **kwargs, method="adc1")
    return ExcitedStates(state, property_method="isr0")


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
def adc4(*args, **kwargs):
    return run_adc(*args, **kwargs, method="adc4")


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
    string += "|{:^70s}|\n".format(
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
