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

from .LazyMp import LazyMp
from .Tensor import Tensor
from .Symmetry import Symmetry
from .AdcMatrix import AdcMatrix, AdcMatrixlike
from .AdcMethod import AdcMethod
from .functions import (add, contract, copy, divide, dot, empty_like,
                        linear_combination, multiply, nosym_like, ones_like,
                        subtract, transpose, zeros_like)
from .memory_pool import memory_pool
from .AdcBlockView import AdcBlockView
from .ExcitedStates import ExcitedStates
from .DataHfProvider import DataHfProvider, DictHfProvider
from .ReferenceState import ReferenceState
from .caching_policy import DefaultCachingPolicy, GatherStatisticsPolicy
from .AmplitudeVector import AmplitudeVector
from .OneParticleOperator import OneParticleOperator

from libadcc import HartreeFockProvider, get_n_threads, set_n_threads

# This has to be the last set of import
from .guess import (guess_symmetries, guess_zero, guesses_any, guesses_singlet,
                    guesses_spin_flip, guesses_triplet)
from .workflow import run_adc

__all__ = ["run_adc", "AdcMatrix", "AdcBlockView", "AdcMatrixlike", "AdcMethod",
           "Symmetry", "ReferenceState",
           "add", "contract", "copy", "divide", "dot", "empty_like",
           "linear_combination", "multiply", "nosym_like", "ones_like",
           "subtract", "transpose", "zeros_like",
           "memory_pool", "set_n_threads", "get_n_threads", "AmplitudeVector",
           "HartreeFockProvider", "ExcitedStates",
           "Tensor", "DictHfProvider", "DataHfProvider", "OneParticleOperator",
           "guesses_singlet", "guesses_triplet", "guesses_any",
           "guess_symmetries", "guesses_spin_flip", "guess_zero",
           "DefaultCachingPolicy", "GatherStatisticsPolicy", "LazyMp",
           "adc0", "adc1", "adc2", "adc2x", "adc3",
           "cvs_adc0", "cvs_adc1", "cvs_adc2", "cvs_adc2x", "cvs_adc3",
           "banner"]

__version__ = "0.14.1"
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


def banner(colour=sys.stdout.isatty(), show_doi=True, show_website=True):
    """Return a nice banner describing adcc and its components

    The returned string contains version information, maintainer emails
    and references.

    Parameters
    ----------
    colour : bool
        Should colour be used in the print out
    show_doi : bool
        Should DOI and publication information be printed.
    show_website : bool
        Should a website for each project be printed.

    """
    import libadcc as adccore

    if colour:
        yellow = '\033[93m'
        green = '\033[92m'
        cyan = '\033[96m'
        grey = '\033[38;5;248m'
        white = '\033[0m'
    else:
        yellow = ''
        green = ''
        cyan = ''
        grey = ''
        white = ''

    empty = "|" + 78 * " " + "|\n"
    maxlen = max(7, max(len(comp["name"]) for comp in adccore.__components__))

    def string_component(name, version, authors=None, description=None,
                         email=None, doi=None, website=None, licence=None):
        fmt = "|   " + green + "{0:<" + str(maxlen) + "s}" + white
        fmt += "  {1:8s}  " + yellow + "{2:<" + str(62 - maxlen) + "}"
        fmt += white + " |"
        fmt_email = "|     " + maxlen * " " + "{0:8s}  " + cyan
        fmt_email += "{1:<" + str(62 - maxlen) + "}" + white + " |"
        fmt_cite = "|     " + maxlen * " " + "{0:8s}  " + grey
        fmt_cite += "{1:<" + str(62 - maxlen) + "}" + white + " |"
        fmt_other = "|     " + maxlen * " " + "{0:8s}  {1:<"
        fmt_other += str(62 - maxlen) + "} |"

        string = fmt.format(name, "version", version) + "\n"
        if authors:
            if isinstance(authors, str):
                authors = authors.replace(" and ", ", ").split(", ")

            groups = []
            cbuffer = []
            for i, author in enumerate(authors):
                if len(", ".join(cbuffer) + author) + 2 <= 62 - maxlen:
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
                string += fmt_other.format(authors, joined) + "\n"
        if doi and show_doi:
            string += fmt_cite.format("citation", "DOI " + doi) + "\n"
        if website and show_website:
            string += fmt_other.format("website", website) + "\n"
        if email:
            string += fmt_email.format("email", email) + "\n"
        return string

    string = "+" + 78 * "-" + "+\n"
    string += "|{0:^78s}|\n".format(
        "adcc:  Seamlessly connect your host program to ADC"
    )
    string += "+" + 78 * "-" + "+\n"
    string += empty
    string += string_component("adcc", __version__, __authors__,
                               email=__email__, licence=__license__,
                               website=__url__, doi="10.1002/wcms.1462")
    string += empty
    bt = ""
    if adccore.__build_type__ not in ["Release"]:
        bt = "  " + adccore.__build_type__
    string += string_component("adccore", adccore.__version__ + bt,
                               adccore.__authors__, licence=__license__)
    string += empty

    string += "+{:-^78s}+\n".format("  Integrated third-party components  ")
    for comp in adccore.__components__:
        if show_doi or show_website:
            string += empty
        string += string_component(**comp)

    string += empty
    string += "+" + 78 * "-" + "+"
    return string
