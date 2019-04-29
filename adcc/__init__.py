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
from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .functions import (add, contract, copy, divide, dot, empty_like,
                        linear_combination, multiply, nosym_like, ones_like,
                        subtract, transpose, zeros_like)
from .memory_pool import memory_pool
from .thread_pool import thread_pool
from .caching_policy import DefaultCachingPolicy, GatherStatisticsPolicy
from .AmplitudeVector import AmplitudeVector
from .state_densities import attach_state_densities
from .tmp_build_reference_state import tmp_build_reference_state

from libadcc import HartreeFockProvider, HartreeFockSolution_i, LazyMp, Tensor

# This has to be the last set of import
from .guess import (guess_zero, guesses_any, guesses_singlet,
                    guesses_spin_flip, guesses_triplet)
from .run_adc import run_adc

__all__ = ["run_adc", "AdcMatrix", "AdcMethod",
           "add", "contract", "copy", "divide", "dot", "empty_like",
           "linear_combination", "multiply", "nosym_like", "ones_like",
           "subtract", "transpose", "zeros_like",
           "memory_pool", "thread_pool", "AmplitudeVector",
           "attach_state_densities", "tmp_build_reference_state",
           "HartreeFockProvider", "HartreeFockSolution_i", "Tensor",
           "guesses_singlet", "guesses_triplet", "guesses_any",
           "guesses_spin_flip", "guess_zero", "DefaultCachingPolicy",
           "GatherStatisticsPolicy", "LazyMp",
           "adc0", "adc1", "adc2", "adc2x", "adc3",
           "cvs_adc0", "cvs_adc1", "cvs_adc2", "cvs_adc2x",
           "banner"]

__version__ = "0.9.2"
__licence__ = "LGPL v3"
__authors__ = "Michael F. Herbst and Maximilian Scheurer"
__email__ = "info@michael-herbst.com"
# feel free to add your name above


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


def banner(colour=True):
    """
    Return a string, which is a descriptive banner for this
    adcc version, its components and the respective authors.
    """
    import libadcc as adccore

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

    empty = "|" + 78 * " " + "|\n"
    maxlen = max(7, max(len(comp["name"]) for comp in adccore.__components__))

    def string_component(name, version, authors=None, description=None,
                         email=None):
        fmt = "|    " + green + "{0:<" + str(maxlen) + "s}" + white
        fmt += "  {1:7s}  " + yellow + "{2:<" + str(62 - maxlen) + "}"
        fmt += white + " |"
        fmt_author = "|      " + maxlen * " " + "{0:7s}  {1:<"
        fmt_author += str(62 - maxlen) + "} |"
        fmt_email = "|      " + maxlen * " " + "{0:7s}  " + cyan
        fmt_email += "{1:<" + str(62 - maxlen) + "}" + white + " |"

        string = fmt.format(name, "version", version) + "\n"
        if authors:
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
                string += fmt_author.format(authors, joined) + "\n"
        if email:
            string += fmt_email.format("email", email) + "\n"
        return string

    string = "+" + 78 * "-" + "+\n"
    string += "|{0:^78s}|\n".format(
        "adcc:  Seamlessly connect your host program to ADC"
    )
    string += "+" + 78 * "-" + "+\n"
    string += string_component("adcc", __version__, __authors__,
                               email=__email__) + empty
    string += string_component("adccore", adccore.__version__,
                               adccore.__authors__) + empty

    string += "| Interfaced and integrated components: " + 39 * " " + "|\n"
    for comp in adccore.__components__:
        string += string_component(**comp)

    string += "+" + 78 * "-" + "+"
    return string
