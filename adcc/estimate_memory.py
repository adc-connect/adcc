#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import numpy as np

from .MoSpaces import MoSpaces
from .AdcMethod import AdcMethod


def estimate_n_floats(method, n_o, n_c, n_v, max_subspace=0,
                      restricted=False):
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)

    eriimport = "vv"
    count = {
        "oo": 0, "ov": 0, "vv": 0,
        "oooo": 0, "ooov": 0, "ovov": 0,
        "oovv": 0, "ovvv": 0, "vvvv": 0,
    }
    sizes = {
        "o": n_o,
        "c": n_c,
        "v": n_v,
        "oo": n_o * n_o / 2,
        "ov": n_o * n_o,
        "vv": n_v * n_v / 2,
        "oooo": (n_o**2 + n_o + 1) * (n_o**2 + n_o) / 8,
        "ooov": (n_o**2 + n_o) * n_o * n_v / 2,
        "oovv": (n_o**2 + n_o) * (n_v**2 + n_v) / 4,
        "ovov": (n_o * n_v + 1) * (n_o * n_v) / 2,
        "ovvv": n_o * n_v * (n_v**2 + n_v) / 2,
        "vvvv": (n_v**2 + n_v + 1) * (n_v**2 + n_v) / 8,
    }

    if method.is_core_valence_separated:
        raise NotImplementedError("CVS not yet implemented")
    else:
        count["oo"] += 4  # Fock + coefficients + TDMs
        count["ov"] += 6  # Fock + coefficients + TDMs + df
        count["vv"] += 4  # Fock + coefficients + TDMs

    # Computed states (singles block)
    count["ov"] += max_subspace

    if method.level >= 1:
        count["oo"] += 1  # MP2 diffdm
        count["ov"] += 1  # MP2 diffdm
        count["vv"] += 1  # MP2 diffdm

        count["ooov"] += 1  # ERI
        count["oovv"] += 2  # ERI + T2
        count["ovov"] += 1  # ERI
        count["ovvv"] += 1  # ERI
        eriimport = "ovvv"

    if method.level >= 2:
        count["oooo"] += 1  # ERI
        count["vvvv"] += 1  # ERI
        count["oovv"] += 4  # t2eri + TD2

        # Even though vvvv is needed, the import
        # relevant during iteration is ovvv for ADC(2)
        eriimport = "ovvv"

        # Computed states (doubles block)
        count["ovov"] += max_subspace

    if method.name in ["adc2", "adc2x"]:
        count["oo"] += 1  # I1 intermediate
        count["vv"] += 1  # I2 intermediate
        eriimport = "vvvv"

    if method.level >= 3:
        count["ovov"] += 1  # M11 intermediate
        count["ovvv"] += 3  # t2eri + Pia intermediate
        count["ooov"] += 3  # t2eri + Pib intermediate

    spinfac2 = 1  # Spin factor for rank 2 tensors
    spinfac4 = 1  # for rank 4 tensors
    if restricted:
        spinfac2 = 1 / 4   # Only 1 out of 4 spin-blocks stored
        spinfac4 = 3 / 16  # Only 3 out of 16 needed for rank 4

    n_floats = 0
    n_floats += spinfac2 * sum(count[key] * sizes[key]
                               for key in ["oo", "ov", "vv"])
    n_floats += spinfac4 * sum(count[key] * sizes[key]
                               for key in ["oooo", "ooov", "oovv", "ovov",
                                           "ovvv", "vvvv"])

    # The EriBuilder import creates an overhead:
    n_floats += spinfac4 * np.prod([sizes[c] for c in eriimport])
    return n_floats


def estimate_minimal_memory(data_or_matrix, method, core_orbitals=None,
                            frozen_core=None, frozen_virtual=None,
                            max_subspace=None, n_singlets=0, n_triplets=0,
                            n_states=0, n_spin_flip=0, **kwargs):
    """
    Provide a crude estimate of the lower bound of required memory in bytes
    when running the :py:func:`adcc.run_adc` method or any
    other of :ref:`adcn-methods` with the passed keyword arguments
    and computing transition properties (such as the oscillator
    strengths) afterwards.

    The value returned is only a rough estimate of the **minimal**
    storage required to keep all computed objects in memory.
    In actual calculations this value **will most likely be overshot**.
    This values is only intended to serve as a quick check the user
    did not make an error with the passed input parameters, requesting
    a computation clearly overshooting the capacities of his machine.

    For the list of kwargs see :py:func:`adcc.run_adc`. Only rudimentary
    checking is performed on the sanity of the passed arguments.
    """
    if hasattr(data_or_matrix, "mospaces"):
        mospaces = data_or_matrix.mospaces
    else:
        mospaces = MoSpaces(data_or_matrix, core_orbitals=core_orbitals,
                            frozen_core=frozen_core,
                            frozen_virtual=frozen_virtual)

    n_c = 0
    if mospaces.has_core_occupied_space:
        n_c = mospaces.n_orbs("o2")
    n_o = mospaces.n_orbs("o1")
    n_v = mospaces.n_orbs("v1")

    # Estimate minimal number of floats to store, expand to bytes and return
    n_states = max(n_singlets, n_triplets, n_states, n_spin_flip)
    max_subspace = 10 * n_states
    n_floats = estimate_n_floats(method, n_o, n_c, n_v, max_subspace,
                                 restricted=mospaces.restricted)
    byte_per_float = 8  # 64bit = double
    return byte_per_float * n_floats
