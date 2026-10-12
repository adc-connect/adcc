## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
from collections import namedtuple
from math import sqrt

from adcc import block as b
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate

__all__ = ["block"]

# TODO One thing one could still do to improve timings is implement a "fast einsum"
#      that does not call opt_einsum, but directly dispatches to libadcc. This could
#      lower the call overhead in the applies for the cases where we have only a
#      trivial einsum to do. For the moment I'm not convinced that is worth the
#      effort ... I suppose it only makes a difference for the cheaper ADC variants
#      (ADC(0), ADC(1), CVS-ADC(0-2)-x), but then on the other hand they are not
#      really so much our focus.


#
# Dispatch routine
#
"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix. `diagonal` is an `AmplitudeVector`
containing the expression to the diagonal of the ADC matrix from this block.
"""
AdcBlock = namedtuple("AdcBlock", ["apply", "diagonal"])


def block(ground_state, spaces, order, variant=None, intermediates=None):
    """
    Gets ground state, potentially intermediates, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied_space,
    while one would probably want in the long run that one can have an "o2" space,
    but not do CVS.
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError(
            "Cannot run a general (non-core-valence approximated) "
            "ADC method on top of a ground state with a "
            "core-valence separation."
        )
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError(
            "Cannot run a core-valence approximated ADC method on "
            "top of a ground state without a "
            "core-valence separation."
        )

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError(
            "Could not dispatch: "
            f"spaces={spaces} order={order} variant={variant}. "
            "Probably the secular matrix is not implemented for "
            "the requested method."
        )
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def block_ph_ph_0(hf, mp, intermediates):
    pass


block_cvs_ph_ph_0 = block_ph_ph_0


def diagonal_pphh_pphh_0(hf):
    pass


def block_pphh_pphh_0(hf, mp, intermediates):
    pass


def block_ppphhh_ppphhh_0(hf, mp, intermediates):
    pass


#
# 0th order coupling
#
def block_ph_pphh_0(hf, mp, intermediates):
    pass


def block_pphh_ph_0(hf, mp, intermediates):
    pass


def block_ph_ppphhh_0(hf, mp, intermediates):
    pass


def block_ppphhh_ph_0(hf, mp, intermediates):
    pass


def block_pphh_ppphhh_0(hf, mp, intermediates):
    pass


def block_ppphhh_pphh_0(hf, mp, intermediates):
    pass


#
# 1st order main
#
def block_ph_ph_1(hf, mp, intermediates):
    pass


def block_pphh_pphh_1(hf, mp, intermediates):
    pass


#
# 1st order coupling
#
def block_ph_pphh_1(hf, mp, intermediates):
    pass


def block_cvs_ph_pphh_1(hf, mp, intermediates):
    pass


def block_pphh_ph_1(hf, mp, intermediates):
    pass


def block_ph_ppphhh_1(hf, mp, intermediates):
    pass


def block_ppphhh_ph_1(hf, mp, intermediates):
    pass


def block_pphh_ppphhh_1(hf, mp, intermediates):
    pass


def block_ppphhh_pphh_1(hf, mp, intermediates):
    pass


#
# 2nd order main
#
def block_ph_ph_2(hf, mp, intermediates):
    pass


def block_pphh_pphh_2(hf, mp, intermediates):
    pass


#
# 2nd order coupling
#
def block_ph_pphh_2(hf, mp, intermediates):
    pass


def block_pphh_ph_2(hf, mp, intermediates):
    pass


def block_ph_ppphhh_2(hf, mp, intermediates):
    pass


def block_ppphhh_ph_2(hf, mp, intermediates):
    pass


#
# 3rd order main
#
def block_ph_ph_3(hf, mp, intermediates):
    pass


#
# 3rd order coupling
#
def block_ph_pphh_3(hf, mp, intermediates):
    pass


def block_pphh_ph_3(hf, mp, intermediates):
    pass


#
# 4th order main
#
def block_ph_ph_4(hf, mp, intermediates):
    pass
