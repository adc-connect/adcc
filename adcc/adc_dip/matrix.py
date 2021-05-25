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
from math import sqrt
from collections import namedtuple

from adcc import block as b
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate
from adcc.AmplitudeVector import AmplitudeVector

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
    and the perturbation theory order for the block, variant is "cvs" or sth
    like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied_space,
    while one would probably want in the long run that one can have an "o2" space,
    but not do CVS
    """
    raise NotImplemented
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError("Cannot run a general (non-core-valence approximated) "
                         "ADC method on top of a ground state with a "
                         "core-valence separation.")
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError("Cannot run a core-valence approximated ADC method on "
                         "top of a ground state without a "
                         "core-valence separation.")

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError("Could not dispatch: "
                         f"spaces={spaces} order={order} variant={variant}")
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def block_hh_hh_0(hf, mp, intermediates):
    diagonal = AmplitudeVector(hh=direct_sum("-a-i->ia", hf.foo.diagonal(),
                                  hf.foo.diagonal()))

    return AdcBlock(lambda ampl: 0, diagonal)


def block_phhh_phhh_0(hf, mp, intermediates):
    df = direct_sum("a-i->ia",
                    hf.fvv.diagonal(),
                    hf.foo.diagonal())

    # Possible speed up trough symmetrisation possible, but not pursued
    # at the present moment.
    diagonal = AmplitudeVector(phhh=direct_sum("ij+ma->maij",
                                       direct_sum("-k-l->lk", hf.foo.diagonal(), hf.foo.diagonal()),
                                       df))

    return AdcBlock(lambda ampl: 0, diagonal)


#
# 0th order coupling
#
def block_hh_phhh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_phhh_hh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


#
# 1st order main
#
def block_hh_hh_1(hf, mp, intermediates):
    diagonal = AmplitudeVector(hh=(
        direct_sum("-i-k>ki",
                   hf.foo.diagonalize(), hf.foo.diagonalize()).symmetrise ()
        + 0.5 * hf.oooo.diagonalize()
    ))

    def apply(ampl):
        return AmplitudeVector(hh=(
            - einsum("ik,kj->ij", hf.foo, ampl.hh)
            - einsum("jk,ik->ij", hf.foo, ampl.hh)
            + 0.25 * einsum("ijkl,kl->ij", hf.oooo, ampl.hh)
        ))
    return AdcBlock(apply, diagonal)


#
# 1st order coupling
#
def block_hh_phhh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(hh=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.phhh).antisymmetrise(0, 1)
            + einsum("ijbc,jabc->ia", ampl.phhh, hf.ooov).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, 0)


def block_phhh_hh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(phhh=(
            + einsum("il,jkla->ijka", ampl.hh, hf.ooov).antisymmetrise(0, 1)
            - einsum("kila,lj->ijka", hf.ooov, ampl.hh).antisymmetrise(0, 1)
            + einsum("ijla,kl->ijka", hf.ooov, ampl.hh) # Possible symmetrisation?
            - einsum("ijla,lk->ijka", hf.ooov, ampl.hh)
        ))
    return AdcBlock(apply, 0)


#
# 2nd order main
#
def block_hh_hh_2(hf, mp, intermediates):
    # Not sure if this will work directly...
    itm = intermediates.adc2_itm

    diagonal = Amplitude(hh=(
        + direct_sum("-i-k->ki", hf.foo.diagonal(), hf.foo.diagonal())
        - hf.foooo.diagonal()
        - itm.diagonal()
        + 0.5 * einsum("ikab,jlab->ik", mp.t2oo, hf.oovv)
    ))

    def apply(ampl):
        return NotImplemented

    return AdcBlock(apply, diagonal)


#
# Intermediates
#
@register_as_intermediate
def adc2_itm(hf, mp, intermediates):
    return 2 * einsum("ikab,jkab->ij", mp.t2oo, hf.oovv).symmetrise(0, 1)

