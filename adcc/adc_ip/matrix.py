# Matrix vector products for IP-ADC(2)


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
                         f"spaces={spaces} order={order} variant=variant")
    return globals()[fn](reference_state, ground_state, intermediates)


#
# 0th order main
#
def diagonal_phh_phh_0(hf):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum("-i-J+a->iJa", 
                     hf.foo.diagonal(), fCC.diagonal(), hf.fvv.diagonal())
    # Symmetrise like in PP-ADC?
    return AmplitudeVector(phh=res)
    
    
def block_phh_phh_0(hf, mp ,intermediates):
    # M_{22}
    def apply(ampl):
        return AmplitudeVector(phh=(
            + einsum("ab,ijb->ija", hf.fvv, ampl.phh)
            - 2 * einsum("ik,kja->ija", hf.foo, ampl.phh).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_phh_phh_0(hf))


#
# 1st order coupling
#
def block_h_phh_1(hf, mp, intermediates):
    # M_{12}
    def apply(ampl):
        return AmplitudeVector(h=
           + 1 / sqrt(2) * einsum("jkib,jkb->i", hf.ooov, ampl.phh))
    return AdcBlock(apply, 0)


def block_phh_h_1(hf, mp, intermediates):
    # M_{21}
    def apply(ampl):
        return AmplitudeVector(phh=
            + 1 / sqrt(2) * einsum("ijka,k->ija", hf.ooov, ampl.h))
    return AdcBlock(apply, 0)


#
# 2nd order main
#
def block_h_h_2(hf, mp, intermediates):
    # M_{11}
    i1 = intermediates.adc2_ip_i1
    diagonal = AmplitudeVector(h=i1.diagonal())
    def apply(ampl):
        return AmplitudeVector(h=einsum("ij,j->i", i1, ampl.h))
    return AdcBlock(apply, diagonal)


#
# Intermediates
#

@register_as_intermediate
def adc2_ip_i1(hf, mp, intermediates):
    return - hf.foo + 0.5 * einsum("ikab,jkab->ij", 
                                   mp.t2oo, hf.oovv).symmetrise()