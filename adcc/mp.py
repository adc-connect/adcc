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
from adcc import block as b
from adcc.functions import direct_sum, einsum

from .Intermediates import register_as_intermediate

# TODO Maybe merge this with LazyMp ??


def compute_mp2_diffdm(hf, mp, space, apply_cvs=None):
    """Expressions to compute the MP2 difference density matrix

    If `apply_cvs` is `False` then the CVS approximation will not be
    applied during the density matrix computation even if it is
    applied in the hf reference. This is done in order to be
    able to obtain the *full* MP2 difference density (needed for
    MP properties and MP energies), while keeping the CVS is needed
    for the CVS-ADC(3) matrix for example.
    """
    if apply_cvs is None:
        apply_cvs = hf.has_core_occupied_space
    # elif apply_cvs is False:
    #     raise NotImplementedError("Not applying the CVS if reference has "
    #                               "it is not yet implemented")
    # assert not hf.has_core_occupied_space or apply_cvs

    # TODO: too much if/else..
    if apply_cvs or not hf.has_core_occupied_space:
        t2 = mp.t2(b.oovv)
        if space == "oo":
            return -0.5 * einsum("ikab,jkab->ij", t2, t2)
        elif space == "ov":
            df = direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
            return -0.5 * (+ einsum("ijbc,jabc->ia", t2, hf.ovvv)
                           + einsum("jkib,jkab->ia", hf.ooov, t2)) / df
        elif space == "vv":
            return 0.5 * einsum("ijac,ijbc->ab", t2, t2)
        else:
            raise NotImplementedError(f"Space {space} not implemented.")
    else:
        if space == "cc":
            return -0.5 * (+ einsum("kIab,kJab->IJ", mp.t2oc, mp.t2oc)
                           + einsum('LIab,LJab->IJ', mp.t2cc, mp.t2cc))
        elif space == "co":
            return -0.5 * (+ einsum("kIab,kjab->Ij", mp.t2oc, mp.t2oo)
                           + einsum("ILab,jLab->Ij", mp.t2cc, mp.t2oc))
        elif space == "cv":
            return -0.5 * (
                - 1.0 * einsum("jIbc,jabc->Ia", mp.t2oc, hf.ovvv)
                + 1.0 * einsum("jkIb,jkab->Ia", hf.oocv, mp.t2oo)
                + 1.0 * einsum("jMIb,jMab->Ia", hf.occv, mp.t2oc)
                + 1.0 * einsum("ILbc,Labc->Ia", mp.t2cc, hf.cvvv)
                + 1.0 * einsum("kLIb,kLab->Ia", hf.occv, mp.t2oc)
                + 1.0 * einsum("LMIb,LMab->Ia", hf.cccv, mp.t2cc)
            ) / mp.df(b.cv)
        elif space == "oc":
            return -0.5 * (+ einsum("kiab,kJab->iJ", mp.t2oo, mp.t2oc)
                           + einsum("iLab,JLab->iJ", mp.t2oc, mp.t2cc))
        elif space == "oo":
            return -0.5 * (+ einsum("ikab,jkab->ij", mp.t2oo, mp.t2oo)
                           + einsum("iLab,jLab->ij", mp.t2oc, mp.t2oc))
        elif space == "ov":
            return -0.5 * (
                + einsum("ijbc,jabc->ia", mp.t2oo, hf.ovvv)
                + einsum("jkib,jkab->ia", hf.ooov, mp.t2oo)
                + einsum("jMib,jMab->ia", hf.ocov, mp.t2oc)
                + einsum("iLbc,Labc->ia", mp.t2oc, hf.cvvv)
                + einsum("kLib,kLab->ia", hf.ocov, mp.t2oc)
                + einsum("iMLb,LMab->ia", hf.occv, mp.t2cc)
                - einsum("iLMb,LMab->ia", hf.occv, mp.t2cc)
            ) / mp.df(b.ov)
        elif space == "vv":
            return (
                + 0.5 * einsum("klac,klbc->ab", mp.t2oo, mp.t2oo)
                + 0.5 * einsum("IJac,IJbc->ab", mp.t2cc, mp.t2cc)
                + 1.0 * einsum("kJac,kJbc->ab", mp.t2oc, mp.t2oc)
            )
        else:
            raise NotImplementedError(f"Space {space} not implemented.")


#
# Register cvs_p0 intermedites
#
@register_as_intermediate
def cvs_p0_oo(hf, mp, intermediates):
    return compute_mp2_diffdm(hf, mp, "oo", apply_cvs=True)


@register_as_intermediate
def cvs_p0_ov(hf, mp, intermediates):
    return compute_mp2_diffdm(hf, mp, "ov", apply_cvs=True)


@register_as_intermediate
def cvs_p0_vv(hf, mp, intermediates):
    return compute_mp2_diffdm(hf, mp, "vv", apply_cvs=True)
