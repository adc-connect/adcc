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

# TODO Maybe merge this with LazyMp ??


def compute_mp2_diffdm(hf, mp, space, apply_cvs=None):
    """Expressions to compute the MP2 difference density matrix

    If `apply_cvs` is `False` than the CVS approximation will not be
    applied during the density matrix computation even if it is
    applied in the hf reference. This is done in order to be
    able to obtain the *full* MP2 difference density (needed for
    MP properties and MP energies), while keeping the CVS is needed
    for the CVS-ADC(3) matrix for example.
    """
    if apply_cvs is None:
        apply_cvs = hf.has_core_occupied_space
    elif apply_cvs is False:
        raise NotImplementedError("Not applying the CVS if reference has "
                                  "it is not yet implemented")

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
