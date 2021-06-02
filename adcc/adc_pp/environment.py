#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2021 by the adcc authors
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
from adcc import OneParticleOperator, AmplitudeVector
from .matrix import AdcBlock


def block_ph_ph_0_pe(hf, mp, intermediates):
    """
    Constructs an :py:class:`AdcBlock` that describes the
    linear response coupling to the polarizable environment
    from PE via a CIS-like transition density as described
    in 10.1021/ct300763v, eq 63. Since the contribution
    depends on the input amplitude itself,
    a diagonal term cannot be formulated.
    """
    op = hf.operators

    def apply(ampl):
        tdm = OneParticleOperator(mp, is_symmetric=False)
        tdm.vo = ampl.ph.transpose()
        vpe = op.pe_induction_elec(tdm)
        return AmplitudeVector(ph=vpe.ov)
    return AdcBlock(apply, 0)
