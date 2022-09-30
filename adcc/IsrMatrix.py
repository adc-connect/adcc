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
import libadcc

from .AdcMatrix import AdcMatrixlike
from .LazyMp import LazyMp
from .adc_pp import bmatrix as ppbmatrix
from .timings import Timer, timed_member_call
from .AdcMethod import AdcMethod
from .AmplitudeVector import AmplitudeVector


class IsrMatrix(AdcMatrixlike):
    # Default perturbation-theory orders for the matrix blocks (== standard ADC-PP).
    default_block_orders = {
        #             ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),
        "adc0":  dict(ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc1":  dict(ph_ph=1, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc2":  dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=0),     # noqa: E501
        "adc2x": dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=1),     # noqa: E501
        "adc3":  dict(ph_ph=3, ph_pphh=2,    pphh_ph=2,    pphh_pphh=1),     # noqa: E501
    }

    def __init__(self, method, hf_or_mp, dips, block_orders=None):
        """
        Initialise an ISR matrix of a given one-particle operator
        for the provided ADC method.

        Parameters
        ----------
        method : str or adcc.AdcMethod
            Method to use.
        hf_or_mp : adcc.ReferenceState or adcc.LazyMp
            HF reference or MP ground state.
        dips : adcc.OneParticleOperator or list of adcc.OneParticleOperator objects
            One-particle matrix elements associated with the dipole operator.
        block_orders : optional
            The order of perturbation theory to employ for each matrix block.
            If not set, defaults according to the selected ADC method are chosen.
        """
        if isinstance(hf_or_mp, (libadcc.ReferenceState,
                                 libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)
        if not isinstance(hf_or_mp, LazyMp):
            raise TypeError("hf_or_mp is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)

        if not isinstance(dips, list):
            self.dips = [dips]
        else:
            self.dips = dips

        self.timer = Timer()
        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = method.is_core_valence_separated
        self.ndim = 2
        self.extra_terms = []

        # Determine orders of PT in the blocks
        if block_orders is None:
            block_orders = self.default_block_orders[method.base_method.name]
        else:
            tmp_orders = self.default_block_orders[method.base_method.name].copy()
            tmp_orders.update(block_orders)
            block_orders = tmp_orders

        # Sanity checks on block_orders
        for block in block_orders.keys():
            if block not in ("ph_ph", "ph_pphh", "pphh_ph", "pphh_pphh"):
                raise ValueError(f"Invalid block order key: {block}")
        if block_orders["ph_pphh"] != block_orders["pphh_ph"]:
            raise ValueError("ph_pphh and pphh_ph should always have "
                             "the same order")
        if block_orders["ph_pphh"] is not None \
           and block_orders["pphh_pphh"] is None:
            raise ValueError("pphh_pphh cannot be None if ph_pphh isn't.")
        self.block_orders = block_orders

        # Build the blocks
        with self.timer.record("build"):
            variant = None
            if self.is_core_valence_separated:
                variant = "cvs"
            blocks = [{
                block: ppbmatrix.block(self.ground_state, dip, block.split("_"),
                                       order=order, variant=variant)
                for block, order in self.block_orders.items() if order is not None
            } for dip in self.dips]
            # TODO Rename to self.block in 0.16.0
            self.blocks_ph = [{
                b: bl[b].apply for b in bl
            } for bl in blocks]

    @timed_member_call()
    def matvec(self, v):
        """
        Compute the matrix-vector product of the ISR one-particle
        operator and return the result.
        """
        ret = [
            sum(block(v) for block in bl_ph.values())
            for bl_ph in self.blocks_ph
        ]
        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def rmatvec(self, v):
        # Hermitian operators
        if all(dip.is_symmetric for dip in self.dips):
            return self.matvec(v)
        else:
            diffv = [dip.ov + dip.vo.transpose((1, 0)) for dip in self.dips]
            # anti-Hermitian operators
            if all(dv.dot(dv) < 1e-12 for dv in diffv):
                return [
                    AmplitudeVector(ph=-1.0 * mv.ph, pphh=-1.0 * mv.pphh)
                    for mv in self.matvec(v)
                ]
            # operators without any symmetry
            else:
                return NotImplemented

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return [self.matvec(ov) for ov in other]
        return NotImplemented
