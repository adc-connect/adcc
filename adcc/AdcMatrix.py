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
from os import name
from numpy.lib.function_base import blackman
import libadcc
import warnings
import numpy as np

from .LazyMp import LazyMp
from .adc_pp import matrix as ppmatrix
from .timings import Timer, timed_member_call
from .AdcMethod import AdcMethod
from .Intermediates import Intermediates
from .AmplitudeVector import QED_AmplitudeVector, AmplitudeVector, gs_vec


class AdcMatrixlike:
    """
    Base class marker for all objects like ADC matrices.
    """
    pass


class AdcMatrix(AdcMatrixlike):
    # Default perturbation-theory orders for the matrix blocks (== standard ADC-PP).
    default_block_orders = {
        #             ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),
        #"adc0":  dict(gs_gs=0, gs_ph=0, ph_gs=0, ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc0":  dict(ph_gs=0, ph_ph=0, pphh_gs=None, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc1":  dict(ph_gs=1, ph_ph=1, pphh_gs=None, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc2":  dict(ph_gs=2, ph_ph=2, pphh_gs=1   , ph_pphh=1,    pphh_ph=1,    pphh_pphh=0),     # noqa: E501
        "adc2x": dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=1),     # noqa: E501
        "adc3":  dict(ph_ph=3, ph_pphh=2,    pphh_ph=2,    pphh_pphh=1),     # noqa: E501
    }

    def __init__(self, method, hf_or_mp, block_orders=None, intermediates=None):
        """
        Initialise an ADC matrix.

        Parameters
        ----------
        method : str or AdcMethod
            Method to use.
        hf_or_mp : adcc.ReferenceState or adcc.LazyMp
            HF reference or MP ground state
        block_orders : optional
            The order of perturbation theory to employ for each matrix block.
            If not set, defaults according to the selected ADC method are chosen.
        intermediates : adcc.Intermediates or NoneType
            Allows to pass intermediates to re-use to this class.
        """
        if isinstance(hf_or_mp, (libadcc.ReferenceState,
                                 libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)
        if not isinstance(hf_or_mp, LazyMp):
            raise TypeError("mp_results is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)

        self.timer = Timer()
        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = method.is_core_valence_separated
        self.ndim = 2

        if method.base_method.name == "adc2x" or method.base_method.name == "adc3":
            NotImplementedError("Neither adc2x nor adc3 are implemented for QED-ADC")


        self.intermediates = intermediates
        if self.intermediates is None:
            self.intermediates = Intermediates(self.ground_state)


        # Determine orders of PT in the blocks
        if block_orders is None:
            block_orders = self.default_block_orders[method.base_method.name]
        else:
            tmp_orders = self.default_block_orders[method.base_method.name].copy()
            tmp_orders.update(block_orders)
            block_orders = tmp_orders

        # Sanity checks on block_orders
        for block in block_orders.keys():
            if block not in ("ph_gs", "gs_ph", "pphh_gs", "ph_ph", "ph_pphh", "pphh_ph", "pphh_pphh"):
                raise ValueError(f"Invalid block order key: {block}")
        if block_orders["ph_pphh"] != block_orders["pphh_ph"]:
            raise ValueError("ph_pphh and pphh_ph should always have "
                             "the same order")
        if block_orders["ph_pphh"] is not None \
           and block_orders["pphh_pphh"] is None:
            raise ValueError("pphh_pphh cannot be None if ph_pphh isn't.")
        self.block_orders = block_orders

        # Build the blocks and diagonals
        """#non-qed
        with self.timer.record("build"):
            variant = None
            if method.is_core_valence_separated:
                variant = "cvs"
            self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=order, intermediates=self.intermediates,
                                      variant=variant)
                for block, order in block_orders.items() if order is not None
            }
            #for block, order in block_orders.items():
            #    if order is not None:
            #        print(block, order) 
            self.__diagonal = sum(bl.diagonal for bl in self.blocks_ph.values()
                                  if bl.diagonal)
            self.__diagonal.evaluate()
            self.__init_space_data(self.__diagonal)
            #print("following is self.__init_space_data(self.__diagonal)")
            #print(self.__init_space_data(self.__diagonal))
        """

        # here we try to use the whole functionality given for the non-qed case (with the AmplitudeVector class)
        # and then use these to build the QED_AmplitudeVector class. Then we do something like
        # self.blocks_ph = QED_AmplitudeVector(val0, self.blocks_ph0, val1, self.blocks_ph1) and
        # self.diagonal = QED_AmplitudeVector(val0, self.__diagonal0, val1, self.__diagonal1)
        
        with self.timer.record("build"):
            variant = None
            if method.is_core_valence_separated:
                variant = "cvs"

            # first electronic part

            self.elec = AdcMatrix_submatrix(method, hf_or_mp, "elec")
            
            self.blocks_ph_00 = {  # TODO Rename to self.block in 0.16.0
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=order, intermediates=self.intermediates,
                                      variant=variant)
                for block, order in block_orders.items() if order is not None
            }
            """
            for block, order in block_orders.items():
                if order is not None:
                    print(block, order) 
            self.__diagonal_0 = sum(bl.diagonal for bl in self.blocks_ph_0.values()
                                  if bl.diagonal)
            self.__diagonal_0.evaluate()
            #self.__init_space_data_0(self.__diagonal_0)
            """

            # now for coupling parts

            self.elec_couple = AdcMatrix_submatrix(method, hf_or_mp, "elec_couple")
            
            self.blocks_ph_10 = {  # TODO Rename to self.block in 0.16.0
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=str(order) + "_couple", intermediates=self.intermediates,
                                      variant=variant)
                for block, order in block_orders.items() if order is not None
            }    

            self.phot_couple = AdcMatrix_submatrix(method, hf_or_mp, "phot_couple")
            
            self.blocks_ph_01 = {  # TODO Rename to self.block in 0.16.0
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=str(order) + "_phot_couple", intermediates=self.intermediates,
                                      variant=variant)
                for block, order in block_orders.items() if order is not None
            }

            # now for photonic part

            self.phot = AdcMatrix_submatrix(method, hf_or_mp, "phot")
            
            self.blocks_ph_11 = {  # TODO Rename to self.block in 0.16.0
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=str(order) + "_phot", intermediates=self.intermediates,
                                      variant=variant)
                for block, order in block_orders.items() if order is not None
            }
            """
            #for block, order in block_orders.items():
            #    if order is not None:
            #        print(block, order) 
            self.__diagonal_1 = sum(bl.diagonal for bl in self.blocks_ph_1.values()
                                  if bl.diagonal)
            self.__diagonal_1.evaluate()
            #self.__init_space_data_0(self.__diagonal_1)
            """

            if method.base_method.name == "adc2":
                self.phot2 = AdcMatrix_submatrix(method, hf_or_mp, "phot2")

                self.blocks_ph_22 = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot2", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }

                self.elec_couple_inner = AdcMatrix_submatrix(method, hf_or_mp, "elec_couple_inner")

                self.blocks_ph_21 = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_couple_inner", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }

                self.elec_couple_edge = AdcMatrix_submatrix(method, hf_or_mp, "elec_couple_edge")

                self.blocks_ph_20 = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_couple_edge", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }

                self.phot_couple_inner = AdcMatrix_submatrix(method, hf_or_mp, "phot_couple_inner")

                self.blocks_ph_12 = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot_couple_inner", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }

                self.phot_couple_edge = AdcMatrix_submatrix(method, hf_or_mp, "phot_couple_edge")

                self.blocks_ph_02 = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot_couple_edge", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }

            # build QED_AmplitudeVector
            
            self.blocks_ph_1_temp = {}
            for key in self.blocks_ph_10:
                self.blocks_ph_1_temp[key + "_couple"] = self.blocks_ph_10[key]
            for key in self.blocks_ph_01:
                self.blocks_ph_1_temp[key + "_phot_couple"] = self.blocks_ph_01[key]
            for key in self.blocks_ph_11:
                self.blocks_ph_1_temp[key + "_phot"] = self.blocks_ph_11[key]

            if hasattr(self.elec.diagonal(), "pphh"):
                for key in self.blocks_ph_20:
                    self.blocks_ph_1_temp[key + "_couple_edge"] = self.blocks_ph_20[key]
                for key in self.blocks_ph_21:
                    self.blocks_ph_1_temp[key + "_couple_inner"] = self.blocks_ph_21[key]
                for key in self.blocks_ph_02:
                    self.blocks_ph_1_temp[key + "_phot_couple_edge"] = self.blocks_ph_02[key]
                for key in self.blocks_ph_12:
                    self.blocks_ph_1_temp[key + "_phot_couple_inner"] = self.blocks_ph_12[key]
                for key in self.blocks_ph_22:
                    self.blocks_ph_1_temp[key + "_phot2"] = self.blocks_ph_22[key]
            self.blocks_ph = {**self.blocks_ph_00, **self.blocks_ph_1_temp}

            self.__diagonal_gs = 0 #sum(self.blocks_ph[block].diagonal for block in self.blocks_ph
                                   #  if "ph_gs" in block and not block.endswith("phot")) # coupling gs_gs blocks have diagonal = 0
            self.__diagonal_gs1 = sum(self.blocks_ph[block].diagonal for block in self.blocks_ph
                                    if "ph_gs" in block and block.endswith("phot"))
            

            #print("following should be the blocks, that make up the diagonal for gs_gs")
            #for block in self.blocks_ph:
            #    if "ph_gs" in block and block.endswith("phot"):
            #        print("this is the block ", block)
            #print(self.__diagonal_gs)
            #print(self.__diagonal_gs1)
            #self.blocks_ph = {**self.elec.blocks_ph, **self.blocks_ph_1_temp}
            #self.blocks_ph = QED_AmplitudeVector(0, self.elec.blocks_ph, 0, self.phot.blocks_ph)
            #self.__diagonal = QED_AmplitudeVector(0, self.elec.diagonal, 0, self.phot.diagonal)
            #self.__init_space_data_qed(self.elec.diagonal) # here we need to adapt the attributes, defined via this function
            if hasattr(self.elec.diagonal(), "pphh"): # this needs to be adapted, so that the diagonals for gs and gs1 are actually those from matrix.py,
            # but if this is only for the guesses, this should only be a perfomance issue and not relevant for the precision
                #print("from adcmatrix init diagonal has pphh part")
                self.__diagonal_gs2 = sum(self.blocks_ph[block].diagonal for block in self.blocks_ph
                                    if "ph_gs" in block and block.endswith("phot2"))

                self.__diagonal = QED_AmplitudeVector(self.__diagonal_gs, self.elec.diagonal().ph, self.elec.diagonal().pphh,
                                                    self.__diagonal_gs1, self.phot.diagonal().ph, self.phot.diagonal().pphh,
                                                    self.__diagonal_gs2, self.phot2.diagonal().ph, self.phot2.diagonal().pphh)
            else:
                #print("from adcmatrix init diagonal has no pphh part!!!!!!!!!!!!!!!!!!!")
                self.__diagonal = QED_AmplitudeVector(self.__diagonal_gs, self.elec.diagonal().ph, None,
                                                    self.__diagonal_gs1, self.phot.diagonal().ph, None)
            self.__init_space_data_qed(self.elec.diagonal()) # here we need to adapt the attributes, defined via this function
            # namely self.axis_spaces, self.axis_lengths, self.shape
            self.__diagonal.evaluate()
            print(self.elec.blocks_ph)
            print(self.blocks_ph)
            print(self.axis_spaces)
            print(self.axis_lengths)
            print(self.shape)
            print(self.elec.axis_lengths)
            print(self.elec.shape)
            #print("matvec has to be corrected, since .gs and .gs1 parts are not yet included, also remaining blocks have to included")
            


            """
            #self.blocks_ph = QED_AmplitudeVector(0, self.blocks_ph_0, 0, self.blocks_ph_1)
            self.blocks_ph_1_temp = {}
            for key in self.blocks_ph_1:
                self.blocks_ph_1_temp[key + "_phot"] = self.blocks_ph_1[key]
            self.blocks_ph = {**self.blocks_ph_0, **self.blocks_ph_1_temp}
            self.__diagonal = QED_AmplitudeVector(0, self.__diagonal_0, 0, self.__diagonal_1)
            self.__init_space_data_qed(self.__diagonal_0) # here we need to adapt the attributes, defined via this function
            # namely self.axis_spaces, self.axis_lengths, self.shape
            print(self.blocks_ph_0)
            print(self.blocks_ph_1)
            print(self.blocks_ph)
            print(self.axis_spaces)
            print(self.axis_lengths)
            print(self.shape)
            """
            

            # since the guesses need the whole matrix object as input, istead of just the diagonals, it is useful to define
            # self.elec and self.phot as the the corresponding original matrices
            # this is done in one class, since we also require the full matrix for the solver
       

    def __init_space_data_qed(self, diagonal0):
        self.__init_space_data(diagonal0)
        axis_spaces0 = self.axis_spaces # this (and the following) will be assigned in the submatrix class as well
        axis_lengths0 = self.axis_lengths
        shape0 = self.shape
        if hasattr(diagonal0, "pphh"):
            self.shape = ((shape0[0]+1) * 3, (shape0[0]+1) * 3)
        else:
            self.shape = ((shape0[0]+1) * 2, (shape0[0]+1) * 2)
        self.axis_spaces["gs"] = ["o0", "v0"] 
        self.axis_lengths["gs"] = 1
        # axis_spaces and axis_lengths both are dicts, referring to "ph", "pphh" as a key, so either make extra blocks here, or probably better
        # make a longer list in funciton 'axis_blocks', so that later everything for "ph" and "pphh" is just used twice
        # also it seems, that self.axis_lengths is not further used, but shape[0] is rather used for that, so not necessary to build self.axis_lengths correctly


    def __init_space_data(self, diagonal):
        """Update the cached data regarding the spaces of the ADC matrix"""
        self.axis_spaces = {}
        self.axis_lengths = {}
        for block in diagonal.blocks_ph:
            self.axis_spaces[block] = getattr(diagonal, block).subspaces
            self.axis_lengths[block] = np.prod([
                self.mospaces.n_orbs(sp) for sp in self.axis_spaces[block]
            ])
        self.shape = (sum(self.axis_lengths.values()),
                      sum(self.axis_lengths.values()))
        #print(self.axis_spaces)
        #print(self.axis_lengths)
        #print(self.shape)


    def __repr__(self):
        ret = f"AdcMatrix({self.method.name}, "
        for b, o in self.block_orders.items():
            ret += f"{b}={o}, "
        return ret + ")"

    def __len__(self):
        return self.shape[0]

    @property
    def blocks(self):
        # TODO Remove in 0.16.0
        return self.__diagonal.blocks

    def has_block(self, block):
        warnings.warn("The has_block function is deprecated and "
                      "will be removed in 0.16.0. "
                      "Use `in matrix.axis_blocks` in the future.")
        return self.block_spaces(block) is not None

    def block_spaces(self, block):
        warnings.warn("The block_spaces function is deprecated and "
                      "will be removed in 0.16.0. "
                      "Use `matrix.axis_spaces[block]` in the future.")
        return {
            "g": self.axis_spaces.get("gs", None),
            "s": self.axis_spaces.get("ph", None),
            "d": self.axis_spaces.get("pphh", None),
            "t": self.axis_spaces.get("ppphhh", None),
        }[block]

    #@property #non-qed
    #def axis_blocks(self):
        """
        Return the blocks used along one of the axes of the ADC matrix
        (e.g. ['ph', 'pphh']).
        """
    #    return list(self.axis_spaces.keys())

    @property
    def axis_blocks(self): # this is only for debugging and has not been adapted yet
        #print(list(self.axis_spaces.keys()) + list(self.axis_spaces.keys()))
        return list(self.axis_spaces.keys()) + list(self.axis_spaces.keys())

    def diagonal(self, block=None):
        """Return the diagonal of the ADC matrix"""
        if block is not None:
            warnings.warn("Support for the block argument will be dropped "
                          "in 0.16.0.")
            if block == "g":
                return self.__diagonal.gs 
            if block == "s":
                return self.__diagonal.ph
            if block == "d":
                return self.__diagonal.pphh
        return self.__diagonal

    def compute_apply(self, block, tensor):
        warnings.warn("The compute_apply function is deprecated and "
                      "will be removed in 0.16.0.")
        if block in ("gg", "gs", "sg" ,"ss", "sd", "ds", "dd"):
            warnings.warn("The singles-doubles interface is deprecated and "
                          "will be removed in 0.16.0.")
            block = {"gg": "gs_gs", "gs": "gs_ph", "sg": "ph_gs",
                     "ss": "ph_ph", "sd": "ph_pphh",
                     "ds": "pphh_ph", "dd": "pphh_pphh"}[block]
        return self.block_apply(block, tensor)

    def block_apply(self, block, tensor):
        """
        Compute the application of a block of the ADC matrix
        with another AmplitudeVector or Tensor. Non-matching blocks
        in the AmplitudeVector will be ignored.
        """
        if not isinstance(tensor, libadcc.Tensor):
            raise TypeError("tensor should be an adcc.Tensor")

        with self.timer.record(f"apply/{block}"):
            outblock, inblock = block.split("_")
            ampl = QED_AmplitudeVector(**{inblock: tensor})
            ret = self.blocks_ph[block].apply(ampl)
            return getattr(ret, outblock)

    @timed_member_call()
    def matvec(self, v): # for this sum the __add__ and/or __radd__ in QED_AmplitudeVector needs to be adjusted
        # maybe also this function has to be adapted
        # lets try differentiating via _phot or not in self.blocks_ph and then do the summation over the blocks,
        # which can then be forwarded to AmplitudeVector for ph and pphh and gs_vec for gs
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result.
        """
        #print("printing matvec stuff from AdcMatrix")

        #res = v.zeros_like()

        #elec1 = self.elec.matvec(v)
        #elec2 = self.phot_couple.matvec(v)
        #print(type(elec1.ph), elec1.ph)
        #print(type(elec2.ph), elec2.ph)

        # maybe also gs_ph and gs_pphh blocks can be included here, by adding them to the ph_ph and pphh_pphh blocks, respectively.
        # this should be possible, since we give v as ampl in matrix.py, so we can select e.g. ampl.gs1 in the ph_ph block.
        #print("this is v.pphh from matvec ", v.pphh)
        #try:
        #    self.phot_couple.matvec(v).ph + self.phot_couple_edge.matvec(v)
        #    print("works yeaaaaaaaaaaaaah")
        #except:
        #    print("god damn")
        #print("shape of phot couple edge", self.phot_couple_edge, type(self.phot_couple_edge))
        #print("shape of phot couple inner", self.phot_couple_inner.matvec(v), type(self.phot_couple_inner.matvec(v)))

        phot_couple_edge_with_doubles = AmplitudeVector(ph=self.phot_couple_edge.matvec(v), pphh=v.pphh.zeros_like())
        elec_couple_edge_with_doubles = AmplitudeVector(ph=self.elec_couple_edge.matvec(v), pphh=v.pphh.zeros_like())

        elec_part = self.elec.matvec(v) + self.phot_couple.matvec(v) + phot_couple_edge_with_doubles #self.phot_couple_edge.matvec(v)
        phot_part = self.elec_couple.matvec(v) + self.phot.matvec(v) + self.phot_couple_inner.matvec(v)
        #phot2_part = self.elec_couple_edge.matvec(v) + self.elec_couple_inner.matvec(v) + self.phot2.matvec(v)
        phot2_part = elec_couple_edge_with_doubles + self.elec_couple_inner.matvec(v) + self.phot2.matvec(v)
        gs_part = 0
        gs1_part = 0
        gs2_part = 0

        """
        elec_part = self.elec.matvec(v) + self.phot_couple.matvec(v)
        phot_part = self.elec_couple.matvec(v) + self.phot.matvec(v)
        gs_part = 0
        gs1_part = 0

        if "pphh" in elec_part.blocks_ph:
            gs2_part = 0
            elec_part = elec_part + self.phot_couple_edge.matvec(v)
            phot_part = phot_part + self.phot_couple_inner.matvec(v)
            phot2_part = self.elec_couple_edge.matvec(v) + self.elec_couple_inner.matvec(v) + self.phot2.matvec(v)
        """
        #else:
        #    elec_part = self.elec.matvec(v) + self.phot_couple.matvec(v)
        #    phot_part = self.elec_couple.matvec(v) + self.phot.matvec(v)

        for block in self.blocks_ph:
            if "gs" in block and not block.startswith("gs"):
                #print(block, type(self.blocks_ph[block].apply(v)), self.blocks_ph[block].apply(v))
                if "pphh" in elec_part.blocks_ph:
                    if "phot2" in block:
                        gs2_part += self.blocks_ph[block].apply(v)
                    if "phot_couple_edge" in block:
                        gs_part += self.blocks_ph[block].apply(v)
                    if "phot_couple_inner" in block:
                        gs1_part += self.blocks_ph[block].apply(v)
                    if "couple_edge" in block:
                        gs2_part += self.blocks_ph[block].apply(v)
                    if "couple_inner" in block:
                        gs2_part += self.blocks_ph[block].apply(v)
                if "phot_couple" in block:
                    gs_part += self.blocks_ph[block].apply(v) # check whether the if statements always grant the correct blocks!!!!!!!!!!!!!!!!!!!!!!!!!
                elif "phot" in block:
                    gs1_part += self.blocks_ph[block].apply(v)
                elif "couple" in block:
                    gs1_part += self.blocks_ph[block].apply(v)
                else: # elec
                    gs_part += self.blocks_ph[block].apply(v)
        """
            elif "gs" in block and block.startswith("gs"):
                if "phot_couple" in block:
                    elec_part += self.blocks_ph[block].apply(v)
                elif "phot" in block:
                    phot_part += self.blocks_ph[block].apply(v)
                elif "couple" in block:
                    phot_part += self.blocks_ph[block].apply(v)
                else: # elec
                    elec_part += self.blocks_ph[block].apply(v)
        """

        # try .gs and .gs1 parts with floats first, and then give them to the QED_AmplitudeVector in the end
        # if that doesnt work, use QED_AmplitudeVectors from the beginning, which then has to be implemented in matrix.py as well

        """
        for block in self.blocks_ph:
            if "gs" not in block:
                print(block)
                if block.endswith("_phot_couple"):
                    res.elec = sum(res.elec,self.blocks_ph[block].apply(v))
                elif block.endswith("_couple"):
                    res.phot = sum(res.phot,self.blocks_ph[block].apply(v))
                elif block.endswith("_phot"):
                    res.phot = sum(res.phot,self.blocks_ph[block].apply(v))
                elif block in self.blocks_ph_00:
                    print(type(res.elec), type(self.blocks_ph[block].apply(v)))
                    res.elec = sum(res.elec,self.blocks_ph[block].apply(v))
                else:
                    print("block {} has not been taken into account in matvec".format(block))
        """


        """
        self.blocks_ph_elec = {}
        self.blocks_ph_phot = {}
        self.blocks_ph_elec0 = {} # this is only necessary, if gs block part in ph/pphh space are in explicit blocks, and not e.g. in ph_ph block
        self.blocks_ph_phot0 = {} # this is only necessary, if gs block part in ph/pphh space are in explicit blocks, and not e.g. in ph_ph_phot block
        for block in self.blocks_ph:
            if block.endswith("_phot"):
                if not block.startswith("block_gs") and "gs" in block: # find all gs space blocks (with ampl.gs/ampl.gs1)
                    if self.blocks_ph[block].gs == None:
                        self.blocks_ph_phot0[block] = self.blocks_ph[block].gs1
                    else:
                        self.blocks_ph_phot0[block] = self.blocks_ph[block].gs
                else:
                    if self.blocks_ph[block].elec == None:
                        self.blocks_ph_phot[block] = self.blocks_ph[block].phot
                    else:
                        self.blocks_ph_phot[block] = self.blocks_ph[block].elec
                #print(block, self.blocks_ph[block])
            else:
                if not block.startswith("block_gs") and "gs" in block: # find all gs1 space blocks (with ampl.gs/ampl.gs1)
                    if self.blocks_ph[block].gs == None:
                        self.blocks_ph_elec0[block] = self.blocks_ph[block].gs1
                    else:
                        self.blocks_ph_elec0[block] = self.blocks_ph[block].gs
                else:
                    if self.blocks_ph[block].elec == None:
                        self.blocks_ph_elec[block] = self.blocks_ph[block].phot
                    else:
                        self.blocks_ph_elec[block] = self.blocks_ph[block].elec
                #print(block, self.blocks_ph[block])

        res = v.zeros_like()
        #print("res", res.gs, res.elec, res.gs1, res.phot)

        res.gs = sum(block.apply(v) for block in self.blocks_ph_elec.values())# if hasattr(block.apply(v), "gs")) # __add__ from gs_vec
        res.elec = sum(block.apply(v) for block in self.blocks_ph_elec.values())# if not hasattr(block.apply(v), "gs")) # __add__ from AmplitudeVector
        res.gs1 = sum(block.apply(v) for block in self.blocks_ph_phot.values())# if hasattr(block.apply(v), "gs")) # __add__ from gs_vec
        res.phot = sum(block.apply(v) for block in self.blocks_ph_phot.values())# if not hasattr(block.apply(v), "gs")) # __add__ from AmplitudeVector
        """

        #for block in self.blocks_ph_elec.values():
        #    print(type(block.apply(v)))
        #    print(block.apply(v))
        #for block in self.blocks_ph:
        #    print(block)
        """
        for block in self.blocks_ph_elec.values():
            #print(type(block.apply(v)))
            if hasattr(block.apply(v), "gs"): # for gs QED_AmplitudeVector, while for ph and pphh AmplitudeVector
                #print(block)
                if block.apply(v).gs != None:
                    #print(res.gs, block.apply(v).gs)
                    res.gs += block.apply(v).gs
                elif block.apply(v).gs1 != None:
                    res.gs += block.apply(v).gs1
            #else:
            #    if block.apply(v) != None:
            #        res.elec = sum(res.elec, block.apply(v))

        for block in self.blocks_ph_phot.values():
            if hasattr(block.apply(v), "gs"): # for gs QED_AmplitudeVector, while for ph and pphh AmplitudeVector
                #print(block)
                if block.apply(v).gs != None:
                    #print("printing line 405 now")
                    #print(res.gs1, block.apply(v).gs)
                    res.gs1 += block.apply(v).gs
                elif block.apply(v).gs1 != None:
                    res.gs1 += block.apply(v).gs1
            #else:
            #    if block.apply(v) != None:
            #        res.phot += block.apply(v)
        
        #return sum(block.apply(v) for block in self.blocks_ph.values())
        """
        #print(res.gs, res.elec, res.gs1, res.phot)
        if "pphh" in elec_part.blocks_ph:
            return QED_AmplitudeVector(gs_part, elec_part.ph, elec_part.pphh, gs1_part, phot_part.ph, phot_part.pphh,
                                         gs2_part, phot2_part.ph, phot2_part.pphh)
        else:
            return QED_AmplitudeVector(gs_part, elec_part.ph, None, gs1_part, phot_part.ph, None)

        


    def rmatvec(self, v):
        # ADC matrix is symmetric
        return self.matvec(v)

    def compute_matvec(self, ampl):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result.
        """
        warnings.warn("The compute_matvec function is deprecated and "
                      "will be removed in 0.16.0.")
        return self.matvec(ampl)

    def __matmul__(self, other):
        if isinstance(other, QED_AmplitudeVector):
            return self.matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, QED_AmplitudeVector) for elem in other):
                return [self.matvec(ov) for ov in other]
        return NotImplemented

    def block_view(self, block):
        """
        Return a view into the AdcMatrix that represents a single
        block of the matrix. Currently only diagonal blocks are supported.
        """
        b1, b2 = block.split("_")
        if b1 != b2:
            raise NotImplementedError("Off-diagonal block views not yet "
                                      "implemented.")
            # TODO For off-diagonal blocks we probably need a different
            #      data structure as the AdcMatrix class as these block
            #      are inherently different than an AdcMatrix (no Hermiticity
            #      for example) and basically they only need to support some
            #      form of matrix-vector product and some stastics like
            #      spaces and sizes etc.
        block_orders = {bl: None for bl in self.block_orders.keys()}
        block_orders[block] = self.block_orders[block]
        return AdcMatrix(self.method, self.ground_state,
                         block_orders=block_orders,
                         intermediates=self.intermediates)

    def construct_symmetrisation_for_blocks(self):
        """
        Construct the symmetrisation functions, which need to be
        applied to relevant blocks of an AmplitudeVector in order
        to symmetrise it to the right symmetry in order to be used
        with the various matrix-vector-products of this function.

        Most importantly the returned functions antisymmetrise
        the occupied and virtual parts of the doubles parts
        if this is sensible for the method behind this adcmatrix.

        Returns a dictionary block identifier -> function
        """
        ret = {}
        if self.is_core_valence_separated:
            # CVS doubles part is antisymmetric wrt. (i,K,a,b) <-> (i,K,b,a)
            ret["pphh"] = lambda v: v.antisymmetrise([(2, 3)])
        else:
            def symmetrise_generic_adc_doubles(invec):
                # doubles part is antisymmetric wrt. (i,j,a,b) <-> (i,j,b,a)
                scratch = invec.antisymmetrise([(2, 3)])
                # doubles part is symmetric wrt. (i,j,a,b) <-> (j,i,b,a)
                return scratch.symmetrise([(0, 1), (2, 3)])
            ret["pphh"] = symmetrise_generic_adc_doubles
        return ret

    def dense_basis(self, axis_blocks=None, ordering="adcc"):
        """
        Return the list of indices and their values
        of the dense basis representation

        ordering: adcc, spin, spatial
        """
        ret = []
        if axis_blocks is None:
            axis_blocks = self.axis_blocks
        if not isinstance(axis_blocks, list):
            axis_blocks = [axis_blocks]

        # Define function to impose the order in the basis
        if ordering == "adcc":
            def reduce_index(n_orbsa, idx):
                return idx, idx
        elif ordering == "spin":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spin, then by spatial
                return (is_beta, spatial)
        elif ordering == "spatial":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spatial, then by spin
                return (spatial, is_beta)

        if "gs" in axis_blocks:
            ret_g = []
            ret_g.append([(0, 0), 1])
            ret.extend(ret_g)

        if "ph" in axis_blocks:
            ret_s = []
            sp_s = self.axis_spaces["ph"]
            n_orbs_s = [self.mospaces.n_orbs(sp) for sp in sp_s]
            n_orbsa_s = [self.mospaces.n_orbs_alpha(sp) for sp in sp_s]
            for i in range(n_orbs_s[0]):
                for a in range(n_orbs_s[1]):
                    ret_s.append([((i, a), 1)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_s, idx) for idx, factor in x)
            ret_s.sort(key=sortfctn)
            ret_s.sort(key=sortfctn)
            ret.extend(ret_s)

        if "pphh" in axis_blocks:
            ret_d = []
            sp_d = self.axis_spaces["pphh"]
            n_orbsa_d = [self.mospaces.n_orbs_alpha(sp) for sp in sp_d]

            if sp_d[0] == sp_d[1] and sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / 2),
                               ((j, i, a, b), -1 / 2),
                               ((i, j, b, a), -1 / 2),
                               ((j, i, b, a), +1 / 2)]
                              for i in range(nso) for j in range(i)
                              for a in range(nsv) for b in range(a)])
            elif sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / np.sqrt(2)),
                               ((i, j, b, a), -1 / np.sqrt(2))]
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(a)])
            else:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                nsw = self.mospaces.n_orbs(sp_d[3])
                ret_d.append([((i, j, b, a), 1)
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(nsw)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_d, idx) for idx, factor in x)
            ret_d.sort(key=sortfctn)
            ret_d.sort(key=sortfctn)
            ret.extend(ret_d)

        if any(b not in ("gs" ,"ph", "pphh") for b in self.axis_blocks):
            raise NotImplementedError("Blocks other than gs, ph and pphh "
                                      "not implemented")
        return ret

    def to_ndarray(self, out=None): # this is not adapted to gs
        """
        Return the ADC matrix object as a dense numpy array. Converts the sparse
        internal representation of the ADC matrix to a dense matrix and return
        as a numpy array.

        Notes
        -----

        This method is only intended to be used for debugging and
        visualisation purposes as it involves computing a large amount of
        matrix-vector products and the returned array consumes a considerable
        amount of memory.

        The resulting matrix has no spin symmetry imposed, which means that
        its eigenspectrum may contain non-physical excitations (e.g. with linear
        combinations of α->β and α->α components in the excitation vector).

        This function has not been sufficiently tested to be considered stable.
        """
        # TODO Update to ph / pphh
        # TODO Still uses deprecated functions
        import tqdm

        from adcc import guess_zero

        # Get zero amplitude of the appropriate symmetry
        # (TODO: Only true for C1, where there is only a single irrep)
        ampl_zero = guess_zero(self)
        assert self.mospaces.point_group == "C1"

        # Build the shape of the returned array
        # Since the basis of the doubles block is not the unit vectors
        # this *not* equal to the shape of the AdcMatrix object
        basis = {b: self.dense_basis(b) for b in self.axis_blocks}
        mat_len = sum(len(basis[b]) for b in basis)

        if out is None:
            out = np.zeros((mat_len, mat_len))
        else:
            if out.shape != (mat_len, mat_len):
                raise ValueError("Output array has shape ({0:}, {1:}), but "
                                 "shape ({2:}, {2:}) is required."
                                 "".format(*out.shape, mat_len))
            out[:] = 0  # Zero all data in out.

        # Check for the cases actually implemented
        if any(b not in ("ph", "pphh") for b in self.axis_blocks):
            raise NotImplementedError("Blocks other than ph and pphh "
                                      "not implemented")
        if "ph" not in self.axis_blocks:
            raise NotImplementedError("Block 'ph' needs to be present")

        # Extract singles-singles block (contiguous)
        assert "ph" in self.axis_blocks
        n_orbs_ph = [self.mospaces.n_orbs(sp) for sp in self.axis_spaces["ph"]]
        n_ph = np.prod(n_orbs_ph)
        assert len(basis["ph"]) == n_ph
        view_ss = out[:n_ph, :n_ph].reshape(*n_orbs_ph, *n_orbs_ph)
        for i in range(n_orbs_ph[0]):
            for a in range(n_orbs_ph[1]):
                ampl = ampl_zero.copy()
                ampl.ph[i, a] = 1
                view_ss[:, :, i, a] = (self @ ampl).ph.to_ndarray()

        # Extract singles-doubles and doubles-doubles block
        if "pphh" in self.axis_blocks:
            assert self.axis_blocks == ["ph", "pphh"]
            view_sd = out[:n_ph, n_ph:].reshape(*n_orbs_ph, len(basis["pphh"]))
            view_dd = out[n_ph:, n_ph:]
            for j, bas1 in tqdm.tqdm(enumerate(basis["pphh"]),
                                     total=len(basis["pphh"])):
                ampl = ampl_zero.copy()
                for idx, val in bas1:
                    ampl.pphh[idx] = val
                ret_ampl = self @ ampl
                view_sd[:, :, j] = ret_ampl.ph.to_ndarray()

                for i, bas2 in enumerate(basis["pphh"]):
                    view_dd[i, j] = sum(val * ret_ampl.pphh[idx]
                                        for idx, val in bas2)

            out[n_ph:, :n_ph] = np.transpose(out[:n_ph, n_ph:])
        return out



class AdcMatrix_submatrix(AdcMatrixlike):
    # Default perturbation-theory orders for the matrix blocks (== standard ADC-PP).
    default_block_orders = {
        #             ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),
        #"adc0":  dict(gs_gs=0, gs_ph=0, ph_gs=0, ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc0":  dict(ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc1":  dict(ph_ph=1, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc2":  dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=0),     # noqa: E501
        "adc2x": dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=1),     # noqa: E501
        "adc3":  dict(ph_ph=3, ph_pphh=2,    pphh_ph=2,    pphh_pphh=1),     # noqa: E501
    }

    def __init__(self, method, hf_or_mp, subblock, block_orders=None, intermediates=None):
        """
        Initialise an ADC matrix.

        Parameters
        ----------
        method : str or AdcMethod
            Method to use.
        hf_or_mp : adcc.ReferenceState or adcc.LazyMp
            HF reference or MP ground state
        elec_or_phot: string, either "elec" or "phot"
            electronic or photonic part of qed-matrix, without groundstate
        block_orders : optional
            The order of perturbation theory to employ for each matrix block.
            If not set, defaults according to the selected ADC method are chosen.
        intermediates : adcc.Intermediates or NoneType
            Allows to pass intermediates to re-use to this class.
        """
        if isinstance(hf_or_mp, (libadcc.ReferenceState,
                                 libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)
        if not isinstance(hf_or_mp, LazyMp):
            raise TypeError("mp_results is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)

        self.timer = Timer()
        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = method.is_core_valence_separated
        self.ndim = 2

        self.intermediates = intermediates
        if self.intermediates is None:
            self.intermediates = Intermediates(self.ground_state)

        # Determine orders of PT in the blocks
        if block_orders is None:
            block_orders = self.default_block_orders[method.base_method.name]
        else:
            tmp_orders = self.default_block_orders[method.base_method.name].copy()
            tmp_orders.update(block_orders)
            block_orders = tmp_orders

        # Sanity checks on block_orders
        for block in block_orders.keys():
            if block not in ("gs_gs", "gs_ph", "ph_gs" ,"ph_ph", "ph_pphh", "pphh_ph", "pphh_pphh"):
                raise ValueError(f"Invalid block order key: {block}")
        if block_orders["ph_pphh"] != block_orders["pphh_ph"]:
            raise ValueError("ph_pphh and pphh_ph should always have "
                             "the same order")
        if block_orders["ph_pphh"] is not None \
           and block_orders["pphh_pphh"] is None:
            raise ValueError("pphh_pphh cannot be None if ph_pphh isn't.")
        self.block_orders = block_orders

        # Build the blocks and diagonals
        
        with self.timer.record("build"):
            variant = None
            if method.is_core_valence_separated:
                variant = "cvs"
            self.blocks_ph = {}
            if subblock == "elec":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=order, intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "elec_couple":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_couple", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "phot_couple":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot_couple", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "phot":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "phot2":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot2", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "phot_couple_inner":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot_couple_inner", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "phot_couple_edge":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_phot_couple_edge", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "elec_couple_inner":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_couple_inner", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            elif subblock == "elec_couple_edge":
                self.blocks_ph = {  # TODO Rename to self.block in 0.16.0
                    block: ppmatrix.block(self.ground_state, block.split("_"),
                                        order=str(order) + "_couple_edge", intermediates=self.intermediates,
                                        variant=variant)
                    for block, order in block_orders.items() if order is not None
                }
            else:
                ValueError("Using this class you have to give the parameter elec_or_phot")
            
            
        
            #for block, order in block_orders.items():
            #    if order is not None:
            #        print(block, order) 
            #for bl in self.blocks_ph.values():
            #    print(bl.diagonal)
            #for bl in self.blocks_ph:
            #    print(bl)
            self.__diagonal = sum(bl.diagonal for bl in self.blocks_ph.values()
                                  if bl.diagonal)
            self.__diagonal.evaluate()
            self.__init_space_data(self.__diagonal)
            #print("following is self.__init_space_data(self.__diagonal)")
            #print(self.__init_space_data(self.__diagonal))
        
        """
        # here we try to use the whole functionality given for the non-qed case (with the AmplitudeVector class)
        # and then use these to build the QED_AmplitudeVector class. Then we do something like
        # self.blocks_ph = QED_AmplitudeVector(val0, self.blocks_ph0, val1, self.blocks_ph1) and
        # self.diagonal = QED_AmplitudeVector(val0, self.__diagonal0, val1, self.__diagonal1)
        
        with self.timer.record("build"):
            variant = None
            if method.is_core_valence_separated:
                variant = "cvs"

            # first electronic part

            self.blocks_ph_0 = {  # TODO Rename to self.block in 0.16.0
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=order, intermediates=self.intermediates,
                                      variant=variant)
                for block, order in block_orders.items() if order is not None
            }
            for block, order in block_orders.items():
                if order is not None:
                    print(block, order) 
            self.__diagonal_0 = sum(bl.diagonal for bl in self.blocks_ph_0.values()
                                  if bl.diagonal)
            self.__diagonal_0.evaluate()
            #self.__init_space_data_0(self.__diagonal_0)

            # now for photonic part

            self.blocks_ph_1 = {  # TODO Rename to self.block in 0.16.0
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=str(order) + "_phot", intermediates=self.intermediates,
                                      variant=variant)
                for block, order in block_orders.items() if order is not None
            }
            #for block, order in block_orders.items():
            #    if order is not None:
            #        print(block, order) 
            self.__diagonal_1 = sum(bl.diagonal for bl in self.blocks_ph_1.values()
                                  if bl.diagonal)
            self.__diagonal_1.evaluate()
            #self.__init_space_data_0(self.__diagonal_1)

            # build QED_AmplitudeVector

            self.blocks_ph = QED_AmplitudeVector(0, self.blocks_ph_0, 0, self.blocks_ph_1)
            self.__diagonal = QED_AmplitudeVector(0, self.__diagonal_0, 0, self.__diagonal_1)
            self.__init_space_data_qed(self.__diagonal_0) # here we need to adapt the attributes, defined via this function
            # namely self.axis_spaces, self.axis_lengths, self.shape
            print(self.axis_spaces)
            print(self.axis_lengths)
            print(self.shape)

            # since the guesses need the whole matrix object as input, istead of just the diagonals, it is useful to define
            # self.elec and self.phot as the the corresponding original matrices
            # this is done in one class, since we also require the full matrix for the solver
            """
    """
    def __init_space_data_qed(self, diagonal0):
        self.__init_space_data(diagonal0)
        axis_spaces0 = self.axis_spaces
        axis_lengths0 = self.axis_lengths
        shape0 = self.shape
        self.shape = (shape0[0] * 2 + 1, shape0[0] * 2 + 1)
        self.axis_spaces["gs"] = ["o0", "v0"]
        self.axis_lengths["gs"] = 1
        # axis_spaces and axis_lengths both are dicts, referring to "ph", "pphh" as a key, so either make extra blocks here, or probably better
        # make a longer list in funciton 'axis_blocks', so that later everything for "ph" and "pphh" is just used twice
        # also it seems, that self.axis_lengths is not further used, but shape[0] is rather used for that, so not necessary to build self.axis_lengths correctly
    """

    def __init_space_data(self, diagonal):
        """Update the cached data regarding the spaces of the ADC matrix"""
        self.axis_spaces = {}
        self.axis_lengths = {}
        for block in diagonal.blocks_ph:
            self.axis_spaces[block] = getattr(diagonal, block).subspaces
            self.axis_lengths[block] = np.prod([
                self.mospaces.n_orbs(sp) for sp in self.axis_spaces[block]
            ])
        self.axis_spaces["gs"] = ["o0", "v0"] 
        self.axis_lengths["gs"] = 1
        self.shape = (sum(self.axis_lengths.values()),
                      sum(self.axis_lengths.values()))
        
        #print(self.axis_spaces)
        #print(self.axis_lengths)
        #print(self.shape)


    def __repr__(self):
        ret = f"AdcMatrix({self.method.name}, "
        for b, o in self.block_orders.items():
            ret += f"{b}={o}, "
        return ret + ")"

    def __len__(self):
        return self.shape[0]

    @property
    def blocks(self):
        # TODO Remove in 0.16.0
        return self.__diagonal.blocks

    def has_block(self, block):
        warnings.warn("The has_block function is deprecated and "
                      "will be removed in 0.16.0. "
                      "Use `in matrix.axis_blocks` in the future.")
        return self.block_spaces(block) is not None

    def block_spaces(self, block):
        warnings.warn("The block_spaces function is deprecated and "
                      "will be removed in 0.16.0. "
                      "Use `matrix.axis_spaces[block]` in the future.")
        return {
            "g": self.axis_spaces.get("gs", None),
            "s": self.axis_spaces.get("ph", None),
            "d": self.axis_spaces.get("pphh", None),
            "t": self.axis_spaces.get("ppphhh", None),
        }[block]

    @property #non-qed
    def axis_blocks(self):
        """
        Return the blocks used along one of the axes of the ADC matrix
        (e.g. ['ph', 'pphh']).
        """
        return list(self.axis_spaces.keys())

    #@property #qed
    #def axis_blocks(self):
        #print(list(self.axis_spaces.keys()) + list(self.axis_spaces.keys()))
    #    return list(self.axis_spaces.keys()) + list(self.axis_spaces.keys())

    def diagonal(self, block=None):
        """Return the diagonal of the ADC matrix"""
        if block is not None:
            warnings.warn("Support for the block argument will be dropped "
                          "in 0.16.0.")
            if block == "g":
                return self.__diagonal.gs #this is just a float, but it could be e.g. omega, for which I dont know if required data is given in this function
            if block == "s":
                return self.__diagonal.ph
            if block == "d":
                return self.__diagonal.pphh
        return self.__diagonal

    def compute_apply(self, block, tensor):
        warnings.warn("The compute_apply function is deprecated and "
                      "will be removed in 0.16.0.")
        if block in ("gg", "gs", "sg" ,"ss", "sd", "ds", "dd"):
            warnings.warn("The singles-doubles interface is deprecated and "
                          "will be removed in 0.16.0.")
            block = {"gg": "gs_gs", "gs": "gs_ph", "sg": "ph_gs",
                     "ss": "ph_ph", "sd": "ph_pphh",
                     "ds": "pphh_ph", "dd": "pphh_pphh"}[block]
        return self.block_apply(block, tensor)

    def block_apply(self, block, tensor):
        """
        Compute the application of a block of the ADC matrix
        with another AmplitudeVector or Tensor. Non-matching blocks
        in the AmplitudeVector will be ignored.
        """
        if not isinstance(tensor, libadcc.Tensor):
            raise TypeError("tensor should be an adcc.Tensor")

        with self.timer.record(f"apply/{block}"):
            outblock, inblock = block.split("_")
            ampl = AmplitudeVector(**{inblock: tensor})
            ret = self.blocks_ph[block].apply(ampl)
            return getattr(ret, outblock)

    @timed_member_call()
    def matvec(self, v):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result.
        """
        return sum(block.apply(v) for block in self.blocks_ph.values())

    def rmatvec(self, v):
        # ADC matrix is symmetric
        return self.matvec(v)

    def compute_matvec(self, ampl):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result.
        """
        warnings.warn("The compute_matvec function is deprecated and "
                      "will be removed in 0.16.0.")
        return self.matvec(ampl)

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return [self.matvec(ov) for ov in other]
        return NotImplemented

    def block_view(self, block):
        """
        Return a view into the AdcMatrix that represents a single
        block of the matrix. Currently only diagonal blocks are supported.
        """
        b1, b2 = block.split("_")
        if b1 != b2:
            raise NotImplementedError("Off-diagonal block views not yet "
                                      "implemented.")
            # TODO For off-diagonal blocks we probably need a different
            #      data structure as the AdcMatrix class as these block
            #      are inherently different than an AdcMatrix (no Hermiticity
            #      for example) and basically they only need to support some
            #      form of matrix-vector product and some stastics like
            #      spaces and sizes etc.
        block_orders = {bl: None for bl in self.block_orders.keys()}
        block_orders[block] = self.block_orders[block]
        return AdcMatrix(self.method, self.ground_state,
                         block_orders=block_orders,
                         intermediates=self.intermediates)

    def construct_symmetrisation_for_blocks(self):
        """
        Construct the symmetrisation functions, which need to be
        applied to relevant blocks of an AmplitudeVector in order
        to symmetrise it to the right symmetry in order to be used
        with the various matrix-vector-products of this function.

        Most importantly the returned functions antisymmetrise
        the occupied and virtual parts of the doubles parts
        if this is sensible for the method behind this adcmatrix.

        Returns a dictionary block identifier -> function
        """
        ret = {}
        if self.is_core_valence_separated:
            # CVS doubles part is antisymmetric wrt. (i,K,a,b) <-> (i,K,b,a)
            ret["pphh"] = lambda v: v.antisymmetrise([(2, 3)])
        else:
            def symmetrise_generic_adc_doubles(invec):
                # doubles part is antisymmetric wrt. (i,j,a,b) <-> (i,j,b,a)
                scratch = invec.antisymmetrise([(2, 3)])
                # doubles part is symmetric wrt. (i,j,a,b) <-> (j,i,b,a)
                return scratch.symmetrise([(0, 1), (2, 3)])
            ret["pphh"] = symmetrise_generic_adc_doubles
        return ret

    def dense_basis(self, axis_blocks=None, ordering="adcc"):
        """
        Return the list of indices and their values
        of the dense basis representation

        ordering: adcc, spin, spatial
        """
        ret = []
        if axis_blocks is None:
            axis_blocks = self.axis_blocks
        if not isinstance(axis_blocks, list):
            axis_blocks = [axis_blocks]

        # Define function to impose the order in the basis
        if ordering == "adcc":
            def reduce_index(n_orbsa, idx):
                return idx, idx
        elif ordering == "spin":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spin, then by spatial
                return (is_beta, spatial)
        elif ordering == "spatial":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spatial, then by spin
                return (spatial, is_beta)

        if "gs" in axis_blocks:
            ret_g = []
            ret_g.append([(0, 0), 1])
            ret.extend(ret_g)

        if "ph" in axis_blocks:
            ret_s = []
            sp_s = self.axis_spaces["ph"]
            n_orbs_s = [self.mospaces.n_orbs(sp) for sp in sp_s]
            n_orbsa_s = [self.mospaces.n_orbs_alpha(sp) for sp in sp_s]
            for i in range(n_orbs_s[0]):
                for a in range(n_orbs_s[1]):
                    ret_s.append([((i, a), 1)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_s, idx) for idx, factor in x)
            ret_s.sort(key=sortfctn)
            ret_s.sort(key=sortfctn)
            ret.extend(ret_s)

        if "pphh" in axis_blocks:
            ret_d = []
            sp_d = self.axis_spaces["pphh"]
            n_orbsa_d = [self.mospaces.n_orbs_alpha(sp) for sp in sp_d]

            if sp_d[0] == sp_d[1] and sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / 2),
                               ((j, i, a, b), -1 / 2),
                               ((i, j, b, a), -1 / 2),
                               ((j, i, b, a), +1 / 2)]
                              for i in range(nso) for j in range(i)
                              for a in range(nsv) for b in range(a)])
            elif sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / np.sqrt(2)),
                               ((i, j, b, a), -1 / np.sqrt(2))]
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(a)])
            else:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                nsw = self.mospaces.n_orbs(sp_d[3])
                ret_d.append([((i, j, b, a), 1)
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(nsw)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_d, idx) for idx, factor in x)
            ret_d.sort(key=sortfctn)
            ret_d.sort(key=sortfctn)
            ret.extend(ret_d)

        if any(b not in ("gs" ,"ph", "pphh") for b in self.axis_blocks):
            raise NotImplementedError("Blocks other than gs, ph and pphh "
                                      "not implemented")
        return ret

    def to_ndarray(self, out=None): # this is not adapted to gs
        """
        Return the ADC matrix object as a dense numpy array. Converts the sparse
        internal representation of the ADC matrix to a dense matrix and return
        as a numpy array.

        Notes
        -----

        This method is only intended to be used for debugging and
        visualisation purposes as it involves computing a large amount of
        matrix-vector products and the returned array consumes a considerable
        amount of memory.

        The resulting matrix has no spin symmetry imposed, which means that
        its eigenspectrum may contain non-physical excitations (e.g. with linear
        combinations of α->β and α->α components in the excitation vector).

        This function has not been sufficiently tested to be considered stable.
        """
        # TODO Update to ph / pphh
        # TODO Still uses deprecated functions
        import tqdm

        from adcc import guess_zero

        # Get zero amplitude of the appropriate symmetry
        # (TODO: Only true for C1, where there is only a single irrep)
        ampl_zero = guess_zero(self)
        assert self.mospaces.point_group == "C1"

        # Build the shape of the returned array
        # Since the basis of the doubles block is not the unit vectors
        # this *not* equal to the shape of the AdcMatrix object
        basis = {b: self.dense_basis(b) for b in self.axis_blocks}
        mat_len = sum(len(basis[b]) for b in basis)

        if out is None:
            out = np.zeros((mat_len, mat_len))
        else:
            if out.shape != (mat_len, mat_len):
                raise ValueError("Output array has shape ({0:}, {1:}), but "
                                 "shape ({2:}, {2:}) is required."
                                 "".format(*out.shape, mat_len))
            out[:] = 0  # Zero all data in out.

        # Check for the cases actually implemented
        if any(b not in ("ph", "pphh") for b in self.axis_blocks):
            raise NotImplementedError("Blocks other than ph and pphh "
                                      "not implemented")
        if "ph" not in self.axis_blocks:
            raise NotImplementedError("Block 'ph' needs to be present")

        # Extract singles-singles block (contiguous)
        assert "ph" in self.axis_blocks
        n_orbs_ph = [self.mospaces.n_orbs(sp) for sp in self.axis_spaces["ph"]]
        n_ph = np.prod(n_orbs_ph)
        assert len(basis["ph"]) == n_ph
        view_ss = out[:n_ph, :n_ph].reshape(*n_orbs_ph, *n_orbs_ph)
        for i in range(n_orbs_ph[0]):
            for a in range(n_orbs_ph[1]):
                ampl = ampl_zero.copy()
                ampl.ph[i, a] = 1
                view_ss[:, :, i, a] = (self @ ampl).ph.to_ndarray()

        # Extract singles-doubles and doubles-doubles block
        if "pphh" in self.axis_blocks:
            assert self.axis_blocks == ["ph", "pphh"]
            view_sd = out[:n_ph, n_ph:].reshape(*n_orbs_ph, len(basis["pphh"]))
            view_dd = out[n_ph:, n_ph:]
            for j, bas1 in tqdm.tqdm(enumerate(basis["pphh"]),
                                     total=len(basis["pphh"])):
                ampl = ampl_zero.copy()
                for idx, val in bas1:
                    ampl.pphh[idx] = val
                ret_ampl = self @ ampl
                view_sd[:, :, j] = ret_ampl.ph.to_ndarray()

                for i, bas2 in enumerate(basis["pphh"]):
                    view_dd[i, j] = sum(val * ret_ampl.pphh[idx]
                                        for idx, val in bas2)

            out[n_ph:, :n_ph] = np.transpose(out[:n_ph, n_ph:])
        return out



class AdcBlockView(AdcMatrix):
    def __init__(self, fullmatrix, block):
        warnings.warn("The AdcBlockView class got deprecated and will be "
                      "removed in 0.16.0. Use the matrix.block_view "
                      "function instead.")
        assert isinstance(fullmatrix, AdcMatrix)

        self.__fullmatrix = fullmatrix
        self.__block = block
        if block == "s":
            block_orders = dict(ph_ph=fullmatrix.block_orders["ph_ph"],
                                ph_pphh=None, pphh_ph=None, pphh_pphh=None)
        else:
            raise NotImplementedError(f"Block {block} not implemented")
        super().__init__(fullmatrix.method, fullmatrix.ground_state,
                         block_orders=block_orders,
                         intermediates=fullmatrix.intermediates)


class AdcMatrixShifted(AdcMatrix):
    def __init__(self, matrix, shift=0.0):
        """
        Initialise a shifted ADC matrix. Applying this class to a vector ``v``
        represents an efficient version of ``matrix @ v + shift * v``.

        Parameters
        ----------
        matrix : AdcMatrix
            Matrix which is shifted
        shift : float
            Value by which to shift the matrix
        """
        super().__init__(matrix.method, matrix.ground_state,
                         block_orders=matrix.block_orders,
                         intermediates=matrix.intermediates)
        self.shift = shift

    def matvec(self, in_ampl):
        out = super().matvec(in_ampl)
        out = out + self.shift * in_ampl
        return out

    def to_ndarray(self, out=None):
        super().to_ndarray(self, out)
        out = out + self.shift * np.eye(*out.shape)
        return out

    def block_apply(self, block, in_vec):
        ret = super().block_apply(block, in_vec)
        inblock, outblock = block.split("_")
        if inblock == outblock:
            ret += self.shift * in_vec
        return ret

    def diagonal(self, block=None):
        out = super().diagonal(block)
        out = out + self.shift  # Shift the diagonal
        return out

    def block_view(self, block):
        raise NotImplementedError("Block-view not yet implemented for "
                                  "shifted ADC matrices.")
        # TODO The way to implement this is to ask the inner matrix to
        #      a block_view and then wrap that in an AdcMatrixShifted.
