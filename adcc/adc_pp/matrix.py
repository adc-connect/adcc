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
import numpy as np
from numpy.lib.function_base import blackman

from adcc import block as b
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate
from adcc.AmplitudeVector import AmplitudeVector, QED_AmplitudeVector
from adcc.ReferenceState import ReferenceState
from adcc.OneParticleOperator import OneParticleOperator

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
    but not do CVS
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



# Since we already have all ph and pphh routines at hand, we will construct 4 dirrefent matrices,
# which will be evaluated by the original AdcMatrix, which is now the submatrix.
# We will construct the matrix as follows:
# elec        phot_couple
# elec_couple phot
# where we wont write elec explicitly.
# These blocks can then be forwarded to the AdcMatrix_submatrix, except for the gs part, which has to be
# treated separately. In the matvec we then construct the vector as:
# elec
# phot
# We therefore build an AmplitudeVector for the ph and pphh blocks, while inserting QED_AmplitudeVector.ph or .pphh,
# which are also AmplitudeVector classes
# It also seems smart, to insert ph_gs and pphh_gs into the ph_ph and pphh_ph blocks, respectively. This should also be fine with
# construct_symmetrisation_for_blocks, which enforces the correct symmetry for the doubles block, which does probably not (?)
# make a difference for the gs_pphh blocks, since they reduce to gs, so it should be fine to do so. -> It should be fine,
# since after using the Jacobi-davidson preconditioner by dividing the residuals with the shifted matrix-diagonal, the symmetry could
# be lost (?), but in the gs part, which is already only one value, the symmetry is still enforced.
# Maybe its also a good idea to pack gs_gs, gs_ph and gs_pphh into one block, which returns one number for apply and the gs_gs part
# for the diagonal.
# Using both of these "block in other block" ideas would leave one with 24 total blocks, instead of 36, for each order 
# (original 4 blocks + ph_gs and pphh_gs) per "ADC submatrix"
# maybe return float from ph/pphh_gs blocks, instead of QED_AmplitudeVector.gs/.gs1

# For QED-ADC(2) we then require also the 2 photon block, so we introduce the naming convention:
# elec              phot_couple         phot_couple_outer
# elec_couple       phot                phot_couple_inner
# elec_couple_edge  elec_couple_inner   phot2




#
# 0th order main
#


"""
def block_gs_gs_0(hf, mp, intermediates): # this is zero, but we want to give some test value here
    omega = float(ReferenceState.get_qed_omega(hf))
    #diagonal = QED_AmplitudeVector(gs=omega)
    def apply(ampl):
        #print("printing type(ampl)")
        #print(type(ampl), ampl)
        #print(type(ampl.gs))
        #print(ampl.gs)
        return QED_AmplitudeVector(gs=(0 * ampl.gs))
    return AdcBlock(apply, 0)

def block_gs_gs_0_couple(hf, mp, intermediates):
    def apply(ampl):
        return QED_AmplitudeVector(gs=(0 * ampl.gs))
    return AdcBlock(apply, 0)

def block_gs_gs_0_phot_couple(hf, mp, intermediates):
    def apply(ampl):
        return QED_AmplitudeVector(gs1=(0 * ampl.gs1))
    return AdcBlock(apply, 0)

#block_gs_gs_0_phot = block_gs_gs_0

def block_gs_gs_0_phot(hf, mp, intermediates): # this is zero, but we want to give some test value here
    omega = float(ReferenceState.get_qed_omega(hf))
    #diagonal = QED_AmplitudeVector(gs=omega)
    def apply(ampl):
        #print(type(ampl.gs1), ampl)
        #print(ampl.gs)
        return QED_AmplitudeVector(gs1=(0 * ampl.gs1))
    return AdcBlock(apply, 0)

def block_gs_ph_0(hf, mp, intermediates): # this is zero, but we want to give some test value here
    #return AdcBlock(lambda ampl: 0, 0)
    #omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        terms = [(mp.df(b.ov), ampl.ph)]
        return QED_AmplitudeVector(gs=sum(mpdf.dot(amplph)
                                    for mpdf, amplph in terms))
    return AdcBlock(apply, 0)

def block_gs_ph_0_couple(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)
    #def apply(ampl):
    #    return QED_AmplitudeVector(ph=(omega * ampl.ph))
    #return AdcBlock(apply, 0)

def block_gs_ph_0_phot_couple(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

def block_gs_ph_0_phot(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

def block_ph_gs_0(hf, mp, intermediates): # this is zero, but we want to give some test value here
    #return AdcBlock(lambda ampl: 0, 0)
    #omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(ph=(mp.df(b.ov) * ampl.gs))
    return AdcBlock(apply, 0)

def block_ph_gs_0_couple(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

def block_ph_gs_0_phot_couple(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

def block_ph_gs_0_phot(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)
"""

#
# 0th order gs blocks (gs_ph blocks in ph_ph, which are zero for this order)
#

def block_ph_gs_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

block_ph_gs_0_couple = block_ph_gs_0_phot_couple = block_ph_gs_0_phot = block_ph_gs_0
block_ph_gs_0_couple_edge = block_ph_gs_0_phot_couple_edge = block_ph_gs_0_phot2 = block_ph_gs_0
block_ph_gs_0_couple_inner = block_ph_gs_0_phot_couple_inner = block_ph_gs_0


def block_pphh_gs_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl:0, 0)

block_pphh_gs_0_couple = block_pphh_gs_0_phot_couple = block_pphh_gs_0_phot = block_pphh_gs_0
block_pphh_gs_0_couple_edge = block_pphh_gs_0_phot_couple_edge = block_pphh_gs_0_phot2 = block_pphh_gs_0
block_pphh_gs_0_couple_inner = block_pphh_gs_0_phot_couple_inner = block_pphh_gs_0

"""
def block_gs_ph_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(0 * ampl.ph))
    return AdcBlock(apply, 0)

def block_gs_ph_0_phot_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(0 * ampl.ph))
    return AdcBlock(apply, 0)

def block_gs_ph_0_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(0 * ampl.ph1))
    return AdcBlock(apply, 0)

def block_gs_ph_0_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(0 * ampl.ph1))
    return AdcBlock(apply, 0)
"""
#def block_pphh_gs_0(hf, mp, intermediates):
#    return AdcBlock(0, 0)

#block_pphh_gs_0_couple = block_pphh_gs_0_phot_couple = block_pphh_gs_0_phot = block_pphh_gs_0

#
# 0th order main
#

def block_ph_ph_0(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    if hasattr(hf, "coupling"):# and hasattr(hf, "qed_hf"):
        diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(), #change to QED_AmplitudeVector
                                                fCC.diagonal()))

        #np.insert(diagonal, 0, 0.)
        #diagonal = np.zeros(np.add(diagonal_.shape, np.array([0])))
        #diagonal[1:,1:] = diagonal_
        print("new stuff works???")
        #print(diagonal)

        def apply(ampl):
            mvprod = AmplitudeVector(ph=( #change to QED_AmplitudeVector
                + einsum("ib,ab->ia", ampl.ph, hf.fvv)
                - einsum("IJ,Ja->Ia", fCC, ampl.ph)
            ))
            #np.insert(mvprod, 0, 0.)
            #mvprod = np.zeros(np.add(mv_prod_.shape, np.array([0])))
            #mvprod[1:,1:] = mvprod_
            return mvprod 
    else:
        diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                                fCC.diagonal()))

        def apply(ampl):
            return AmplitudeVector(ph=(
                + einsum("ib,ab->ia", ampl.ph, hf.fvv)
                - einsum("IJ,Ja->Ia", fCC, ampl.ph)
            ))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_0 = block_ph_ph_0

def block_ph_ph_0_couple(hf, mp, intermediates): # we also give these blocks zero diagonals, so the submatrix routine does not require adjustments
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    return AdcBlock(lambda ampl: 0, diagonal)
#    def apply(ampl):
#        return AmplitudeVector(ph=(0 * ampl.ph))
#    return AdcBlock(apply, diagonal)

block_ph_ph_0_phot_couple = block_ph_ph_0_couple
block_ph_ph_0_phot_couple_edge = block_ph_ph_0_phot_couple_inner = block_ph_ph_0_couple_edge = block_ph_ph_0_couple_inner = block_ph_ph_0_couple

#def block_ph_ph_0_phot_couple(hf, mp, intermediates):
#    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
#    def apply(ampl):
#        return AmplitudeVector(ph=(0 * ampl.ph1))
#    return AdcBlock(apply, diagonal)


def block_ph_ph_0_phot(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    if hasattr(hf, "coupling"):# and hasattr(hf, "qed_hf"):
        diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                                fCC.diagonal()))

        #np.insert(diagonal, 0, 0.)
        print("new stuff works???")
        #print(diagonal)

        def apply(ampl):
            mvprod = AmplitudeVector(ph=(
                + einsum("ib,ab->ia", ampl.ph1, hf.fvv)
                - einsum("IJ,Ja->Ia", fCC, ampl.ph1)
            ))
            #np.insert(mvprod, 0, 0.)
            return mvprod
    else:
        raise NotImplementedError("coupling needs to be given to reference wavefunction in input file for QED-ADC")
    return AdcBlock(apply, diagonal)

def block_ph_ph_0_phot2(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    if hasattr(hf, "coupling"):# and hasattr(hf, "qed_hf"):
        diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                                fCC.diagonal()))

        #np.insert(diagonal, 0, 0.)
        print("new stuff works???")
        #print(diagonal)

        def apply(ampl):
            mvprod = AmplitudeVector(ph=(
                + einsum("ib,ab->ia", ampl.ph2, hf.fvv)
                - einsum("IJ,Ja->Ia", fCC, ampl.ph2)
            ))
            #np.insert(mvprod, 0, 0.)
            return mvprod
    else:
        raise NotImplementedError("coupling needs to be given to reference wavefunction in input file for QED-ADC")
    return AdcBlock(apply, diagonal)

#def block_ph_ph1_0(hf, mp, intermediates):
#    return AdcBlock(lambda ampl: 0, 0)

#def block_ph1_ph_0(hf, mp, intermediates):
#    return AdcBlock(lambda ampl: 0, 0)



def diagonal_pphh_pphh_0(hf):
    # Note: adcman similarly does not symmetrise the occupied indices
    #       (for both CVS and general ADC)
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    res = direct_sum("-i-J+a+b->iJab",
                     hf.foo.diagonal(), fCC.diagonal(),
                     hf.fvv.diagonal(), hf.fvv.diagonal())
    return AmplitudeVector(pphh=res.symmetrise(2, 3))


def block_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_pphh_pphh_0_couple(hf, mp, intermediates):
    diagonal = AmplitudeVector(pphh=(mp.t2oo.zeros_like()))
    return AdcBlock(lambda ampl: 0, diagonal)


block_pphh_pphh_0_phot_couple = block_pphh_pphh_0_couple
block_pphh_pphh_0_phot_couple_edge = block_pphh_pphh_0_phot_couple_inner = block_pphh_pphh_0_couple_edge = block_pphh_pphh_0_couple_inner = block_pphh_pphh_0_couple

def block_pphh_pphh_0_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh1, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh1).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_cvs_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


def block_pphh_pphh_0_phot2(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh2, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh2).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_0(hf))


#
# 0th order coupling
#
def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_cvs_ph_pphh_0 = block_ph_pphh_0
block_cvs_pphh_ph_0 = block_pphh_ph_0

block_pphh_ph_0_couple = block_pphh_ph_0_phot_couple = block_pphh_ph_0_phot = block_pphh_ph_0
block_pphh_ph_0_couple_edge = block_pphh_ph_0_couple_inner = block_pphh_ph_0_phot_couple_edge = block_pphh_ph_0_phot_couple_inner = block_pphh_ph_0_phot2 = block_pphh_ph_0
block_ph_pphh_0_couple = block_ph_pphh_0_phot_couple = block_ph_pphh_0_phot = block_ph_pphh_0
block_ph_pphh_0_couple_edge = block_ph_pphh_0_couple_inner = block_ph_pphh_0_phot_couple_edge = block_ph_pphh_0_phot_couple_inner = block_ph_pphh_0_phot2 = block_ph_pphh_0



#
# 1st order gs blocks (gs_ph blocks in ph_ph)
#

def block_ph_gs_1(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

def block_ph_gs_1_phot(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return omega * ampl.gs1
    return AdcBlock(apply, omega)

#def block_ph_gs_1_phot_couple(hf, mp, intermediates):
#    return AdcBlock(lambda ampl: 0, 0)
block_ph_gs_1_phot_couple = block_ph_gs_1
block_ph_gs_1_phot_couple_edge = block_ph_gs_1_couple_edge = block_ph_gs_1

def block_ph_gs_1_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return (-1) * sqrt(0.5 * omega) * einsum("jb,jb->", mp.qed_t1_df(b.ov), ampl.ph)
    return AdcBlock(apply, 0)
    #return AdcBlock(lambda ampl: 0, 0)

def block_ph_gs_1_phot2(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return 2 * omega * ampl.gs2
    return AdcBlock(apply, omega)

def block_ph_gs_1_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return (-1) * sqrt(omega) * einsum("jb,jb->", mp.qed_t1_df(b.ov), ampl.ph1)
    return AdcBlock(apply, 0)

def block_ph_gs_1_phot_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return (1 - sqrt(2)) * sqrt(0.5 * omega) * einsum("jb,jb->", mp.qed_t1_df(b.ov), ampl.ph2)
    return AdcBlock(apply, 0)



def block_pphh_gs_1(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

block_pphh_gs_1_couple = block_pphh_gs_1_phot_couple = block_pphh_gs_1_phot = block_pphh_gs_1
block_pphh_gs_1_couple_edge = block_pphh_gs_1_couple_inner = block_pphh_gs_1_phot_couple_edge = block_pphh_gs_1_phot_couple_inner = block_pphh_gs_1_phot2 = block_pphh_gs_1



"""
def block_gs_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(0 * ampl.ph))
    return AdcBlock(apply, 0)

def block_gs_ph_1_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(0 * ampl.ph1))
    return AdcBlock(apply, 0)

def block_gs_ph_1_phot_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(mp.qed_t1_df(b.ov) * (-ampl.gs1.as_float())))
    return AdcBlock(apply, 0)

def block_gs_ph_1_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(0 * ampl.ph))
    return AdcBlock(apply, 0)
"""


#
# 1st order main
#
def block_ph_ph_1(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov
    omega = float(ReferenceState.get_qed_omega(hf)) # only for test purposes
    if hasattr(hf, "coupling") and not hasattr(hf, "qed_hf"):
        #diag_qed_term = einsum("klkl->", hf.oooo)
        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
            - einsum("IaIa->Ia", CvCv)  # order 1
            #+ (1/2) * einsum("klkl->", hf.oooo)
            + (1/2) * direct_sum("i-a->ia", einsum("ii->i", mp.qed_t0_df(b.oo)), einsum("aa->a", mp.qed_t0_df(b.vv)))
            #+ (1/2) * einsum("ia,ia->", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) #reintroduced (actually canceled from -E_0 (1))
        ))

        #d_vv = zeros_like(hf.fvv)
        #d_vv.set_mask("aa", 1.0)

        def apply(ampl):
            return AmplitudeVector(ph=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", fCC, ampl.ph)     # 0
                - einsum("JaIb,Jb->Ia", CvCv, ampl.ph)  # 1
                #+ (1/2) * einsum("klkl->", hf.oooo) * ampl.ph
                + (1/2) * einsum("ij,ja->ia", mp.qed_t0_df(b.oo), ampl.ph)
                - (1/2) * einsum("ib,ab->ia", ampl.ph, mp.qed_t0_df(b.vv))
                #+ (1/2) * einsum("ia,ia->", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph #reintroduced (actually canceled from -E_0 (1)
            ))
    elif hasattr(hf, "coupling") and hasattr(hf, "qed_hf"):
        omega = float(ReferenceState.get_qed_omega(hf)) # omega is only here for a test...actually omega does not appear in this block
        diagonal = AmplitudeVector(ph=( #change to QED_AmplitudeVector
            + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
            - einsum("IaIa->Ia", CvCv)  # order 1
        ))
        #print(diagonal.set_random())
        #np.insert(diagonal, 0, 0)
        #diagonal = np.zeros(np.add(diagonal_.shape, np.array([0])))
        #diagonal[1:,1:] = diagonal_

        def apply(ampl):
            mvprod = AmplitudeVector(ph=(  #change to QED_AmplitudeVector               # PT order
                + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", fCC, ampl.ph)     # 0
                - einsum("JaIb,Jb->Ia", CvCv, ampl.ph)  # 1
            ))
            #np.insert(mvprod, 0, 0)
            #mvprod = np.zeros(np.add(mv_prod_.shape, np.array([0])))
            #mvprod[1:,1:] = mvprod_
            return mvprod
    else:
        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
            - einsum("IaIa->Ia", CvCv)  # order 1
        ))

        def apply(ampl):
            return AmplitudeVector(ph=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", fCC, ampl.ph)     # 0
                - einsum("JaIb,Jb->Ia", CvCv, ampl.ph)  # 1
            ))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_1 = block_ph_ph_1


def block_ph_ph_1_phot(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov
    if hasattr(hf, "coupling") and not hasattr(hf, "qed_hf"):
        #diag_qed_term = einsum("klkl->", hf.oooo)
        omega = float(ReferenceState.get_qed_omega(hf))

        # Build two Kronecker deltas
        d_oo = zeros_like(hf.foo)
        d_vv = zeros_like(hf.fvv)
        d_oo.set_mask("ii", 1.0)
        d_vv.set_mask("aa", 1.0)

        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
            - einsum("IaIa->Ia", CvCv)  # order 1
            #+ (1/2) * einsum("klkl->", hf.oooo)
            + (1/2) * direct_sum("i-a->ia", einsum("ii->i", mp.qed_t0_df(b.oo)), einsum("aa->a", mp.qed_t0_df(b.vv)))
            #+ (1/2) * einsum("ia,ia->", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) #reintroduced (actually canceled from -E_0 (1))
            + einsum("ii,aa->ia", d_oo, d_vv) * omega
        ))

        def apply(ampl):
            return AmplitudeVector(ph=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph1, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", fCC, ampl.ph1)     # 0
                - einsum("JaIb,Jb->Ia", CvCv, ampl.ph1)  # 1
                #+ (1/2) * einsum("klkl->", hf.oooo) * ampl.ph
                + (1/2) * einsum("ij,ja->ia", mp.qed_t0_df(b.oo), ampl.ph1)
                - (1/2) * einsum("ib,ab->ia", ampl.ph1, mp.qed_t0_df(b.vv))
                #+ (1/2) * einsum("ia,ia->", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph #reintroduced (actually canceled from -E_0 (1)
                + omega * ampl.ph1
            ))
    elif hasattr(hf, "coupling") and hasattr(hf, "qed_hf"):
        omega = float(ReferenceState.get_qed_omega(hf))

        # Build two Kronecker deltas
        d_oo = zeros_like(hf.foo)
        d_vv = zeros_like(hf.fvv)
        d_oo.set_mask("ii", 1.0)
        d_vv.set_mask("aa", 1.0)

        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
            - einsum("IaIa->Ia", CvCv)  # order 1
            + einsum("ii,aa->ia", d_oo, d_vv) * omega
        ))
        #np.insert(diagonal, 0, omega)

        def apply(ampl):
            mvprod = AmplitudeVector(ph=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph1, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", fCC, ampl.ph1)     # 0
                - einsum("JaIb,Jb->Ia", CvCv, ampl.ph1)  # 1
                + omega * ampl.ph1
            ))
            #np.insert(mvprod, 0, omega)
            return mvprod
    else:
        raise NotImplementedError("and not hasattr(hf, qed_hf)")
    return AdcBlock(apply, diagonal)


def block_ph_ph_1_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    if hasattr(hf, "coupling"):# and not hasattr(hf, "qed_hf"):
        def apply(ampl):
            return AmplitudeVector(ph=(
                sqrt(omega / 2) * (- einsum("ib,ab->ia", ampl.ph, mp.qed_t1_df(b.vv))
                                        + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph))
            ))

            #add_axis1 = - np.sqrt(omega / 2) * mp.qed_t1_df(b.ov)

            #np.insert(mvprod, 0, - np.sqrt(omega / 2) * mp.qed_t1_df(b.ov), axis=1) #ground to excited state coupling
            #also insert axis0 zeros, so that final matvecproduct has dimesions (i+1) x (a+1)
    #else:
    #    raise NotImplementedError("and not hasattr(hf, qed_hf)")
    return AdcBlock(apply, diagonal)


def block_ph_ph_1_phot_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    if hasattr(hf, "coupling"):# and not hasattr(hf, "qed_hf"):
        def apply(ampl):
            return AmplitudeVector(ph=(
                sqrt(omega / 2) * ( - einsum("ib,ab->ia", ampl.ph1, mp.qed_t1_df(b.vv))
                                    + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph1)
                                    - mp.qed_t1_df(b.ov) * ampl.gs1.as_float()) # gs_ph block 
            ))

            #add_axis0 = - np.sqrt(omega / 2) * mp.qed_t1_df(b.ov)

            #np.insert(mvprod, 0, - np.sqrt(omega / 2) * mp.qed_t1_df(b.ov), axis=0) #ground to excited state coupling
            #also insert axis1 zeros, so that final matvecproduct has dimesions (i+1) x (a+1)
    #else:
    #    raise NotImplementedError("and not hasattr(hf, qed_hf)")
    return AdcBlock(apply, diagonal)


def block_ph_ph_1_couple_edge(hf, mp, intermediates):
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    return AdcBlock(lambda ampl: 0, diagonal)

block_ph_ph_1_phot_couple_edge = block_ph_ph_1_couple_edge

def block_ph_ph_1_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    if hasattr(hf, "coupling"):# and not hasattr(hf, "qed_hf"):
        def apply(ampl):
            return AmplitudeVector(ph=(
                sqrt(omega) * (- einsum("ib,ab->ia", ampl.ph1, mp.qed_t1_df(b.vv))
                                        + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph1))
                + (1 - sqrt(2)) * sqrt(omega / 2) * mp.qed_t1_df(b.ov) * ampl.gs1.as_float() # gs part
            ))
    return AdcBlock(apply, diagonal)

def block_ph_ph_1_phot_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    if hasattr(hf, "coupling"):# and not hasattr(hf, "qed_hf"):
        def apply(ampl):
            return AmplitudeVector(ph=(
                sqrt(omega) * ( - einsum("ib,ab->ia", ampl.ph2, mp.qed_t1_df(b.vv))
                                    + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph2)
                                    - mp.qed_t1_df(b.ov) * ampl.gs2.as_float()) # gs_ph block 
            ))
    return AdcBlock(apply, diagonal)

def block_ph_ph_1_phot2(hf, mp, intermediates):
    fCC = hf.fcc if hf.has_core_occupied_space else hf.foo
    CvCv = hf.cvcv if hf.has_core_occupied_space else hf.ovov
    if hasattr(hf, "coupling") and not hasattr(hf, "qed_hf"):
        #diag_qed_term = einsum("klkl->", hf.oooo)
        omega = float(ReferenceState.get_qed_omega(hf))

        # Build two Kronecker deltas
        d_oo = zeros_like(hf.foo)
        d_vv = zeros_like(hf.fvv)
        d_oo.set_mask("ii", 1.0)
        d_vv.set_mask("aa", 1.0)

        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), fCC.diagonal())  # order 0
            - einsum("IaIa->Ia", CvCv)  # order 1
            #+ (1/2) * einsum("klkl->", hf.oooo)
            + (1/2) * direct_sum("i-a->ia", einsum("ii->i", mp.qed_t0_df(b.oo)), einsum("aa->a", mp.qed_t0_df(b.vv)))
            #+ (1/2) * einsum("ia,ia->", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) #reintroduced (actually canceled from -E_0 (1))
            + einsum("ii,aa->ia", d_oo, d_vv) * omega * 2
        ))

        def apply(ampl):
            return AmplitudeVector(ph=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph2, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", fCC, ampl.ph2)     # 0
                - einsum("JaIb,Jb->Ia", CvCv, ampl.ph2)  # 1
                #+ (1/2) * einsum("klkl->", hf.oooo) * ampl.ph
                + (1/2) * einsum("ij,ja->ia", mp.qed_t0_df(b.oo), ampl.ph2)
                - (1/2) * einsum("ib,ab->ia", ampl.ph2, mp.qed_t0_df(b.vv))
                #+ (1/2) * einsum("ia,ia->", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph #reintroduced (actually canceled from -E_0 (1)
                + 2 * omega * ampl.ph2
            ))
    else:
        raise NotImplementedError("and not hasattr(hf, qed_hf)")
    return AdcBlock(apply, diagonal)



def diagonal_pphh_pphh_1(hf):
    # Fock matrix and ovov diagonal term (sometimes called "intermediate diagonal")
    dinterm_ov = (direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())
                  - 2.0 * einsum("iaia->ia", hf.ovov)).evaluate()

    if hf.has_core_occupied_space:
        dinterm_Cv = (direct_sum("a-I->Ia", hf.fvv.diagonal(), hf.fcc.diagonal())
                      - 2.0 * einsum("IaIa->Ia", hf.cvcv)).evaluate()
        diag_oC = einsum("iJiJ->iJ", hf.ococ)
    else:
        dinterm_Cv = dinterm_ov
        diag_oC = einsum("ijij->ij", hf.oooo).symmetrise()

    diag_vv = einsum("abab->ab", hf.vvvv).symmetrise()
    return AmplitudeVector(pphh=(
        + direct_sum("ia+Jb->iJab", dinterm_ov, dinterm_Cv).symmetrise(2, 3)
        + direct_sum("iJ+ab->iJab", diag_oC, diag_vv)
    ))


def block_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(  # 0th order
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
            # 1st order
            + (
                -4 * einsum("ikac,kbjc->ijab", ampl.pphh, hf.ovov)
            ).antisymmetrise(0, 1).antisymmetrise(2, 3)
            + 0.5 * einsum("ijkl,klab->ijab", hf.oooo, ampl.pphh)
            + 0.5 * einsum("ijcd,abcd->ijab", ampl.pphh, hf.vvvv)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


def block_cvs_pphh_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            # 0th order
            + 2.0 * einsum("iJac,bc->iJab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 1.0 * einsum("ik,kJab->iJab", hf.foo, ampl.pphh)
            - 1.0 * einsum("JK,iKab->iJab", hf.fcc, ampl.pphh)
            # 1st order
            + (
                - 2.0 * einsum("iKac,KbJc->iJab", ampl.pphh, hf.cvcv)
                + 2.0 * einsum("icka,kJbc->iJab", hf.ovov, ampl.pphh)
            ).antisymmetrise(2, 3)
            + 1.0 * einsum("iJlK,lKab->iJab", hf.ococ, ampl.pphh)
            + 0.5 * einsum("iJcd,abcd->iJab", ampl.pphh, hf.vvvv)
        ))
    return AdcBlock(apply, diagonal_pphh_pphh_1(hf))


#
# 1st order coupling
#
def block_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)

def block_ph_pphh_1_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh1)
            + einsum("ijbc,jabc->ia", ampl.pphh1, hf.ovvv)
        ))
    return AdcBlock(apply, 0)

def block_ph_pphh_1_phot2(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh2)
            + einsum("ijbc,jabc->ia", ampl.pphh2, hf.ovvv)
        ))
    return AdcBlock(apply, 0)



def block_cvs_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + sqrt(2) * einsum("jKIb,jKab->Ia", hf.occv, ampl.pphh)
            - 1 / sqrt(2) * einsum("jIbc,jabc->Ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)

def block_pphh_ph_1_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph1, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph1).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)

def block_pphh_ph_1_phot2(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph2, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph2).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + sqrt(2) * einsum("jIKb,Ka->jIab",
                               hf.occv, ampl.ph).antisymmetrise(2, 3)
            - 1 / sqrt(2) * einsum("Ic,jcab->jIab", ampl.ph, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_ph_pphh_1_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(ph=(
            -4 * sqrt(omega/2) * einsum("kc,ikac->ia", mp.qed_t1_df(b.ov), ampl.pphh)
                            #+ einsum("jb,jiba->ia", mp.qed_t1_df(b.ov), ampl.pphh)
                            #- einsum("kb,ikba->ia", mp.qed_t1_df(b.ov), ampl.pphh)
                            #- einsum("jc,jiac->ia", mp.qed_t1_df(b.ov), ampl.pphh))
        ))
    return AdcBlock(apply, 0)


def block_ph_pphh_1_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(ph=(
            -4 * sqrt(omega) * einsum("kc,ikac->ia", mp.qed_t1_df(b.ov), ampl.pphh1)
                            #+ einsum("jb,jiba->ia", mp.qed_t1_df(b.ov), ampl.pphh)
                            #- einsum("kb,ikba->ia", mp.qed_t1_df(b.ov), ampl.pphh)
                            #- einsum("jc,jiac->ia", mp.qed_t1_df(b.ov), ampl.pphh))
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(pphh=(
            4 * sqrt(omega/2) * einsum("jb,ia->ijab", mp.qed_t1(b.ov), einsum("ia,ia->ia", mp.df(b.ov), ampl.ph)).antisymmetrise(0,1).antisymmetrise(2,3)
                            #+ einsum("ia,jb,jb->ijab", mp.qed_t1(b.ov), mp.df(b.ov), ampl.ph)
                            #- einsum("ja,ib,ib->ijab", mp.qed_t1(b.ov), mp.df(b.ov), ampl.ph)
                            #- einsum("ib,ja,ja->ijab", mp.qed_t1(b.ov), mp.df(b.ov), ampl.ph))
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(pphh=(
            4 * sqrt(omega/2) * einsum("jb,ia->ijab", mp.qed_t1(b.ov), einsum("ia,ia->ia", mp.df(b.ov), ampl.ph1)).antisymmetrise(0,1).antisymmetrise(2,3)
            + (1 - sqrt(2)) * 4 * sqrt(omega / 2) * einsum("jb,ia->ijab", mp.qed_t1_df(b.ov), ampl.ph1).antisymmetrise(0,1).antisymmetrise(2,3)
                            #+ einsum("ia,jb,jb->ijab", mp.qed_t1(b.ov), mp.df(b.ov), ampl.ph)
                            #- einsum("ja,ib,ib->ijab", mp.qed_t1(b.ov), mp.df(b.ov), ampl.ph)
                            #- einsum("ib,ja,ja->ijab", mp.qed_t1(b.ov), mp.df(b.ov), ampl.ph))
        ))
    return AdcBlock(apply, 0)



def block_ph_pphh_1_phot_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(ph=(
            sqrt(omega/2) * einsum("kc,ia,ikac->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh1) 
                            + einsum("jb,ia,jiba->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh1)
                            - einsum("kb,ia,ikba->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh1)
                            - einsum("jc,ia,jiac->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh1)
        ))
    return AdcBlock(apply, 0)


def block_ph_pphh_1_phot_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(ph=(
            sqrt(omega/2) * einsum("kc,ia,ikac->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh2)
                            + einsum("jb,ia,jiba->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh2)
                            - einsum("kb,ia,ikba->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh2)
                            - einsum("jc,ia,jiac->ia", mp.qed_t1(b.ov), mp.df(b.ov), ampl.pphh2)
            + 4 * (1 - sqrt(2)) * sqrt(omega/2) * einsum("kc,ikac->ia", mp.qed_t1_df(b.ov), ampl.pphh2)
        ))
    return AdcBlock(apply, 0)



def block_pphh_ph_1_phot_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(pphh=(
            -4 * sqrt(omega/2) * einsum("jb,ia->ijab", mp.qed_t1_df(b.ov), ampl.ph1).antisymmetrise(0,1).antisymmetrise(2,3) 
                            #+ einsum("ia,jb->ijab", mp.qed_t1_df(b.ov), ampl.ph1)
                            #- einsum("ja,ib->ijab", mp.qed_t1_df(b.ov), ampl.ph1)
                            #- einsum("ib,ja->ijab", mp.qed_t1_df(b.ov), ampl.ph1))
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1_phot_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    def apply(ampl):
        return AmplitudeVector(pphh=(
            -4 * sqrt(omega) * einsum("jb,ia->ijab", mp.qed_t1_df(b.ov), ampl.ph2).antisymmetrise(0,1).antisymmetrise(2,3) 
                            #+ einsum("ia,jb->ijab", mp.qed_t1_df(b.ov), ampl.ph1)
                            #- einsum("ja,ib->ijab", mp.qed_t1_df(b.ov), ampl.ph1)
                            #- einsum("ib,ja->ijab", mp.qed_t1_df(b.ov), ampl.ph1))
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1_couple_edge(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

block_pphh_ph_1_phot_couple_edge = block_pphh_ph_1_couple_edge

def block_ph_pphh_1_couple_edge(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)

block_ph_pphh_1_phot_couple_edge = block_ph_pphh_1_couple_edge



#
# 2nd order gs blocks (gs_ph blocks in ph_ph) for now these are zero for testing purposes
#

def block_ph_gs_2(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    def apply(ampl):
        return (einsum("jb,jb->", (
                        - 0.25 * einsum("bc,jc->jb", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                        + 0.25 * einsum("jk,kb->jb", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov))
                        - (omega / 2) * einsum("jc,bc->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ (omega / 2) * einsum("jc,bc->jb", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + (omega / 2) * einsum("kb,jk->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- (omega / 2) * einsum("kb,jk->jb", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        - 0.5 * einsum("jkbc,kc->jb", mp.t2oo, mp.qed_t0_df(b.ov))),
                        ampl.ph))
    return AdcBlock(apply, 0)

def block_ph_gs_2_phot_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    diagonal = - sqrt(omega / 2) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov))
    def apply(ampl):
        return ( sqrt(omega / 2) * einsum("jb,jb->", (
                        - einsum("jckb,kc->jb", hf.ovov, mp.qed_t1(b.ov))
                        + omega * mp.qed_t1(b.ov)
                        - 0.5 * einsum("bc,jc->jb", mp.qed_t0_df(b.vv), mp.qed_t1(b.ov))
                        + 0.5 * einsum("jk,kb->jb", mp.qed_t0_df(b.oo), mp.qed_t1(b.ov))
                        - 0.5 * einsum("jc,bc->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ 0.5 * einsum("jc,bc->jb", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("kb,jk->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- 0.5 * einsum("kb,jk->jb", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + einsum("jkbc,kc->jb", mp.t2oo, mp.qed_t1_df(b.ov))),
                        ampl.ph1))
    return AdcBlock(apply, diagonal)



def block_ph_gs_2_couple(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    diagonal = - sqrt(omega / 2) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov))
    def apply(ampl):
        return ( sqrt(omega / 2) * einsum("jb,jb->", (
                        - 0.5 * einsum("jc,bc->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ 0.5 * einsum("jc,bc->jb", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("kb,jk->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- 0.5 * einsum("kb,jk->jb", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + einsum("jkbc,kc->jb", mp.t2oo, mp.qed_t1_df(b.ov))),
                        ampl.ph))
                        #maybe this whole block is just + einsum("kjbc,kc->jb", hf.oovv, mp.qed_t1(b.ov)))
    return AdcBlock(apply, diagonal)

def block_ph_gs_2_phot(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    diagonal = - omega * (sqrt(2) - 1) * einsum("kc,kc->", mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))
    def apply(ampl):
        return (einsum("jb,jb->", (
                        + 0.5 * omega * mp.qed_t0(b.ov)
                        - 0.25 * einsum("bc,jc->jb", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                        + 0.25 * einsum("jk,kb->jb", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov))
                        + sqrt(2) * (
                        - (omega / 2) * einsum("jc,bc->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ (omega / 2) * einsum("jc,bc->jb", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + (omega / 2) * einsum("kb,jk->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- (omega / 2) * einsum("kb,jk->jb", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        )
                        - 0.5 * einsum("jkbc,kc->jb", mp.t2oo, mp.qed_t0_df(b.ov))),
                        ampl.ph1))
                        #(omega * einsum("jb,jb->", einsum("jj,bb->jb", d_oo, d_vv), ampl.ph1)
            #- (omega / 2) * (sqrt(2) - 1) * einsum("jb,jb->", (einsum("jc,bc->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv))
            #                                                    - einsum("jc,bc->jb", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv)) 
            #                                                    - einsum("kb,jk->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo))
            #                                                    + einsum("kb,jk->jb", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))), 
            #                                                ampl.ph1))
    return AdcBlock(apply, diagonal)

def block_ph_gs_2_phot2(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    diagonal = - omega * (sqrt(3) - 1) * einsum("kc,kc->", mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))
    def apply(ampl):
        return (einsum("jb,jb->", (
                        + omega * mp.qed_t0(b.ov)
                        - 0.25 * einsum("bc,jc->jb", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                        + 0.25 * einsum("jk,kb->jb", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov))
                        + sqrt(3) * (
                        - (omega / 2) * einsum("jc,bc->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ (omega / 2) * einsum("jc,bc->jb", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + (omega / 2) * einsum("kb,jk->jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- (omega / 2) * einsum("kb,jk->jb", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        )
                        - 0.5 * einsum("jkbc,kc->jb", mp.t2oo, mp.qed_t0_df(b.ov))),
                        ampl.ph2))
    return AdcBlock(apply, diagonal)


def block_ph_gs_2_phot_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    diagonal = - sqrt(omega) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov))
    def apply(ampl):
        return ( sqrt(omega / 2) * einsum("jb,jb->", (
                        - einsum("jckb,kc->jb", hf.ovov, mp.qed_t1(b.ov))
                        + 2 * omega * mp.qed_t1(b.ov)
                        - 0.5 * einsum("bc,jc->jb", mp.qed_t0_df(b.vv), mp.qed_t1(b.ov))
                        + 0.5 * einsum("jk,kb->jb", mp.qed_t0_df(b.oo), mp.qed_t1(b.ov))
                        - 0.5 * sqrt(2) * einsum("jc,bc->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ 0.5 * einsum("jc,bc->jb", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * sqrt(2) * einsum("kb,jk->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- 0.5 * einsum("kb,jk->jb", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + sqrt(2) * einsum("jkbc,kc->jb", mp.t2oo, mp.qed_t1_df(b.ov))),
                        ampl.ph2))
    return AdcBlock(apply, diagonal)



def block_ph_gs_2_couple_inner(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    diagonal = - sqrt(omega) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov))
    def apply(ampl):
        return ( sqrt(omega) * einsum("jb,jb->", (
                        - 0.5 * einsum("jc,bc->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ 0.5 * einsum("jc,bc->jb", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("kb,jk->jb", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- 0.5 * einsum("kb,jk->jb", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + einsum("jkbc,kc->jb", mp.t2oo, mp.qed_t1_df(b.ov))),
                        ampl.ph1))
                        #maybe this whole block is just + einsum("kjbc,kc->jb", hf.oovv, mp.qed_t1(b.ov)))
    return AdcBlock(apply, diagonal)


def block_ph_gs_2_phot_couple_edge(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    diagonal = - 0.5 * omega * sqrt(2) * einsum("kc,kc->", mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))

    def apply(ampl):
        return (- 0.5 * omega * sqrt(2) * (einsum("jc,bc,jb->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
                                         - einsum("kb,jk,jb", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)))

    return AdcBlock(apply, diagonal)

def block_ph_gs_2_couple_edge(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    diagonal = - 0.5 * omega * sqrt(2) * einsum("kc,kc->", mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))

    return AdcBlock(lambda ampl: 0, diagonal)



#
# 2nd order main
#
def block_ph_ph_2(hf, mp, intermediates):
    #omega = float(ReferenceState.get_qed_omega(hf))
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2
    #intermediates expect a dimensionality unequal zero, so qed_i0 cannot be cashed this way, but its just a sum over two indices
    #qed_i0_terms = [(mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))]
    #qed_i0 = 2 * sum(qed_t1_df.dot(qed_t1) for qed_t1_df, qed_t1 in qed_i0_terms)
    #qed_i1 = intermediates.adc2_qed_i1
    #qed_i2 = intermediates.adc2_qed_i2

    #d_oo = zeros_like(hf.foo)
    #d_vv = zeros_like(hf.fvv)
    #d_oo.set_mask("ii", 1.0)
    #d_vv.set_mask("aa", 1.0)

    term_t2_eri = (
        + einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
        + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo)
    ).evaluate()

    if hasattr(hf, "coupling"):
        omega = float(ReferenceState.get_qed_omega(hf))
        #qed_i0_terms = [(mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))]
        #qed_i0 = 2 * sum(qed_t1_df.dot(qed_t1) for qed_t1_df, qed_t1 in qed_i0_terms)
        qed_i1 = intermediates.adc2_qed_ph_ph_2_i1
        qed_i2 = intermediates.adc2_qed_ph_ph_2_i2
        #qed_i1 = intermediates.adc2_qed_i1
        #qed_i2 = intermediates.adc2_qed_i2
        #qed_i1_0 = intermediates.adc2_qed_i1_0
        #qed_i2_0 = intermediates.adc2_qed_i2_0
        qed_gs_part = intermediates.adc2_qed_ph_ph_2_gs_part
        if hasattr(hf, "qed_hf"):
            diagonal = AmplitudeVector(ph=(
                + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
                - einsum("IaIa->Ia", hf.ovov)
                - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
                + (-omega/2) * (#qed_i0 #this term and the following are additional qed terms
                - direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
                #+ (1/2) * einsum("iaia->ia", einsum("ia,jb->iajb", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov))
                #                             + einsum("jb,ia->iajb", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov))))
                + (1/2) * 2 * einsum("ia,ia->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))
            ))
            print(mp.energy_correction(2))
            def apply(ampl):
                return AmplitudeVector(ph=(
                    + einsum("ib,ab->ia", ampl.ph, i1)
                    - einsum("ij,ja->ia", i2, ampl.ph)
                    - einsum("jaib,jb->ia", hf.ovov, ampl.ph)    # 1
                    - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph)  # 2
                    + (-omega/2) * (#qed_i0 * ampl.ph #this term and the following are additional qed terms
                    - einsum("ib,ab->ia", ampl.ph, qed_i1)
                    - einsum("ij,ja->ia", qed_i2, ampl.ph)
                    #+ (1/2) * einsum("iajb,jb->ia", einsum("ia,jb->iajb", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov))
                    #                                 + einsum("jb,ia->iajb", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)), ampl.ph))
                    + (1/2) * (mp.qed_t1(b.ov) * mp.qed_t1_df(b.ov).dot(ampl.ph) 
                            + mp.qed_t1_df(b.ov) * mp.qed_t1(b.ov).dot(ampl.ph)))
                ))
        else:
            diagonal = AmplitudeVector(ph=(
                + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
                - einsum("IaIa->Ia", hf.ovov)
                - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
                + direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
                + (-omega/2) * (
                #- direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
                + (1/2) * 2 * einsum("ia,ia->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))
                #+ direct_sum("a+i->ia", qed_i1_0.diagonal(), qed_i2_0.diagonal())
                - einsum("ka,ikia->ia", mp.qed_t0(b.ov), hf.ooov)
                - einsum("ic,iaac->ia", mp.qed_t0(b.ov), hf.ovvv)
                #- (omega/2) * einsum("ia,ia->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) # reintroduced (actually canceled from -E_0 (2))
                #- (1/4) * einsum("ia,ia->", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov))
                #- (1/4) * einsum("ijab,ijab->", mp.td2(b.oovv), einsum("ia,jb->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) - einsum("ib,ja->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)))
            ))
            print(mp.energy_correction(2))
            def apply(ampl):
                return AmplitudeVector(ph=(
                    + einsum("ib,ab->ia", ampl.ph, i1)
                    - einsum("ij,ja->ia", i2, ampl.ph)
                    - einsum("jaib,jb->ia", hf.ovov, ampl.ph)    # 1
                    - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph)  # 2
                    + einsum("ib,ab->ia", ampl.ph, qed_i1)
                    + einsum("ij,ja->ia", qed_i2, ampl.ph)
                    + (-omega/2) * (
                    #- einsum("ib,ab->ia", ampl.ph, qed_i1)
                    #- einsum("ij,ja->ia", qed_i2, ampl.ph)
                    + (1/2) * (mp.qed_t1(b.ov) * mp.qed_t1_df(b.ov).dot(ampl.ph) 
                            + mp.qed_t1_df(b.ov) * mp.qed_t1(b.ov).dot(ampl.ph)))
                    #+ einsum("ib,ab->ia", ampl.ph, qed_i1_0)
                    #+ einsum("ij,ja->ia", qed_i2_0, ampl.ph)
                    #+ (1/2) * (einsum("ka,jkib,jb->ia", mp.qed_t0(b.ov), hf.ooov, ampl.ph) #this can be done by symmetrize a,b
                    #        + einsum("kb,jkia,jb->ia", mp.qed_t0(b.ov), hf.ooov, ampl.ph))
                    #+ (1/2) * (einsum("ic,jabc,jb->ia", mp.qed_t0(b.ov), hf.ovvv, ampl.ph) #this can be done by symmetrize i,j
                    #        + einsum("jc,iabc,jb->ia", mp.qed_t0(b.ov), hf.ovvv, ampl.ph))
                    + einsum("ijab,jb->ia", einsum("jkib,ka->ijab", hf.ooov, mp.qed_t0(b.ov)).symmetrise(2, 3)
                                            + einsum("ic,jabc->ijab", mp.qed_t0(b.ov), hf.ovvv).symmetrise(0, 1), ampl.ph)
                    + qed_gs_part * ampl.gs.as_float()
                    #+ (0.25 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov)) #this is from the gs_ph contribution
                    #        - 0.25 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                    #        - (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                    #        #+ (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                    #        + (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                    #        #- (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                    #        - 0.5 * einsum("ikac,kc->ia", mp.t2oo, mp.qed_t0_df(b.ov))) * ampl.gs.as_float()
                    #- (omega/2) * einsum("ia,ia->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph # reintroduced (actually canceled from -E_0 (2))
                    #- (1/4) * einsum("ia,ia->", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov)) * ampl.ph
                    #- (1/4) * einsum("ijab,ijab->", mp.td2(b.oovv), einsum("ia,jb->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) - einsum("ib,ja->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov))) * ampl.ph
                ))
    else:
        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
            - einsum("IaIa->Ia", hf.ovov)
            - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
        ))

    # Not used anywhere else, so kept as an anonymous intermediate
    #term_t2_eri = (
    #    + einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
    #    + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo)
    #).evaluate()

        def apply(ampl):
            #print("using non-qed matrix vector products")
            return AmplitudeVector(ph=(
                + einsum("ib,ab->ia", ampl.ph, i1)
                - einsum("ij,ja->ia", i2, ampl.ph)
                - einsum("jaib,jb->ia", hf.ovov, ampl.ph)    # 1
                - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph)  # 2
                #+ (-omega/2) * (qed_i0 * ampl.ph #this term and the following are additional qed terms
                #- einsum("ib,ab->ia", ampl.ph, qed_i1)
                #- einsum("ij,ja->ia", qed_i2, ampl.ph)
                #+ einsum("ijab,jb->ia", einsum("jb,ia->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)), ampl.ph))
            ))
    return AdcBlock(apply, diagonal)


def block_cvs_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), hf.fcc.diagonal())
        - einsum("IaIa->Ia", hf.cvcv)
    ))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", hf.fcc, ampl.ph)
            - einsum("JaIb,Jb->Ia", hf.cvcv, ampl.ph)
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_2_couple(hf, mp, intermediates): #one could cash some of the terms here
    if hasattr(hf, "qed_hf"):
        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
    omega = float(ReferenceState.get_qed_omega(hf))
    gs_part = intermediates.adc2_qed_ph_ph_2_couple_gs_part
    qed_i1 = intermediates.adc2_qed_couple_i1
    qed_i2 = intermediates.adc2_qed_couple_i2

    #d_oo = zeros_like(hf.foo)
    #d_vv = zeros_like(hf.fvv)
    #d_oo.set_mask("ii", 1.0)
    #d_vv.set_mask("aa", 1.0)

    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    def apply(ampl):
        return AmplitudeVector(ph=(0.5 * sqrt(omega / 2) * (einsum("kc,kc,acik,ia->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), 
                                                direct_sum("ia-kc->acik", mp.df(b.ov), mp.df(b.ov)), ampl.ph)
                                        - einsum("kb,ka,ik,ib->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.oo), ampl.ph)
                                                einsum("ic,kc->ik", mp.df(b.ov), - mp.df(b.ov)), ampl.ph) # this is different from mp.diff_df, but why???
                                        - einsum("jc,ic,ac,ja->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.vv), ampl.ph))
                                                einsum("ka,kc->ac", mp.df(b.ov), - mp.df(b.ov)), ampl.ph)) # this is different from mp.diff_df, but why???
                + einsum("ib,ab->ia", ampl.ph, qed_i1)
                + einsum("ij,ja->ia", qed_i2, ampl.ph)
                - 0.5 * sqrt(omega / 2) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph
                                        #- einsum("ka,kb,ib->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph) # could be cashed
                                        #- einsum("ic,jc,ja->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph)) # could be cashed
                + sqrt(omega / 2) * (#einsum("kc,kjic,ja->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph) # could be cashed
                                    #+ einsum("kc,kacb,ib->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph) # could be cashed
                                    + einsum("ka,jkib,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph)
                                    + einsum("ic,jabc,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph))
                + gs_part * ampl.gs.as_float()
                #+ sqrt(omega / 2) * ( # gs_ph contribution
                #        - einsum("kaic,kc->ia", hf.ovov, mp.qed_t1(b.ov))
                #        + omega * mp.qed_t1(b.ov)
                #        - 0.5 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t1(b.ov))
                #        + 0.5 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t1(b.ov))
                #        - 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                #        #+ 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                #        + 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                #        #- 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                #        + einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))
                #        * ampl.gs.as_float()
        ))
    return AdcBlock(apply, diagonal)

#def block_ph_ph_2_couple(hf, mp, intermediates): #for testing
#    if hasattr(hf, "qed_hf"):
#        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
#    #omega = float(ReferenceState.get_qed_omega(hf))
#    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
#    return AdcBlock(lambda ampl: 0, diagonal)


def block_ph_ph_2_phot_couple(hf, mp, intermediates): #one could cash some of the terms here
    if hasattr(hf, "qed_hf"):
        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
    omega = float(ReferenceState.get_qed_omega(hf))
    gs_part = intermediates.adc2_qed_ph_ph_2_phot_couple_gs_part
    qed_i1 = intermediates.adc2_qed_phot_couple_i1
    qed_i2 = intermediates.adc2_qed_phot_couple_i2

    #d_oo = zeros_like(hf.foo)
    #d_vv = zeros_like(hf.fvv)
    #d_oo.set_mask("ii", 1.0)
    #d_vv.set_mask("aa", 1.0)


    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    def apply(ampl):
        return AmplitudeVector(ph=(0.5 * sqrt(omega / 2) * (einsum("kc,kc,acik,ia->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), 
                                                direct_sum("ia-kc->acik", mp.df(b.ov), mp.df(b.ov)), ampl.ph1)
                                        - einsum("ka,kb,ik,ib->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.oo), ampl.ph1)
                                                einsum("ic,kc->ik", mp.df(b.ov), - mp.df(b.ov)), ampl.ph1) # this is different from mp.diff_df, but why???
                                        - einsum("ic,jc,ac,ja->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.vv), ampl.ph1))
                                                einsum("ka,kc->ac", mp.df(b.ov), - mp.df(b.ov)), ampl.ph1)) # this is different from mp.diff_df, but why???
                + einsum("ib,ab->ia", ampl.ph1, qed_i1)
                + einsum("ij,ja->ia", qed_i2, ampl.ph1)
                - 0.5 * sqrt(omega / 2) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph1
                                        #- einsum("kb,ka,ib->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph1)
                                        #- einsum("jc,ic,ja->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph1))
                + sqrt(omega / 2) * (#einsum("kc,kijc,ja->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph1)
                                    #+ einsum("kc,kbca,ib->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph1)
                                    + einsum("kb,ikja,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph1)
                                    + einsum("jc,ibac,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph1))
                #- sqrt(omega / 2) * einsum("kc,ikac->ia", mp.qed_t1(b.ov), hf.oovv) * ampl.gs1.as_float() #gs_ph part
                + gs_part * ampl.gs1.as_float()
                #+ sqrt(omega / 2) * ( #gs_ph part
                #        - 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv))
                #        + 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                #        + 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo))
                #        - 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                #        + einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))
                #        * ampl.gs1.as_float()
        ))
    return AdcBlock(apply, diagonal)

#def block_ph_ph_2_phot_couple(hf, mp, intermediates): #for testing
#    if hasattr(hf, "qed_hf"):
#        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
#    #omega = float(ReferenceState.get_qed_omega(hf))
#    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
#    return AdcBlock(lambda ampl: 0, diagonal)




def block_ph_ph_2_phot(hf, mp, intermediates): #one could cash some of the terms here
    if hasattr(hf, "qed_hf"):
        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
    omega = float(ReferenceState.get_qed_omega(hf))
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2
    #intermediates expect a dimensionality unequal zero, so qed_i0 cannot be cashed this way, but its just a sum over two indices
    #qed_i0_terms = [(mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))]
    #qed_i0 = 2 * sum(qed_t1_df.dot(qed_t1) for qed_t1_df, qed_t1 in qed_i0_terms)
    #qed_i1 = intermediates.adc2_qed_i1
    #qed_i2 = intermediates.adc2_qed_i2

    term_t2_eri = (
        + einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
        + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo)
    ).evaluate()

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    #omega = float(ReferenceState.get_qed_omega(hf))
    #qed_i0_terms = [(mp.qed_t1_df(b.ov), mp.qed_t1(b.ov))]
    #qed_i0 = 2 * sum(qed_t1_df.dot(qed_t1) for qed_t1_df, qed_t1 in qed_i0_terms)
    qed_i1 = intermediates.adc2_qed_i1
    qed_i2 = intermediates.adc2_qed_i2
    qed_i1_0 = intermediates.adc2_qed_i1_0
    qed_i2_0 = intermediates.adc2_qed_i2_0
    gs_part = intermediates.adc2_qed_ph_ph_2_phot_gs_part
    diagonal = AmplitudeVector(ph=(
                + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
                - einsum("IaIa->Ia", hf.ovov)
                - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
                + (-omega/2) * (sqrt(2) - 0.5) * ( # instead of 1 should be sqrt(2)
                - 2 * direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
                + 2 * einsum("ia,ia->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))
                + direct_sum("a+i->ia", qed_i1_0.diagonal(), qed_i2_0.diagonal())
                - einsum("ka,ikia->ia", mp.qed_t0(b.ov), hf.ooov)
                - einsum("ic,iaac->ia", mp.qed_t0(b.ov), hf.ovvv)
                + (1 - sqrt(2)) * omega * einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * einsum("ii,aa->ia", d_oo, d_vv)
                #- (omega/2) * einsum("ia,ia->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) # reintroduced (actually canceled from -E_0 (2))
                #- (1/4) * einsum("ia,ia->", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov))
                #- (1/4) * einsum("ijab,ijab->", mp.td2(b.oovv), einsum("ia,jb->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) - einsum("ib,ja->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)))
        ))

    #print("with d_vv", d_vv * mp.qed_t1_df(b.vv))
    #print("without d_vv", mp.qed_t1_df(b.vv))
    def apply(ampl):
        return AmplitudeVector(ph=(
                    + einsum("ib,ab->ia", ampl.ph1, i1)
                    - einsum("ij,ja->ia", i2, ampl.ph1)
                    - einsum("jaib,jb->ia", hf.ovov, ampl.ph1)    # 1
                    - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph1)  # 2
                    + (-omega/2) * (sqrt(2) - 0.5) * ( 
                    - 2 * einsum("ib,ab->ia", ampl.ph1, qed_i1)
                    - 2 * einsum("ij,ja->ia", qed_i2, ampl.ph1)
                    + (mp.qed_t1(b.ov) * mp.qed_t1_df(b.ov).dot(ampl.ph1) 
                            + mp.qed_t1_df(b.ov) * mp.qed_t1(b.ov).dot(ampl.ph1)))
                    + einsum("ib,ab->ia", ampl.ph1, qed_i1_0)
                    + einsum("ij,ja->ia", qed_i2_0, ampl.ph1)
                    #+ (1/2) * (einsum("ka,jkib,jb->ia", mp.qed_t0(b.ov), hf.ooov, ampl.ph1) #this can be done by symmetrize a,b
                    #        + einsum("kb,jkia,jb->ia", mp.qed_t0(b.ov), hf.ooov, ampl.ph1))
                    #+ (1/2) * (einsum("ic,jabc,jb->ia", mp.qed_t0(b.ov), hf.ovvv, ampl.ph1) #this can be done by symmetrize i,j
                    #        + einsum("jc,iabc,jb->ia", mp.qed_t0(b.ov), hf.ovvv, ampl.ph1))
                    + einsum("ijab,jb->ia", einsum("jkib,ka->ijab", hf.ooov, mp.qed_t0(b.ov)).symmetrise(2, 3)
                                            + einsum("ic,jabc->ijab", mp.qed_t0(b.ov), hf.ovvv).symmetrise(0, 1), ampl.ph1)
                    + (1 - sqrt(2)) * omega * einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph1 
                    #+ omega * einsum("ii,aa->ia", d_oo, d_vv) * ampl.gs1.as_float() #this (and following) is from the gs_ph contribution
                    #- (omega / 2) * (sqrt(2) - 1) * (einsum("ic,ac->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv))
                    #                                - einsum("ic,ac->ia", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv)) 
                    #                                - einsum("ka,ik->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo))
                    #                                + einsum("ka,ik->ia", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))) 
                    #                                * ampl.gs1.as_float()
                    + gs_part * ampl.gs1.as_float()
                    #+ (0.25 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov)) #this is from the gs_ph contribution
                    #        - 0.25 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                    #        + sqrt(2) * (
                    #        - (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv))
                    #        + (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                    #        + (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo))
                    #        - (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                    #        )
                    #        - 0.5 * einsum("ikac,kc->ia", mp.t2oo, mp.qed_t0_df(b.ov))
                    #        + 0.5 * omega * mp.qed_t0(b.ov)) * ampl.gs1.as_float()
                    #- (omega/2) * einsum("ia,ia->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph # reintroduced (actually canceled from -E_0 (2))
                    #- (1/4) * einsum("ia,ia->", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov)) * ampl.ph
                    #- (1/4) * einsum("ijab,ijab->", mp.td2(b.oovv), einsum("ia,jb->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov)) - einsum("ib,ja->ijab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov))) * ampl.ph
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_2_phot2(hf, mp, intermediates): #one could cash some of the terms here
    if hasattr(hf, "qed_hf"):
        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
    omega = float(ReferenceState.get_qed_omega(hf))
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2

    term_t2_eri = (
        + einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
        + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo)
    ).evaluate()

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    qed_i1 = intermediates.adc2_qed_i1
    qed_i2 = intermediates.adc2_qed_i2
    qed_i1_0 = intermediates.adc2_qed_i1_0
    qed_i2_0 = intermediates.adc2_qed_i2_0
    gs_part = intermediates.adc2_qed_ph_ph_2_phot2_gs_part
    diagonal = AmplitudeVector(ph=(
                + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
                - einsum("IaIa->Ia", hf.ovov)
                - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
                + (-omega/2) * (sqrt(3) - 0.5) * ( 
                - 2 * direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
                + 2 * einsum("ia,ia->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))
                + direct_sum("a+i->ia", qed_i1_0.diagonal(), qed_i2_0.diagonal())
                - einsum("ka,ikia->ia", mp.qed_t0(b.ov), hf.ooov)
                - einsum("ic,iaac->ia", mp.qed_t0(b.ov), hf.ovvv)
                + (1 - sqrt(3)) * omega * einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * einsum("ii,aa->ia", d_oo, d_vv)
        ))

    def apply(ampl):
        return AmplitudeVector(ph=(
                    + einsum("ib,ab->ia", ampl.ph2, i1)
                    - einsum("ij,ja->ia", i2, ampl.ph2)
                    - einsum("jaib,jb->ia", hf.ovov, ampl.ph2)    # 1
                    - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph2)  # 2
                    + (-omega/2) * (sqrt(3) - 0.5) * ( 
                    - 2 * einsum("ib,ab->ia", ampl.ph2, qed_i1)
                    - 2 * einsum("ij,ja->ia", qed_i2, ampl.ph2)
                    + (mp.qed_t1(b.ov) * mp.qed_t1_df(b.ov).dot(ampl.ph2) 
                            + mp.qed_t1_df(b.ov) * mp.qed_t1(b.ov).dot(ampl.ph2)))
                    + einsum("ib,ab->ia", ampl.ph2, qed_i1_0)
                    + einsum("ij,ja->ia", qed_i2_0, ampl.ph2)
                    + einsum("ijab,jb->ia", einsum("jkib,ka->ijab", hf.ooov, mp.qed_t0(b.ov)).symmetrise(2, 3)
                                            + einsum("ic,jabc->ijab", mp.qed_t0(b.ov), hf.ovvv).symmetrise(0, 1), ampl.ph2)
                    + (1 - sqrt(3)) * omega * einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph2 
                    + gs_part * ampl.gs2.as_float()
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_2_couple_inner(hf, mp, intermediates): #one could cash some of the terms here
    if hasattr(hf, "qed_hf"):
        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
    omega = float(ReferenceState.get_qed_omega(hf))
    gs_part = intermediates.adc2_qed_ph_ph_2_couple_inner_gs_part
    qed_i1 = intermediates.adc2_qed_couple_i1
    qed_i2 = intermediates.adc2_qed_couple_i2

    #d_oo = zeros_like(hf.foo)
    #d_vv = zeros_like(hf.fvv)
    #d_oo.set_mask("ii", 1.0)
    #d_vv.set_mask("aa", 1.0)

    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    def apply(ampl):
        return AmplitudeVector(ph=(0.5 * sqrt(omega / 2) * (einsum("kc,kc,acik,ia->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), 
                                                direct_sum("ia-kc->acik", mp.df(b.ov), mp.df(b.ov)), ampl.ph1)
                                        - einsum("kb,ka,ik,ib->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.oo), ampl.ph)
                                                einsum("ic,kc->ik", mp.df(b.ov), - mp.df(b.ov)), ampl.ph1) # this is different from mp.diff_df, but why???
                                        - einsum("jc,ic,ac,ja->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.vv), ampl.ph))
                                                einsum("ka,kc->ac", mp.df(b.ov), - mp.df(b.ov)), ampl.ph1)) # this is different from mp.diff_df, but why???
                + sqrt(2) * einsum("ib,ab->ia", ampl.ph1, qed_i1)
                + sqrt(2) * einsum("ij,ja->ia", qed_i2, ampl.ph1)
                - 0.5 * sqrt(omega) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph1
                                        #- einsum("ka,kb,ib->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph) # could be cashed
                                        #- einsum("ic,jc,ja->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph)) # could be cashed
                + sqrt(omega) * (#einsum("kc,kjic,ja->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph) # could be cashed
                                    #+ einsum("kc,kacb,ib->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph) # could be cashed
                                    + einsum("ka,jkib,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph1)
                                    + einsum("ic,jabc,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph1))
                + gs_part * ampl.gs1.as_float()
                + (sqrt(2) - 1) * 0.5 * sqrt(omega/2) * einsum("jb,jb->", mp.qed_t0(b.ov), ampl.ph1) * mp.qed_t1_df(b.ov)
                - (sqrt(2) - 1) * 0.5 * sqrt(omega/2) * (
                    (einsum("kc,kc->", mp.qed_t1_df(b.ov), mp.qed_t0(b.ov)) + einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t0_df(b.ov))) * ampl.ph1
                    - einsum("kb,ka,ib->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph1)
                    - einsum("kb,ka,ib->ia", mp.qed_t0_df(b.ov), mp.qed_t1(b.ov), ampl.ph1)
                    - einsum("jc,ic,ja->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph1)
                    - einsum("jc,ic,ja->ia", mp.qed_t0_df(b.ov), mp.qed_t1(b.ov), ampl.ph1)
                    + einsum("jb,ia,jb->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph1)
                    + einsum("jb,ia,jb->ia", mp.qed_t0_df(b.ov), mp.qed_t1(b.ov), ampl.ph1)
                )
        ))
    return AdcBlock(apply, diagonal)

#def block_ph_ph_2_couple(hf, mp, intermediates): #for testing
#    if hasattr(hf, "qed_hf"):
#        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
#    #omega = float(ReferenceState.get_qed_omega(hf))
#    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
#    return AdcBlock(lambda ampl: 0, diagonal)


def block_ph_ph_2_phot_couple_inner(hf, mp, intermediates): #one could cash some of the terms here
    if hasattr(hf, "qed_hf"):
        raise NotImplementedError("QED-ADC(2) has not been implemented with qed_hf reference")
    omega = float(ReferenceState.get_qed_omega(hf))
    gs_part = intermediates.adc2_qed_ph_ph_2_phot_couple_inner_gs_part
    qed_i1 = intermediates.adc2_qed_phot_couple_i1
    qed_i2 = intermediates.adc2_qed_phot_couple_i2

    #d_oo = zeros_like(hf.foo)
    #d_vv = zeros_like(hf.fvv)
    #d_oo.set_mask("ii", 1.0)
    #d_vv.set_mask("aa", 1.0)


    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    def apply(ampl):
        return AmplitudeVector(ph=(0.5 * sqrt(omega / 2) * (einsum("kc,kc,acik,ia->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), 
                                                direct_sum("ia-kc->acik", mp.df(b.ov), mp.df(b.ov)), ampl.ph2)
                                        - einsum("ka,kb,ik,ib->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.oo), ampl.ph1)
                                                einsum("ic,kc->ik", mp.df(b.ov), - mp.df(b.ov)), ampl.ph2) # this is different from mp.diff_df, but why???
                                        - einsum("ic,jc,ac,ja->ia", mp.qed_t0(b.ov), mp.qed_t1(b.ov), #mp.diff_df(b.vv), ampl.ph1))
                                                einsum("ka,kc->ac", mp.df(b.ov), - mp.df(b.ov)), ampl.ph2)) # this is different from mp.diff_df, but why???
                + einsum("ib,ab->ia", ampl.ph2, qed_i1)
                + einsum("ij,ja->ia", qed_i2, ampl.ph2)
                - 0.5 * sqrt(omega / 2) * einsum("kc,kc->", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph2
                                        #- einsum("kb,ka,ib->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph1)
                                        #- einsum("jc,ic,ja->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph1))
                + sqrt(omega / 2) * (#einsum("kc,kijc,ja->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph1)
                                    #+ einsum("kc,kbca,ib->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph1)
                                    + einsum("kb,ikja,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph2)
                                    + einsum("jc,ibac,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph2))
                #- sqrt(omega / 2) * einsum("kc,ikac->ia", mp.qed_t1(b.ov), hf.oovv) * ampl.gs1.as_float() #gs_ph part
                + gs_part * ampl.gs2.as_float()
                #+ sqrt(omega / 2) * ( #gs_ph part
                #        - 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv))
                #        + 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                #        + 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo))
                #        - 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                #        + einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))
                #        * ampl.gs1.as_float()
                + (sqrt(2) - 1) * 0.5 * sqrt(omega/2) * einsum("jb,jb->", mp.qed_t1_df(b.ov), ampl.ph2) * mp.qed_t0(b.ov)
                - (sqrt(2) - 1) * 0.5 * sqrt(omega/2) * (
                    (einsum("kc,kc->", mp.qed_t1_df(b.ov), mp.qed_t0(b.ov)) + einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t0_df(b.ov))) * ampl.ph2
                    - einsum("ka,kb,ib->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
                    - einsum("ka,kb,ib->ia", mp.qed_t0_df(b.ov), mp.qed_t1(b.ov), ampl.ph2)
                    - einsum("ic,ij,ja->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
                    - einsum("ic,jc,ja->ia", mp.qed_t0_df(b.ov), mp.qed_t1(b.ov), ampl.ph2)
                    + einsum("ia,jb,jb->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
                    + einsum("ia,jb,jb->ia", mp.qed_t0_df(b.ov), mp.qed_t1(b.ov), ampl.ph2)
                )
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_2_couple_edge(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    def apply(ampl):
        return - (omega/2) * sqrt(2) * (
            einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph
            - einsum("ka,kb,ib->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph)
            - einsum("ic,jc,ja->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph)
            + einsum("ia,jb,jb->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph)
            + (einsum("ic,ac->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) 
                - einsum("ka,ik->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov))) * ampl.gs.as_float() # gs-part
        )
    return AdcBlock(apply, diagonal)



def block_ph_ph_2_phot_couple_edge(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    diagonal = AmplitudeVector(ph=mp.df(b.ov).zeros_like())
    def apply(ampl):
        return - (omega/2) * sqrt(2) * (
            einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph2
            - einsum("kb,ka,ib->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
            - einsum("jc,ic,ja->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
            + einsum("jb,ia,jb->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
        )
    return AdcBlock(apply, diagonal)



#
# 2nd order coupling
#
def block_ph_pphh_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", pia_ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, pib_ovvv)
            + einsum("icab,jkcd,jkbd->ia", hf.ovvv, ampl.pphh, mp.t2oo)  # 2nd
            + einsum("ijka,jlbc,klbc->ia", hf.ooov, mp.t2oo, ampl.pphh)  # 2nd
        ))
    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(ph=(1 / sqrt(2)) * (
            + 2.0 * einsum("lKIc,lKac->Ia", pia_occv, ampl.pphh)
            - einsum("lIcd,lacd->Ia", ampl.pphh, pib_ovvv)
            - einsum("jIKa,ljcd,lKcd->Ia", hf.occv, mp.t2oo, ampl.pphh)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_2(hf, mp, intermediates):
    pia_ooov = intermediates.adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(
            (
                + einsum("ic,jcab->ijab", ampl.ph, pib_ovvv)
                + einsum("lkic,kc,jlab->ijab", hf.ooov, ampl.ph, mp.t2oo)  # 2st
            ).antisymmetrise(0, 1)
            + (
                - einsum("ijka,kb->ijab", pia_ooov, ampl.ph)
                - einsum("ijac,kbcd,kd->ijab", mp.t2oo, hf.ovvv, ampl.ph)  # 2st
            ).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_2(hf, mp, intermediates):
    pia_occv = intermediates.cvs_adc3_pia
    pib_ovvv = intermediates.adc3_pib

    def apply(ampl):
        return AmplitudeVector(pphh=(1 / sqrt(2)) * (
            - 2.0 * einsum("jIKa,Kb->jIab", pia_occv, ampl.ph).antisymmetrise(2, 3)
            - einsum("Ic,jcab->jIab", ampl.ph, pib_ovvv)
            - einsum("lKIc,Kc,jlab->jIab", hf.occv, ampl.ph, mp.t2oo)
        ))
    return AdcBlock(apply, 0)


#
# 3rd order main
#
def block_ph_ph_3(hf, mp, intermediates):
    if hf.has_core_occupied_space:
        m11 = intermediates.cvs_adc3_m11
    else:
        m11 = intermediates.adc3_m11
    diagonal = AmplitudeVector(ph=einsum("iaia->ia", m11))

    def apply(ampl):
        return AmplitudeVector(ph=einsum("iajb,jb->ia", m11, ampl.ph))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_3 = block_ph_ph_3


#
# Intermediates
#

@register_as_intermediate
def adc2_i1(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.fvv term.
    return hf.fvv + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def adc2_i2(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.foo term.
    return hf.foo - 0.5 * einsum("ikab,jkab->ij", mp.t2oo, hf.oovv).symmetrise()

# qed intermediates for adc2, without the factor of (omega/2), which is added in the actual matrix builder
@register_as_intermediate
def adc2_qed_i1(hf, mp, intermediates):
    #return (1/2) * einsum("kb,ka->ab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov))
    return (1/2) * (einsum("kb,ka->ab", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) 
                    + einsum("ka,kb->ab", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))


@register_as_intermediate
def adc2_qed_i2(hf, mp, intermediates):
    #return (1/2) * einsum("jc,ic->ij", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov))
    return (1/2) * (einsum("jc,ic->ij", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) 
                    + einsum("ic,jc->ij", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))

#qed intermediates for adc2, for non-qed-hf input
@register_as_intermediate
def adc2_qed_i1_0(hf, mp, intermediates):
    #return (1/2) * einsum("kb,ka->ab", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov))
    return ((1/8) * (einsum("kb,ka->ab", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov)) 
                    + einsum("ka,kb->ab", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov)))
            + einsum("kc,kacb->ab", mp.qed_t0(b.ov), hf.ovvv))


@register_as_intermediate
def adc2_qed_i2_0(hf, mp, intermediates):
    #return (1/2) * einsum("jc,ic->ij", mp.qed_t1_df(b.ov), mp.qed_t1_df(b.ov))
    return ((1/8) * (einsum("jc,ic->ij", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov)) 
                    + einsum("ic,jc->ij", mp.qed_t0(b.ov), mp.qed_t0_df(b.ov)))
            + einsum("kc,kjic->ij", mp.qed_t0(b.ov), hf.ooov))

@register_as_intermediate
def adc2_qed_ph_ph_2_i1(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    return (omega / 2) * intermediates.adc2_qed_i1.evaluate() + intermediates.adc2_qed_i1_0.evaluate()

@register_as_intermediate
def adc2_qed_ph_ph_2_i2(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    return (omega / 2) * intermediates.adc2_qed_i2.evaluate() + intermediates.adc2_qed_i2_0.evaluate()


@register_as_intermediate
def adc2_qed_ph_ph_2_gs_part(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (0.25 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov)) #this is from the gs_ph contribution
                            - 0.25 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                            - (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                            #+ (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                            + (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                            #- (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                            - 0.5 * einsum("ikac,kc->ia", mp.t2oo, mp.qed_t0_df(b.ov)))


@register_as_intermediate
def adc2_qed_ph_ph_2_couple_gs_part(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return sqrt(omega / 2) * ( # gs_ph contribution
                        - einsum("kaic,kc->ia", hf.ovov, mp.qed_t1(b.ov))
                        + omega * mp.qed_t1(b.ov)
                        - 0.5 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t1(b.ov))
                        + 0.5 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t1(b.ov))
                        - 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))


@register_as_intermediate
def adc2_qed_ph_ph_2_couple_inner_gs_part(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return sqrt(omega / 2) * ( # gs_ph contribution
                        - einsum("kaic,kc->ia", hf.ovov, mp.qed_t1(b.ov))
                        + 2 * omega * mp.qed_t1(b.ov)
                        - 0.5 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t1(b.ov))
                        + 0.5 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t1(b.ov))
                        + sqrt(2) * (
                        - 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv) - d_vv * mp.qed_t1_df(b.vv))
                        #+ 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo) - d_oo * mp.qed_t1_df(b.oo))
                        #- 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov))))
                        


@register_as_intermediate
def adc2_qed_ph_ph_2_phot_couple_gs_part(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return sqrt(omega / 2) * ( #gs_ph part
                        - 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo))
                        - 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))


@register_as_intermediate
def adc2_qed_ph_ph_2_phot_couple_inner_gs_part(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return sqrt(omega) * ( #gs_ph part
                        - 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("ic,ac->ia", mp.qed_t0(b.ov), d_vv * mp.qed_t1_df(b.vv))
                        + 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), mp.qed_t1_df(b.oo))
                        - 0.5 * einsum("ka,ik->ia", mp.qed_t0(b.ov), d_oo * mp.qed_t1_df(b.oo))
                        + einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))


@register_as_intermediate
def adc2_qed_ph_ph_2_phot_gs_part(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (0.25 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov)) #this is from the gs_ph contribution
                            - 0.25 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                            + sqrt(2) * (
                            - (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv))
                            + (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                            + (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo))
                            - (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                            )
                            - 0.5 * einsum("ikac,kc->ia", mp.t2oo, mp.qed_t0_df(b.ov))
                            + 0.5 * omega * mp.qed_t0(b.ov))


@register_as_intermediate
def adc2_qed_ph_ph_2_phot2_gs_part(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))

    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (0.25 * einsum("ik,ka->ia", mp.qed_t0_df(b.oo), mp.qed_t0(b.ov)) #this is from the gs_ph contribution
                            - 0.25 * einsum("ac,ic->ia", mp.qed_t0_df(b.vv), mp.qed_t0(b.ov))
                            + sqrt(3) * (
                            - (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.vv))
                            + (omega/2) * einsum("ic,ac->ia", mp.qed_t1(b.ov), d_vv * mp.qed_t1_df(b.vv))
                            + (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.oo))
                            - (omega/2) * einsum("ka,ik->ia", mp.qed_t1(b.ov), d_oo * mp.qed_t1_df(b.oo))
                            )
                            - 0.5 * einsum("ikac,kc->ia", mp.t2oo, mp.qed_t0_df(b.ov))
                            + omega * mp.qed_t0(b.ov))


@register_as_intermediate
def adc2_qed_couple_i1(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    return ( sqrt(omega / 2) * (0.5 * einsum("ka,kb->ab", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov))
                                    + einsum("kc,kacb->ab", mp.qed_t1(b.ov), hf.ovvv)))


@register_as_intermediate
def adc2_qed_couple_i2(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    return ( sqrt(omega / 2) * (0.5 * einsum("ic,jc->ij", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov)) 
                                    + einsum("kc,kjic->ij", mp.qed_t1(b.ov), hf.ooov))) 




@register_as_intermediate
def adc2_qed_phot_couple_i1(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    return ( sqrt(omega / 2) * (0.5 * einsum("kb,ka->ab", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov))
                                + einsum("kc,kbca->ab", mp.qed_t1(b.ov), hf.ovvv)))


@register_as_intermediate
def adc2_qed_phot_couple_i2(hf, mp, intermediates):
    omega = float(ReferenceState.get_qed_omega(hf))
    return ( sqrt(omega / 2) * (0.5 * einsum("jc,ic->ij", mp.qed_t0(b.ov), mp.qed_t1_df(b.ov)) 
                                    + einsum("kc,kijc->ij", mp.qed_t1(b.ov), hf.ooov))) 





def adc3_i1(hf, mp, intermediates):
    # Used for both CVS and general
    td2 = mp.td2(b.oovv)
    p0 = intermediates.cvs_p0 if hf.has_core_occupied_space else mp.mp2_diffdm

    t2eri_sum = (
        + einsum("jicb->ijcb", mp.t2eri(b.oovv, b.ov))  # t2eri4
        - 0.25 * mp.t2eri(b.oovv, b.vv)                 # t2eri5
    )
    return (
        (  # symmetrise a<>b
            + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ijac,ijcb->ab", mp.t2oo, t2eri_sum)
            - 2.0 * einsum("iabc,ic->ab", hf.ovvv, p0.ov)
        ).symmetrise()
        + einsum("iajb,ij->ab", hf.ovov, p0.oo)
        + einsum("acbd,cd->ab", hf.vvvv, p0.vv)
    )


def adc3_i2(hf, mp, intermediates):
    # Used only for general
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    # t2eri4 + t2eri3 / 4
    t2eri_sum = mp.t2eri(b.oovv, b.ov) + 0.25 * mp.t2eri(b.oovv, b.oo)
    return (
        (  # symmetrise i<>j
            + 0.5 * einsum("ikab,jkab->ij", mp.t2oo + td2, hf.oovv)
            - 1.0 * einsum("ikab,jkab->ij", mp.t2oo, t2eri_sum)
            + 2.0 * einsum("kija,ka->ij", hf.ooov, p0.ov)
        ).symmetrise()
        - einsum("ikjl,kl->ij", hf.oooo, p0.oo)
        - einsum("iajb,ab->ij", hf.ovov, p0.vv)
    )


def cvs_adc3_i2(hf, mp, intermediates):
    cvs_p0 = intermediates.cvs_p0
    return (
        + 2.0 * einsum("kIJa,ka->IJ", hf.occv, cvs_p0.ov).symmetrise()
        - 1.0 * einsum("kIlJ,kl->IJ", hf.ococ, cvs_p0.oo)
        - 1.0 * einsum("IaJb,ab->IJ", hf.cvcv, cvs_p0.vv)
    )


@register_as_intermediate
def adc3_m11(hf, mp, intermediates):
    td2 = mp.td2(b.oovv)
    p0 = mp.mp2_diffdm

    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()

    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    t2eri_sum = (
        + 2.0 * mp.t2eri(b.oovv, b.ov).symmetrise((0, 1), (2, 3))  # t2eri4
        + 0.5 * mp.t2eri(b.oovv, b.vv)                             # t2eri5
        + 0.5 * mp.t2eri(b.oovv, b.oo)                             # t2eri3
    )
    return (
        + einsum("ij,ab->iajb", d_oo, hf.fvv + i1)
        - einsum("ij,ab->iajb", hf.foo - i2, d_vv)
        - einsum("jaib->iajb", hf.ovov)
        - (  # symmetrise i<>j and a<>b
            + einsum("jkbc,ikac->iajb", hf.oovv, mp.t2oo + td2)
            - einsum("jkbc,ikac->iajb", mp.t2oo, t2eri_sum)
            - einsum("ibac,jc->iajb", hf.ovvv, 2.0 * p0.ov)
            - einsum("ikja,kb->iajb", hf.ooov, 2.0 * p0.ov)
            - einsum("jaic,bc->iajb", hf.ovov, p0.vv)
            + einsum("ik,jakb->iajb", p0.oo, hf.ovov)
            + einsum("ibkc,kajc->iajb", hf.ovov, 2.0 * t2sq)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("icjd,klac,klbd->iajb", hf.ovov, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("icjd,acbd->iajb", hf.ovov,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        + 0.5 * einsum("ikcd,jlcd,kalb->iajb", mp.t2oo, mp.t2oo, hf.ovov)
        - einsum("iljk,kalb->iajb", hf.oooo, t2sq)
        - einsum("idjc,acbd->iajb", t2sq, hf.vvvv)
    )


@register_as_intermediate
def cvs_adc3_m11(hf, mp, intermediates):
    i1 = adc3_i1(hf, mp, intermediates).evaluate()
    i2 = cvs_adc3_i2(hf, mp, intermediates).evaluate()
    t2sq = einsum("ikac,jkbc->iajb", mp.t2oo, mp.t2oo).evaluate()

    # Build two Kronecker deltas
    d_cc = zeros_like(hf.fcc)
    d_vv = zeros_like(hf.fvv)
    d_cc.set_mask("II", 1.0)
    d_vv.set_mask("aa", 1.0)

    return (
        + einsum("IJ,ab->IaJb", d_cc, hf.fvv + i1)
        - einsum("IJ,ab->IaJb", hf.fcc - i2, d_vv)
        - einsum("JaIb->IaJb", hf.cvcv)
        + (  # symmetrise I<>J and a<>b
            + einsum("JaIc,bc->IaJb", hf.cvcv, intermediates.cvs_p0.vv)
            - einsum("kIJa,kb->IaJb", hf.occv, 2.0 * intermediates.cvs_p0.ov)
        ).symmetrise((0, 2), (1, 3))
        # TODO This hack is done to avoid opt_einsum being smart and instantiating
        #      a tensor of dimension 6 (to avoid the vvvv tensor) in some cases,
        #      which is the right thing to do, but not yet supported.
        # + 0.5 * einsum("IcJd,klac,klbd->IaJb", hf.cvcv, mp.t2oo, mp.t2oo)
        + 0.5 * einsum("IcJd,acbd->IaJb", hf.cvcv,
                       einsum("klac,klbd->acbd", mp.t2oo, mp.t2oo))
        - einsum("lIkJ,kalb->IaJb", hf.ococ, t2sq)
    )


@register_as_intermediate
def adc3_pia(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ooov term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ooov                                            # 1st order
        - 2.0 * mp.t2eri(b.ooov, b.ov).antisymmetrise(0, 1)  # 2nd order
        - 0.5 * mp.t2eri(b.ooov, b.vv)                       # 2nd order
    )


@register_as_intermediate
def cvs_adc3_pia(hf, mp, intermediates):
    # Perturbation theory in CVS-ADC coupling block:
    #       1st                     2nd
    return hf.occv - einsum("jlac,lKIc->jIKa", mp.t2oo, hf.occv)


@register_as_intermediate
def adc3_pib(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.ovvv term.
    return (                          # Perturbation theory in ADC coupling block
        + hf.ovvv                                            # 1st order
        + 2.0 * mp.t2eri(b.ovvv, b.ov).antisymmetrise(2, 3)  # 2nd order
        - 0.5 * mp.t2eri(b.ovvv, b.oo)                       # 2nd order
    )
