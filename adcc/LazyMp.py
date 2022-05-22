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
import libadcc
import numpy as np

from .functions import direct_sum, evaluate, einsum
from .misc import cached_property, cached_member_function
from .ReferenceState import ReferenceState
from .OneParticleOperator import OneParticleOperator, product_trace
from .Intermediates import register_as_intermediate
from .timings import Timer, timed_member_call
from .MoSpaces import split_spaces
from . import block as b


class LazyMp:
    def __init__(self, hf):
        """
        Initialise the class dealing with the Møller-Plesset ground state.
        """
        if isinstance(hf, libadcc.HartreeFockSolution_i):
            hf = ReferenceState(hf)
        if not isinstance(hf, ReferenceState):
            raise TypeError("hf needs to be a ReferenceState "
                            "or a HartreeFockSolution_i")
        self.reference_state = hf
        self.mospaces = hf.mospaces
        self.timer = Timer()
        self.has_core_occupied_space = hf.has_core_occupied_space
        #for qed mp2
        self.get_qed_total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
        self.get_qed_total_dip.oo = hf.get_qed_total_dip(b.oo)
        self.get_qed_total_dip.ov = hf.get_qed_total_dip(b.ov)
        self.get_qed_total_dip.vv = hf.get_qed_total_dip(b.vv)
        self.get_qed_omega = hf.get_qed_omega

    def __getattr__(self, attr):
        # Shortcut some quantities, which are needed most often
        if attr.startswith("t2") and len(attr) == 4:  # t2oo, t2oc, t2cc
            xxvv = b.__getattr__(attr[2:4] + "vv")
            return self.t2(xxvv)
        else:
            raise AttributeError

    @cached_member_function
    def df(self, space):
        """Delta Fock matrix"""
        hf = self.reference_state
        s1, s2 = split_spaces(space)
        fC = hf.fock(s1 + s1).diagonal()
        fv = hf.fock(s2 + s2).diagonal()
        #print("occupied orbital energies", fC)
        #print("unoccupied orbital energies", fv)
        return direct_sum("-i+a->ia", fC, fv)

    @cached_member_function
    def t2(self, space):
        """T2 amplitudes"""
        hf = self.reference_state
        sp = split_spaces(space)
        assert all(s == b.v for s in sp[2:])
        eia = self.df(sp[0] + b.v)
        ejb = self.df(sp[1] + b.v)
        return (
            hf.eri(space) / direct_sum("ia+jb->ijab", eia, ejb).symmetrise((2, 3))
        )

    @cached_member_function
    def td2(self, space):
        """Return the T^D_2 term"""
        if space != b.oovv:
            raise NotImplementedError("T^D_2 term not implemented "
                                      f"for space {space}.")
        t2erit = self.t2eri(b.oovv, b.ov).transpose((1, 0, 2, 3))
        denom = direct_sum(
            'ia,jb->ijab', self.df(b.ov), self.df(b.ov)
        ).symmetrise(0, 1)
        return (
            + 4.0 * t2erit.antisymmetrise(2, 3).antisymmetrise(0, 1)
            - 0.5 * self.t2eri(b.oovv, b.vv)
            - 0.5 * self.t2eri(b.oovv, b.oo)
        ) / denom

    @cached_member_function
    def t2eri(self, space, contraction):
        """
        Return the T2 tensor with ERI tensor contraction intermediates.
        These are called pi1 to pi7 in libadc.
        """
        hf = self.reference_state
        key = space + contraction
        expressions = {
            # space + contraction
            b.ooov + b.vv: ('ijbc,kabc->ijka', b.ovvv),
            b.ooov + b.ov: ('ilab,lkjb->ijka', b.ooov),
            b.oovv + b.oo: ('klab,ijkl->ijab', b.oooo),
            b.oovv + b.ov: ('jkac,kbic->ijab', b.ovov),
            b.oovv + b.vv: ('ijcd,abcd->ijab', b.vvvv),
            b.ovvv + b.oo: ('jkbc,jkia->iabc', b.ooov),
            b.ovvv + b.ov: ('ijbd,jcad->iabc', b.ovvv),
        }
        if key not in expressions:
            raise NotImplementedError("t2eri intermediate not implemented "
                                      f"for space '{space}' and contraction "
                                      f"'{contraction}'.")
        contraction_str, eri_block = expressions[key]
        return einsum(contraction_str, self.t2oo, hf.eri(eri_block))

    @cached_property
    @timed_member_call(timer="timer")
    def mp2_diffdm(self):
        """
        Return the MP2 differensce density in the MO basis.
        """
        hf = self.reference_state
        ret = OneParticleOperator(self.mospaces, is_symmetric=True)
        # NOTE: the following 3 blocks are equivalent to the cvs_p0 intermediates
        # defined at the end of this file
        ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        ret.ov = -0.5 * (
            + einsum("ijbc,jabc->ia", self.t2oo, hf.ovvv)
            + einsum("jkib,jkab->ia", hf.ooov, self.t2oo)
        ) / self.df(b.ov)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)

        if self.has_core_occupied_space:
            # additional terms to "revert" CVS for ground state density
            ret.oo += -0.5 * einsum("iLab,jLab->ij", self.t2oc, self.t2oc)
            ret.ov += -0.5 * (
                + einsum("jMib,jMab->ia", hf.ocov, self.t2oc)
                + einsum("iLbc,Labc->ia", self.t2oc, hf.cvvv)
                + einsum("kLib,kLab->ia", hf.ocov, self.t2oc)
                + einsum("iMLb,LMab->ia", hf.occv, self.t2cc)
                - einsum("iLMb,LMab->ia", hf.occv, self.t2cc)
            ) / self.df(b.ov)
            ret.vv += (
                + 0.5 * einsum("IJac,IJbc->ab", self.t2cc, self.t2cc)
                + 1.0 * einsum("kJac,kJbc->ab", self.t2oc, self.t2oc)
            )
            # compute extra CVS blocks
            ret.cc = -0.5 * (
                + einsum("kIab,kJab->IJ", self.t2oc, self.t2oc)
                + einsum('LIab,LJab->IJ', self.t2cc, self.t2cc)
            )
            ret.co = -0.5 * (
                + einsum("kIab,kjab->Ij", self.t2oc, self.t2oo)
                + einsum("ILab,jLab->Ij", self.t2cc, self.t2oc)
            )
            ret.cv = -0.5 * (
                - einsum("jIbc,jabc->Ia", self.t2oc, hf.ovvv)
                + einsum("jkIb,jkab->Ia", hf.oocv, self.t2oo)
                + einsum("jMIb,jMab->Ia", hf.occv, self.t2oc)
                + einsum("ILbc,Labc->Ia", self.t2cc, hf.cvvv)
                + einsum("kLIb,kLab->Ia", hf.occv, self.t2oc)
                + einsum("LMIb,LMab->Ia", hf.cccv, self.t2cc)
            ) / self.df(b.cv)
        ret.reference_state = self.reference_state
        return evaluate(ret)

    def density(self, level=2):
        """
        Return the MP density in the MO basis with all corrections
        up to the specified order of perturbation theory
        """
        if level == 1:
            return self.reference_state.density
        elif level == 2:
            return self.reference_state.density + self.mp2_diffdm
        else:
            raise NotImplementedError("Only densities for level 1 and 2"
                                      " are implemented.")

    def dipole_moment(self, level=2):
        """
        Return the MP dipole moment at the specified level of
        perturbation theory.
        """
        if level == 1:
            return self.reference_state.dipole_moment
        elif level == 2:
            return self.mp2_dipole_moment
        else:
            raise NotImplementedError("Only dipole moments for level 1 and 2"
                                      " are implemented.")

    @cached_member_function
    def qed_t1_df(self, space):
        #if space != b.ov:
            #raise NotImplementedError("qed_t1 term not implemented "
            #                          f"for space {space}.")
        total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
        if space == b.oo:
            total_dip.oo = self.get_qed_total_dip.oo
            return total_dip.oo
        elif space == b.ov:
            total_dip.ov = self.get_qed_total_dip.ov
            return total_dip.ov
        elif space == b.vv:
            total_dip.vv = self.get_qed_total_dip.vv
            return total_dip.vv
        #return total_dip.ov #/ self.df(b.ov)

    @cached_member_function
    def qed_t1(self, space):
        """ Return new electronic singly excited amplitude in the first order correction to the wavefunction for qed for N=1 """
        if space != b.ov:
            raise NotImplementedError("qed_t1 term not implemented "
                                      f"for space {space}.")
        return self.qed_t1_df(b.ov) / self.df(b.ov) #einsum("kc,kc->kc", self.qed_t1(b.ov), self.df(b.ov))

    @cached_member_function
    def qed_t0_df(self, space):
        total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
        total_dip.oo = self.get_qed_total_dip.oo
        total_dip.ov = self.get_qed_total_dip.ov
        total_dip.vv = self.get_qed_total_dip.vv
        if space == b.ov:
            occ_sum = einsum("ka,ki->ia", total_dip.ov, total_dip.oo)
            virt_sum = einsum("ac,ic->ia", total_dip.vv, total_dip.ov)
            #return einsum("ka,ki->ia", total_dip.ov, total_dip.oo) - einsum("ac,ic->ia", total_dip.vv, total_dip.ov)
            #print(total_dip.ov)
            #return einsum("ia,ia->ia", occ_sum, virt_sum)
        elif space == b.oo:
            occ_sum = einsum("ki,kj->ij", total_dip.oo, total_dip.oo)
            virt_sum = einsum("ic,jc->ij", total_dip.ov, total_dip.ov)
        elif space == b.vv:
            occ_sum = einsum("ka,kb->ab", total_dip.ov, total_dip.ov)
            virt_sum = einsum("ac,bc->ab", total_dip.vv, total_dip.vv)
        return occ_sum - virt_sum

    @cached_member_function
    def qed_t0(self, space):
        """ Return new electronic singly excited amplitude in the first order correction to the wavefunction for qed for N=0 """
        if space != b.ov:
            raise NotImplementedError("qed_t0 term not implemented "
                                      f"for space {space}.")
        return self.qed_t0_df(b.ov) / self.df(b.ov)

    @cached_member_function
    def diff_df(self, space):
        if space == b.ov:
            raise NotImplementedError("This would not make sense to construct!!!")
        elif space == b.vv: # this returns (eps_a - eps_b)
            return einsum("ia,ib->ab", self.df(b.ov), - self.df(b.ov))
        elif space == b.oo: # this returns (- eps_i + eps_j)
            return einsum("ia,ja->ij", self.df(b.ov), - self.df(b.ov))



    @cached_member_function
    def energy_correction(self, level=2):
        """Obtain the MP energy correction at a particular level"""
        qed_mp2_correction = 0
        if level > 3:
            raise NotImplementedError(f"MP({level}) energy correction "
                                      "not implemented.")
        if level < 2: #for qed_mp1 from non-qed-hf also first corrections come into play...for now done in mp2 part here
            return 0.0
        hf = self.reference_state
        is_cvs = self.has_core_occupied_space
        if level == 2 and not is_cvs:
            terms = [(1.0, hf.oovv, self.t2oo)]
            mp2_correction = sum(
                -0.25 * pref * eri.dot(t2)
                for pref, eri, t2 in terms
            )
            if hasattr(hf, "coupling"):
                print("mp2 energy with two electron qed perturbation " + str(mp2_correction))
                #check if qed-hf (psi4.core.Wavefunction) input or non-qed-hf input (standard hf) is given
                total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
                omega, total_dip.ov = ReferenceState.get_qed_omega(hf), self.get_qed_total_dip.ov
                qed_terms = [(omega/2, total_dip.ov, self.qed_t1(b.ov))]
                qed_mp2_correction_1 = sum(
                    -pref * lambda_dip.dot(qed_t)
                    for pref, lambda_dip, qed_t in qed_terms
                )
                if hasattr(hf, "qed_hf"):
                    print("full qed MP2 energy correction (qed-hf) " + str(mp2_correction + qed_mp2_correction_1))
                else:
                    #total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
                    #omega, total_dip.ov = ReferenceState.get_qed_omega(hf), self.get_qed_total_dip.ov
                    #qed_terms_1 = [(omega/2, total_dip.ov, self.qed_t1(b.ov))]
                    #qed_mp2_correction_1 = sum(
                    #    -pref * lambda_dip.dot(qed_t)
                    #    for pref, lambda_dip, qed_t in qed_terms_1
                    #)
                    qed_terms_0 = [(1.0, self.qed_t0(b.ov), self.qed_t0_df(b.ov))]
                    qed_mp2_correction_0 = sum(
                        -0.25 * pref * ampl_t0.dot(ampl_t0_df)
                        for pref, ampl_t0, ampl_t0_df in qed_terms_0
                    )
                    #mp1 terms:
                    qed_mp1_additional_terms = [(0.5, total_dip.ov)]
                    qed_mp1_correction = sum(
                        pref * lambda_dip.dot(lambda_dip)
                        for pref, lambda_dip in qed_mp1_additional_terms
                    )
                    #print(self.qed_t0(b.ov))
                    #print(self.qed_t0_df(b.ov))
                    #print(qed_mp2_correction_0)
                    print("full qed MP2 energy correction (standard hf) " 
                    + str(mp2_correction + qed_mp2_correction_1 + qed_mp2_correction_0))
                    print("qed-mp1 correction, due to standard hf input " + str(qed_mp1_correction))
                    print("new qed-mp2 correction compared to qed-hf " + str(qed_mp2_correction_0))
                    #print("transition dipoles * coupling * sqrt(2 * freq)", total_dip.ov)
                    #print("orbital energy differences", self.df(b.ov))
        elif level == 2 and is_cvs:
            terms = [(1.0, hf.oovv, self.t2oo),
                     (2.0, hf.ocvv, self.t2oc),
                     (1.0, hf.ccvv, self.t2cc)]
        elif level == 3 and not is_cvs:
            terms = [(1.0, hf.oovv, self.td2(b.oovv))]
        elif level == 3 and is_cvs:
            raise NotImplementedError("CVS-MP3 energy correction not implemented.")
        return sum(
            -0.25 * pref * eri.dot(t2)
            for pref, eri, t2 in terms
        ) + qed_mp2_correction

    def energy(self, level=2):
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        if level == 0:
            # Sum of orbital energies ...
            raise NotImplementedError("Total MP(0) energy not implemented.")

        # Accumulator for all energy terms
        energies = [self.reference_state.energy_scf]
        for il in range(2, level + 1):
            energies.append(self.energy_correction(il))
        return sum(energies)

    def to_qcvars(self, properties=False, recurse=False, maxlevel=2):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        qcvars = {}
        for level in range(2, maxlevel + 1):
            try:
                mpcorr = self.energy_correction(level)
                qcvars[f"MP{level} CORRELATION ENERGY"] = mpcorr
                qcvars[f"MP{level} TOTAL ENERGY"] = self.energy(level)
            except NotImplementedError:
                pass
            except ValueError:
                pass

        if properties:
            for level in range(2, maxlevel + 1):
                try:
                    qcvars["MP2 DIPOLE"] = self.dipole_moment(level)
                except NotImplementedError:
                    pass

        if recurse:
            qcvars.update(self.reference_state.to_qcvars(properties, recurse))
        return qcvars

    @property
    def mp2_density(self):
        return self.density(2)

    @cached_property
    def mp2_dipole_moment(self):
        refstate = self.reference_state
        dipole_integrals = refstate.operators.electric_dipole
        mp2corr = -np.array([product_trace(comp, self.mp2_diffdm)
                             for comp in dipole_integrals])
        return refstate.dipole_moment + mp2corr


#
# Register cvs_p0 intermediate
#
@register_as_intermediate
def cvs_p0(hf, mp, intermediates):
    # NOTE: equal to mp2_diffdm if CVS applied for the density
    ret = OneParticleOperator(hf.mospaces, is_symmetric=True)
    ret.oo = -0.5 * einsum("ikab,jkab->ij", mp.t2oo, mp.t2oo)
    ret.ov = -0.5 * (+ einsum("ijbc,jabc->ia", mp.t2oo, hf.ovvv)
                     + einsum("jkib,jkab->ia", hf.ooov, mp.t2oo)) / mp.df(b.ov)
    ret.vv = 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, mp.t2oo)
    return ret
