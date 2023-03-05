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
    def tt2(self, space):
        """Second order triples amplitudes"""
        hf = self.reference_state
        t2 = self.t2(b.oovv).evaluate()
        denom = - direct_sum('ia,jkbc->ijkabc', self.df(b.ov),
                             direct_sum('jb,kc->jkbc', self.df(b.ov),
                             self.df(b.ov)))
        amp = (
            + einsum('idbc,jkad->ijkabc', hf.ovvv, t2)
            + einsum('idab,jkcd->ijkabc', hf.ovvv, t2)
            - einsum('jdab,ikcd->ijkabc', hf.ovvv, t2)
            + einsum('kdab,ijcd->ijkabc', hf.ovvv, t2)
            - einsum('jdbc,ikad->ijkabc', hf.ovvv, t2)
            + einsum('kdbc,ijad->ijkabc', hf.ovvv, t2)
            - einsum('idac,jkbd->ijkabc', hf.ovvv, t2)
            + einsum('jdac,ikbd->ijkabc', hf.ovvv, t2)
            - einsum('kdac,ijbd->ijkabc', hf.ovvv, t2)
            + einsum('jkla,ilbc->ijkabc', hf.ooov, t2)
            - einsum('ikla,jlbc->ijkabc', hf.ooov, t2)
            + einsum('ijla,klbc->ijkabc', hf.ooov, t2)
            + einsum('iklb,jlac->ijkabc', hf.ooov, t2)
            - einsum('jklb,ilac->ijkabc', hf.ooov, t2)
            - einsum('ijlb,klac->ijkabc', hf.ooov, t2)
            - einsum('iklc,jlab->ijkabc', hf.ooov, t2)
            + einsum('jklc,ilab->ijkabc', hf.ooov, t2)
            + einsum('ijlc,klab->ijkabc', hf.ooov, t2)
        )
        return amp / denom

    @cached_member_function
    def ts3(self, space):
        """Third order singles amplitudes"""
        hf = self.reference_state
        p0 = self.mp2_diffdm
        td2 = self.td2(b.oovv)
        tt2 = self.tt2(b.ooovvv)
        denom = - self.df(b.ov)
        amp = (
            - einsum('jaib,jb->ia', hf.ovov, p0.ov)
            + 0.5 * einsum('jkib,jkab->ia', hf.ooov, td2)
            + 0.5 * einsum('jabc,ijbc->ia', hf.ovvv, td2)
            + 0.25 * einsum('jkbc,ijkabc->ia', hf.oovv, tt2)
        )
        return amp / denom

    @cached_member_function
    def td3(self, space):
        """Third order doubles amplitudes"""
        hf = self.reference_state
        p0 = self.mp2_diffdm
        t2 = self.t2(b.oovv).evaluate()
        td2 = self.td2(b.oovv).evaluate()
        tt2 = self.tt2(b.ooovvv).evaluate()
        t2eri_vv = einsum('klbd,klcd->bc', t2, hf.oovv).evaluate()
        t2eri_oo = einsum('jlcd,klcd->jk', t2, hf.oovv).evaluate()
        t2eri_oovv = einsum('jlbd,klcd->jkbc', t2, hf.oovv).evaluate()
        denom = direct_sum('ia,jb->ijab', self.df(b.ov), self.df(b.ov))
        amp = (
            + 2 * einsum('jc,abic->ijab', p0.ov, hf.vvov).antisymmetrise(0, 1)
            + 2 * einsum('kb,kaij->ijab', p0.ov, hf.ovoo).antisymmetrise(2, 3)
            + 4 * einsum('ikac,kbjc->ijab', td2,
                         hf.ovov).antisymmetrise(0, 1).antisymmetrise(2, 3)
            - 0.5 * einsum('ijcd,abcd->ijab', td2, hf.vvvv)
            - 0.5 * einsum('klab,klij->ijab', td2, hf.oooo)
            + einsum('jklabc,klic->ijab', tt2, hf.ooov).antisymmetrise(0, 1)
            + einsum('ijkbcd,kacd->ijab', tt2, hf.ovvv).antisymmetrise(2, 3)
            - 0.25 * einsum('ijac,bc->ijab', t2, t2eri_vv)
            - 0.25 * einsum('ijad,bd->ijab', t2, t2eri_vv)
            + 0.25 * einsum('ijbc,ac->ijab', t2, t2eri_vv)
            + 0.25 * einsum('ijbd,ad->ijab', t2, t2eri_vv)
            + 0.25 * einsum('ijcd,klab,klcd->ijab', t2, t2, hf.oovv)
            - 0.25 * einsum('ikab,jk->ijab', t2, t2eri_oo)
            + 0.25 * einsum('ikac,jkbc->ijab', t2, t2eri_oovv)
            + 0.25 * einsum('ikad,jkbd->ijab', t2, t2eri_oovv)
            - 0.25 * einsum('ikbc,jkac->ijab', t2, t2eri_oovv)
            - 0.25 * einsum('ikbd,jkad->ijab', t2, t2eri_oovv)
            + 0.25 * einsum('il,jlab->ijab', t2eri_oo, t2)
            - 0.25 * einsum('ilab,jl->ijab', t2, t2eri_oo)
            + 0.25 * einsum('ilac,jlbc->ijab', t2, t2eri_oovv)
            + 0.25 * einsum('ilad,jlbd->ijab', t2, t2eri_oovv)
            - 0.25 * einsum('ilbc,jlac->ijab', t2, t2eri_oovv)
            - 0.25 * einsum('ilbd,jlad->ijab', t2, t2eri_oovv)
            + 0.25 * einsum('ik,jkab->ijab', t2eri_oo, t2)
        )
        return amp / denom

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

    @cached_property
    @timed_member_call(timer="timer")
    def mp3_diffdm(self):
        """
        Return the MP3 difference density in the MO basis. mp2_diffdm is included
        """
        t2 = self.t2(b.oovv)
        td2 = self.td2(b.oovv)
        ts3 = self.ts3(b.ov)
        tt2 = self.tt2(b.ooovvv)
        p0 = self.mp2_diffdm
        ret = self.mp2_diffdm

        ret.oo = p0.oo - 0.5 * (
            + einsum('jkab,ikab->ij', t2, td2)
            + einsum('jkab,ikab->ij', td2, t2)
        )
        ret.ov = p0.ov + (
            + ts3
            - einsum('jb,ijab->ia', p0.ov, t2)
            - 0.25 * einsum('jkbc,ijkabc->ia', t2, tt2)
        )
        ret.vv = p0.vv + 0.5 * (
            + einsum('ijac,ijbc->ab', td2, t2)
            + einsum('ijac,ijbc->ab', t2, td2)
        )
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
        elif level == 3:
            return self.reference_state.density + self.mp3_diffdm
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
        elif level == 3:
            return self.mp3_dipole_moment
        else:
            raise NotImplementedError("Only dipole moments for level 1 and 2"
                                      " are implemented.")

    @cached_member_function
    def energy_correction(self, level=2):
        """Obtain the MP energy correction at a particular level"""
        if level > 3:
            raise NotImplementedError(f"MP({level}) energy correction "
                                      "not implemented.")
        if level < 2:
            return 0.0
        hf = self.reference_state
        is_cvs = self.has_core_occupied_space
        if level == 2 and not is_cvs:
            terms = [(1.0, hf.oovv, self.t2oo)]
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
        )

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

    @cached_property
    def mp3_dipole_moment(self):
        # MP2_diffdm is included in mp3corr
        refstate = self.reference_state
        dipole_integrals = refstate.operators.electric_dipole
        mp3corr = -np.array([product_trace(comp, self.mp3_diffdm)
                            for comp in dipole_integrals])
        return refstate.dipole_moment + mp3corr


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
