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
import adcc.block as b
from adcc.OneParticleOperator import OneParticleOperator
from adcc.functions import direct_sum, einsum, evaluate
from adcc.AmplitudeVector import AmplitudeVector
from adcc.solver.conjugate_gradient import conjugate_gradient, default_print


def orbital_response_rhs(hf, g1a, g2a):
    """
    Build the right-hand side for solving the orbital
    response given amplitude-relaxed density matrices (method-specific)
    """
    # TODO: only add non-zero blocks to equations!

    if hf.has_core_occupied_space:
        ret_ov = -1.0 * (
            + 2.0 * einsum("JiKa,JK->ia", hf.cocv, g1a.cc)
            + 2.0 * einsum("icab,bc->ia", hf.ovvv, g1a.vv)
            + 2.0 * einsum('kija,jk->ia', hf.ooov, g1a.oo)
            + 2.0 * einsum("kiJa,Jk->ia", hf.oocv, g1a.co)
            - 2.0 * einsum("iJka,Jk->ia", hf.ocov, g1a.co)
            + 2.0 * einsum("iJKb,JaKb->ia", hf.occv, g2a.cvcv)  # 2PDMs start
            + 2.0 * einsum('lKJa,lKiJ->ia', g2a.occv, hf.ococ)
            + 1.0 * einsum('jabc,ijbc->ia', g2a.ovvv, hf.oovv)
            - 1.0 * einsum('JKab,JKib->ia', g2a.ccvv, hf.ccov)
            - 2.0 * einsum('jKab,jKib->ia', g2a.ocvv, hf.ocov)
            - 1.0 * einsum('jkab,jkib->ia', g2a.oovv, hf.ooov)
            - 2.0 * einsum('jcab,ibjc->ia', g2a.ovvv, hf.ovov)
            - 2.0 * einsum('iJKb,JaKb->ia', g2a.occv, hf.cvcv)
            - 1.0 * einsum('iJbc,Jabc->ia', g2a.ocvv, hf.cvvv)
            - 1.0 * einsum('ijbc,jabc->ia', g2a.oovv, hf.ovvv)
            + 1.0 * einsum('ibcd,abcd->ia', g2a.ovvv, hf.vvvv)
            - 1.0 * einsum('abcd,ibcd->ia', g2a.vvvv, hf.ovvv)  # cvs-adc2x
            + 2.0 * einsum('jakb,ijkb->ia', g2a.ovov, hf.ooov)  # cvs-adc2x
            + 2.0 * einsum('ibjc,jcab->ia', g2a.ovov, hf.ovvv)  # cvs-adc2x
            - 2.0 * einsum('iJkL,kLJa->ia', g2a.ococ, hf.occv)  # cvs-adc2x
        )

        ret_cv = -1.0 * (
            + 2.0 * einsum("JIKa,JK->Ia", hf.cccv, g1a.cc)
            + 2.0 * einsum("Icab,bc->Ia", hf.cvvv, g1a.vv)
            + 2.0 * einsum('kIja,jk->Ia', hf.ocov, g1a.oo)
            + 2.0 * einsum("kIJa,Jk->Ia", hf.occv, g1a.co)
            + 2.0 * einsum("JIka,Jk->Ia", hf.ccov, g1a.co)
            + 2.0 * einsum("IJKb,JaKb->Ia", hf.cccv, g2a.cvcv)  # 2PDMs start
            + 2.0 * einsum("Jcab,IbJc->Ia", hf.cvvv, g2a.cvcv)
            + 2.0 * einsum('lKJa,lKIJ->Ia', g2a.occv, hf.occc)
            - 1.0 * einsum('jabc,jIbc->Ia', g2a.ovvv, hf.ocvv)
            - 1.0 * einsum('JKab,JKIb->Ia', g2a.ccvv, hf.cccv)
            - 2.0 * einsum('jKab,jKIb->Ia', g2a.ocvv, hf.occv)
            - 1.0 * einsum('jkab,jkIb->Ia', g2a.oovv, hf.oocv)
            - 2.0 * einsum('jcab,jcIb->Ia', g2a.ovvv, hf.ovcv)
            - 1.0 * einsum('IJbc,Jabc->Ia', g2a.ccvv, hf.cvvv)
            + 2.0 * einsum('jIKb,jaKb->Ia', g2a.occv, hf.ovcv)
            + 1.0 * einsum('jIbc,jabc->Ia', g2a.ocvv, hf.ovvv)
            + 2.0 * einsum('kJIb,kJab->Ia', g2a.occv, hf.ocvv)
            - 1.0 * einsum('abcd,Ibcd->Ia', g2a.vvvv, hf.cvvv)  # cvs-adc2x
            - 2.0 * einsum('jakb,jIkb->Ia', g2a.ovov, hf.ocov)  # cvs-adc2x
            - 2.0 * einsum('jIlK,lKja->Ia', g2a.ococ, hf.ocov)  # cvs-adc2x
        )
        ret = AmplitudeVector(cv=ret_cv, ov=ret_ov)
    else:
        # equal to the ov block of the energy-weighted density
        # matrix when lambda_ov multipliers are zero
        w_ov = 0.5 * (
            + 1.0 * einsum("ijkl,klja->ia", hf.oooo, g2a.ooov)  # not in cvs-adc
            # - 1.0 * einsum("ibcd,abcd->ia", hf.ovvv, g2a.vvvv)
            - 1.0 * einsum("jkib,jkab->ia", hf.ooov, g2a.oovv)
            + 2.0 * einsum("ijkb,jakb->ia", hf.ooov, g2a.ovov)
            + 1.0 * einsum("ijbc,jabc->ia", hf.oovv, g2a.ovvv)
            - 2.0 * einsum("ibjc,jcab->ia", hf.ovov, g2a.ovvv)
        )
        w_ov = w_ov.evaluate()

        ret_ov = -1.0 * (
            2.0 * w_ov
            # - 1.0 * einsum("klja,ijkl->ia", hf.ooov, g2a.oooo)
            + 1.0 * einsum("abcd,ibcd->ia", hf.vvvv, g2a.ovvv)
            - 2.0 * einsum("jakb,ijkb->ia", hf.ovov, g2a.ooov)  # not in cvs-adc
            + 1.0 * einsum("jkab,jkib->ia", hf.oovv, g2a.ooov)  # not in cvs-adc
            + 2.0 * einsum("jcab,ibjc->ia", hf.ovvv, g2a.ovov)
            - 1.0 * einsum("jabc,ijbc->ia", hf.ovvv, g2a.oovv)
            + 2.0 * einsum("jika,jk->ia", hf.ooov, g1a.oo)
            + 2.0 * einsum("icab,bc->ia", hf.ovvv, g1a.vv)
        )
        ret = AmplitudeVector(ov=ret_ov)

    return ret


def energy_weighted_density_matrix(hf, g1o, g2a):
    if hf.has_core_occupied_space:
        # CVS-ADC0, CVS-ADC1, CVS-ADC2
        w = OneParticleOperator(hf)
        w.cc = - 0.5 * (
            + einsum("JKab,IKab->IJ", g2a.ccvv, hf.ccvv)
            + einsum("kJab,kIab->IJ", g2a.ocvv, hf.ocvv)
        )
        w.cc += (
            - hf.fcc
            - einsum("IK,JK->IJ", g1o.cc, hf.fcc)
            - einsum("ab,IaJb->IJ", g1o.vv, hf.cvcv)
            - einsum("KL,IKJL->IJ", g1o.cc, hf.cccc)
            - einsum("kl,kIlJ->IJ", g1o.oo, hf.ococ)
            + einsum("ka,kJIa->IJ", g1o.ov, hf.occv)
            + einsum("ka,kIJa->IJ", g1o.ov, hf.occv)
            + einsum("Ka,KJIa->IJ", g1o.cv, hf.cccv)
            + einsum("Ka,KIJa->IJ", g1o.cv, hf.cccv)
            - einsum("Lk,kILJ->IJ", g1o.co, hf.occc)
            - einsum("Lk,kJLI->IJ", g1o.co, hf.occc)
            - einsum("JbKc,IbKc->IJ", g2a.cvcv, hf.cvcv)
            - einsum("kJLa,kILa->IJ", g2a.occv, hf.occv)
            - einsum("kLJa,kLIa->IJ", g2a.occv, hf.occv)
            - einsum("kJmL,kImL->IJ", g2a.ococ, hf.ococ)  # cvs-adc2x
        )
        w.oo = - 0.5 * (
            + einsum("jKab,iKab->ij", g2a.ocvv, hf.ocvv)
            + einsum("jkab,ikab->ij", g2a.oovv, hf.oovv)
            + einsum("jabc,iabc->ij", g2a.ovvv, hf.ovvv)
        )
        w.oo += (
            - hf.foo
            - einsum("ij,ii->ij", g1o.oo, hf.foo)
            - einsum("KL,iKjL->ij", g1o.cc, hf.ococ)
            - einsum("ab,iajb->ij", g1o.vv, hf.ovov)
            - einsum("kl,ikjl->ij", g1o.oo, hf.oooo)
            - einsum("ka,ikja->ij", g1o.ov, hf.ooov)
            - einsum("ka,jkia->ij", g1o.ov, hf.ooov)
            - einsum("Ka,iKja->ij", g1o.cv, hf.ocov)
            - einsum("Ka,jKia->ij", g1o.cv, hf.ocov)
            + einsum("Lk,kijL->ij", g1o.co, hf.oooc)
            + einsum("Lk,kjiL->ij", g1o.co, hf.oooc)
            - einsum("jKLa,iKLa->ij", g2a.occv, hf.occv)
            - einsum("jKlM,iKlM->ij", g2a.ococ, hf.ococ)  # cvs-adc2x
            - einsum("jakb,iakb->ij", g2a.ovov, hf.ovov)  # cvs-adc2x
        )
        w.vv = - 0.5 * (
            + einsum("ibcd,iacd->ab", g2a.ovvv, hf.ovvv)
            + einsum("IJbc,IJac->ab", g2a.ccvv, hf.ccvv)
            + einsum("ijbc,ijac->ab", g2a.oovv, hf.oovv)
            + einsum("bcde,acde->ab", g2a.vvvv, hf.vvvv)  # cvs-adc2x
        )
        w.vv += (
            - einsum("ac,cb->ab", g1o.vv, hf.fvv)
            - einsum('JbKc,JaKc->ab', g2a.cvcv, hf.cvcv)
            - einsum("jKIb,jKIa->ab", g2a.occv, hf.occv)
            - einsum("idbc,idac->ab", g2a.ovvv, hf.ovvv)
            - einsum("iJbc,iJac->ab", g2a.ocvv, hf.ocvv)
            - einsum("ibjc,iajc->ab", g2a.ovov, hf.ovov)  # cvs-adc2x
        )
        w.co = 0.5 * (
            - einsum("JKab,iKab->Ji", g2a.ccvv, hf.ocvv)
            + einsum("kJab,ikab->Ji", g2a.ocvv, hf.oovv)
        )
        w.co += (
            + einsum("KL,iKLJ->Ji", g1o.cc, hf.occc)
            - einsum("ab,iaJb->Ji", g1o.vv, hf.ovcv)
            - einsum("kl,kilJ->Ji", g1o.oo, hf.oooc)
            - einsum("Ka,iKJa->Ji", g1o.cv, hf.occv)
            - einsum("Ka,JKia->Ji", g1o.cv, hf.ccov)
            - einsum("ka,ikJa->Ji", g1o.ov, hf.oocv)
            - einsum("ka,Jkia->Ji", g1o.ov, hf.coov)
            - einsum("Lk,kiLJ->Ji", g1o.co, hf.oocc)
            + einsum("Lk,iLkJ->Ji", g1o.co, hf.ococ)
            - einsum("Ik,jk->Ij", g1o.co, hf.foo)
            - einsum("JbKc,ibKc->Ji", g2a.cvcv, hf.ovcv)
            + einsum("kJLa,ikLa->Ji", g2a.occv, hf.oocv)
            - einsum("kLJa,kLia->Ji", g2a.occv, hf.ocov)
            + einsum("kJmL,ikmL->Ji", g2a.ococ, hf.oooc)  # cvs-adc2x
        )
        w.ov = 0.5 * (
            + einsum("jabc,ijbc->ia", g2a.ovvv, hf.oovv)
            - einsum("JKab,JKib->ia", g2a.ccvv, hf.ccov)
            - einsum("jkab,jkib->ia", g2a.oovv, hf.ooov)
            - einsum("abcd,ibcd->ia", g2a.vvvv, hf.ovvv)  # cvs-adc2x
        )
        w.ov += (
            - einsum("ij,ja->ia", hf.foo, g1o.ov)
            + einsum("JaKc,iJKc->ia", g2a.cvcv, hf.occv)
            + einsum("kLJa,iJkL->ia", g2a.occv, hf.ococ)
            - einsum("jKab,jKib->ia", g2a.ocvv, hf.ocov)
            - einsum("jcab,ibjc->ia", g2a.ovvv, hf.ovov)
            + einsum("jakb,ijkb->ia", g2a.ovov, hf.ooov)  # cvs-adc2x
        )
        w.cv = - 0.5 * (
            + einsum("jabc,jIbc->Ia", g2a.ovvv, hf.ocvv)
            + einsum("JKab,JKIb->Ia", g2a.ccvv, hf.cccv)
            + einsum("jkab,jkIb->Ia", g2a.oovv, hf.oocv)
            + einsum("abcd,Ibcd->Ia", g2a.vvvv, hf.cvvv)  # cvs-adc2x
        )
        w.cv += (
            - einsum("IJ,Ja->Ia", hf.fcc, g1o.cv)
            + einsum("JaKc,IJKc->Ia", g2a.cvcv, hf.cccv)
            + einsum("lKJa,lKIJ->Ia", g2a.occv, hf.occc)
            - einsum("kJab,kJIb->Ia", g2a.ocvv, hf.occv)
            - einsum("jcab,jcIb->Ia", g2a.ovvv, hf.ovcv)
            - einsum("jakb,jIkb->Ia", g2a.ovov, hf.ocov)  # cvs-adc2x
        )
    else:
        gi_oo = -0.5 * (
            # + 1.0 * einsum("jklm,iklm->ij", hf.oooo, g2a.oooo)
            + 1.0 * einsum("jabc,iabc->ij", hf.ovvv, g2a.ovvv)
            + 1.0 * einsum("klja,klia->ij", hf.ooov, g2a.ooov)
            + 2.0 * einsum("jkla,ikla->ij", hf.ooov, g2a.ooov)
            + 1.0 * einsum("jkab,ikab->ij", hf.oovv, g2a.oovv)
            + 2.0 * einsum("jakb,iakb->ij", hf.ovov, g2a.ovov)
        )
        gi_vv = -0.5 * (
            + 1.0 * einsum("kjib,kjia->ab", hf.ooov, g2a.ooov)
            # + einsum("bcde,acde->ab", hf.vvvv, g2a.vvvv)
            + 1.0 * einsum("ijcb,ijca->ab", hf.oovv, g2a.oovv)
            + 2.0 * einsum("jcib,jcia->ab", hf.ovov, g2a.ovov)
            + 1.0 * einsum("ibcd,iacd->ab", hf.ovvv, g2a.ovvv)
            + 2.0 * einsum("idcb,idca->ab", hf.ovvv, g2a.ovvv)
        )
        gi_oo = gi_oo.evaluate()
        gi_vv = gi_vv.evaluate()
        w = OneParticleOperator(hf)
        w.ov = 0.5 * (
            - 2.0 * einsum("ij,ja->ia", hf.foo, g1o.ov)
            + 1.0 * einsum("ijkl,klja->ia", hf.oooo, g2a.ooov)
            # - 1.0 * einsum("ibcd,abcd->ia", hf.ovvv, g2a.vvvv)
            - 1.0 * einsum("jkib,jkab->ia", hf.ooov, g2a.oovv)
            + 2.0 * einsum("ijkb,jakb->ia", hf.ooov, g2a.ovov)
            + 1.0 * einsum("ijbc,jabc->ia", hf.oovv, g2a.ovvv)
            - 2.0 * einsum("ibjc,jcab->ia", hf.ovov, g2a.ovvv)
        )
        w.oo = (
            + gi_oo - hf.foo
            - einsum("ik,jk->ij", g1o.oo, hf.foo)
            - einsum("ikjl,kl->ij", hf.oooo, g1o.oo)
            - einsum("ikja,ka->ij", hf.ooov, g1o.ov)
            - einsum("jkia,ka->ij", hf.ooov, g1o.ov)
            - einsum("jaib,ab->ij", hf.ovov, g1o.vv)
        )
        w.vv = gi_vv - einsum("ac,cb->ab", g1o.vv, hf.fvv)
    return evaluate(w)


class OrbitalResponseMatrix:
    def __init__(self, hf):
        self.hf = hf

    @property
    def shape(self):
        no1 = self.hf.mospaces.n_orbs(b.o)
        if self.hf.has_core_occupied_space:
            no1 += self.hf.mospaces.n_orbs(b.c)
        nv1 = self.hf.mospaces.n_orbs(b.v)
        size = no1 * nv1
        return (size, size)

    def __matmul__(self, lam):
        ret_ov = (
            + einsum("ab,ib->ia", self.hf.fvv, lam.ov)
            - einsum("ij,ja->ia", self.hf.foo, lam.ov)
            + einsum("ijab,jb->ia", self.hf.oovv, lam.ov)
            - einsum("ibja,jb->ia", self.hf.ovov, lam.ov)
        )
        if self.hf.has_core_occupied_space:
            ret_ov += (
                + einsum("iJab,Jb->ia", self.hf.ocvv, lam.cv)
                - einsum("ibJa,Jb->ia", self.hf.ovcv, lam.cv)
            )
            ret_cv = (
                + einsum("ab,Ib->Ia", self.hf.fvv, lam.cv)
                - einsum("IJ,Ja->Ia", self.hf.fcc, lam.cv)
                + einsum("IJab,Jb->Ia", self.hf.ccvv, lam.cv)
                - einsum("IbJa,Jb->Ia", self.hf.cvcv, lam.cv)
                + einsum("Ijab,jb->Ia", self.hf.covv, lam.ov)
                - einsum("Ibja,jb->Ia", self.hf.cvov, lam.ov)
            )
            ret = AmplitudeVector(cv=ret_cv, ov=ret_ov)
        else:
            ret = AmplitudeVector(ov=ret_ov)
        # TODO: generalize once other solvent methods are available
        if self.hf.environment == "pe":
            # PE contribution to the orbital Hessian
            ops = self.hf.operators
            dm = OneParticleOperator(self.hf, is_symmetric=True)
            dm.ov = lam.ov
            ret += ops.pe_induction_elec(dm).ov
        return evaluate(ret)


class OrbitalResponsePinv:
    def __init__(self, hf):
        self.hf = hf
        # Terms common to adc and cvs-adc
        fo = hf.fock(b.oo).diagonal()
        fv = hf.fock(b.vv).diagonal()
        fov = direct_sum("-i+a->ia", fo, fv)

        if hf.has_core_occupied_space:
            fc = hf.fock(b.cc).diagonal()
            fcv = direct_sum("-I+a->Ia", fc, fv)
            self.df = AmplitudeVector(cv=fcv, ov=fov)
        else:
            self.df = AmplitudeVector(ov=fov)
        self.df.evaluate()

    @property
    def shape(self):
        no1 = self.hf.mospaces.n_orbs(b.o)
        if self.hf.has_core_occupied_space:
            no1 += self.hf.mospaces.n_orbs(b.c)
        nv1 = self.hf.mospaces.n_orbs(b.v)
        size = no1 * nv1
        return (size, size)

    def __matmul__(self, invec):
        return invec / self.df


def orbital_response(hf, rhs):
    """
    Solves the orbital response equations
    for a given reference state and right-hand side
    """
    # TODO: pass solver arguments
    A = OrbitalResponseMatrix(hf)
    Pinv = OrbitalResponsePinv(hf)
    x0 = (Pinv @ rhs).evaluate()
    lam = conjugate_gradient(A, rhs=rhs, x0=x0, Pinv=Pinv,
                             explicit_symmetrisation=None,
                             callback=default_print)
    return lam.solution
