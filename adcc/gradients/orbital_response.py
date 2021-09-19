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
from adcc.functions import direct_sum, einsum, evaluate, zeros_like

from adcc.Tensor import Tensor
from adcc.AmplitudeVector import AmplitudeVector
import numpy as np

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
            + 2.0 * einsum("kiJa,Jk->ia", hf.oocv, g1a.co)
            - 2.0 * einsum("iJka,Jk->ia", hf.ocov, g1a.co)
            + 2.0*einsum ("iJKb,JaKb->ia", hf.occv, g2a.cvcv) # TODO: new var.
        )

        ret_cv = -1.0 * (
            + 2.0 * einsum("JIKa,JK->Ia", hf.cccv, g1a.cc)
            + 2.0 * einsum("Icab,bc->Ia", hf.cvvv, g1a.vv)
            + 2.0 * einsum("kIJa,Jk->Ia", hf.occv, g1a.co)
            + 2.0 * einsum("JIka,Jk->Ia", hf.ccov, g1a.co)
            + 2.0 * einsum("IJKb,JaKb->Ia", hf.cccv, g2a.cvcv) # TODO: new var.
            + 2.0 * einsum("Jcab,IbJc->Ia", hf.cvvv, g2a.cvcv) # TODO: new var.
        )

        #print("\nRHS OV:\n", ret_cv.evaluate())

        ret = AmplitudeVector(ph=ret_cv, pphh=ret_ov)
        
    else:
        # equal to the ov block of the energy-weighted density
        # matrix when lambda_ov multipliers are zero
        w_ov = 0.5 * (
            + 1.0 * einsum("ijkl,klja->ia", hf.oooo, g2a.ooov)
            # - 1.0 * einsum("ibcd,abcd->ia", hf.ovvv, g2a.vvvv)
            - 1.0 * einsum("jkib,jkab->ia", hf.ooov, g2a.oovv)
            + 2.0 * einsum("ijkb,jakb->ia", hf.ooov, g2a.ovov)
            + 1.0 * einsum("ijbc,jabc->ia", hf.oovv, g2a.ovvv)
            - 2.0 * einsum("ibjc,jcab->ia", hf.ovov, g2a.ovvv)
        )
        w_ov = w_ov.evaluate()
    
        ret = -1.0 * (
            2.0 * w_ov
            # - 1.0 * einsum("klja,ijkl->ia", hf.ooov, g2a.oooo)
            + 1.0 * einsum("abcd,ibcd->ia", hf.vvvv, g2a.ovvv)
            - 2.0 * einsum("jakb,ijkb->ia", hf.ovov, g2a.ooov)
            + 1.0 * einsum("jkab,jkib->ia", hf.oovv, g2a.ooov)
            + 2.0 * einsum("jcab,ibjc->ia", hf.ovvv, g2a.ovov)
            - 1.0 * einsum("jabc,ijbc->ia", hf.ovvv, g2a.oovv)
            + 2.0 * einsum("jika,jk->ia", hf.ooov, g1a.oo)
            + 2.0 * einsum("icab,bc->ia", hf.ovvv, g1a.vv)
        )
    return ret


def energy_weighted_density_matrix(hf, g1o, g2a):
    if hf.has_core_occupied_space:
        # CVS-ADC0; TODO: add 2PDM contributions
        w = OneParticleOperator(hf)
        w.cc = ( 
            - hf.fcc
            - einsum("ik,jk->ij", g1o.cc, hf.fcc)
            - einsum("ab,IaJb->IJ", g1o.vv, hf.cvcv)
            - einsum("KL,IKJL->IJ", g1o.cc, hf.cccc)
            + einsum("ka,kJIa->IJ", g1o.ov, hf.occv)
            + einsum("ka,kIJa->IJ", g1o.ov, hf.occv)
            + einsum("Ka,KJIa->IJ", g1o.cv, hf.cccv)
            + einsum("Ka,KIJa->IJ", g1o.cv, hf.cccv)
            - einsum("Lk,kILJ->IJ", g1o.co, hf.occc)
            - einsum("Lk,kJLI->IJ", g1o.co, hf.occc)
            - einsum("JbKc,IbKc->IJ", g2a.cvcv, hf.cvcv)
        )
        w.oo = (
            - hf.foo
            - einsum("KL,iKjL->ij", g1o.cc, hf.ococ)
            - einsum("ab,iajb->ij", g1o.vv, hf.ovov)
            - einsum("ka,ikja->ij", g1o.ov, hf.ooov)
            - einsum("ka,jkia->ij", g1o.ov, hf.ooov)
            - einsum("Ka,iKja->ij", g1o.cv, hf.ocov)
            - einsum("Ka,jKia->ij", g1o.cv, hf.ocov)
            + einsum("Lk,kijL->ij", g1o.co, hf.oooc)
            + einsum("Lk,kjiL->ij", g1o.co, hf.oooc)
        )
        w.vv = (
            - einsum("ac,cb->ab", g1o.vv, hf.fvv)
            - einsum('JbKc,JaKc->ab', g2a.cvcv, hf.cvcv)
        )
        w.co = (
              einsum("KL,iKLJ->Ji", g1o.cc, hf.occc)
            - einsum("ab,iaJb->Ji", g1o.vv, hf.ovcv)
            - einsum("Ka,iKJa->Ji", g1o.cv, hf.occv)
            - einsum("Ka,JKia->Ji", g1o.cv, hf.ccov)
            - einsum("ka,ikJa->Ji", g1o.ov, hf.oocv)
            - einsum("ka,Jkia->Ji", g1o.ov, hf.coov)
            - einsum("Lk,kiLJ->Ji", g1o.co, hf.oocc)
            + einsum("Lk,iLkJ->Ji", g1o.co, hf.ococ)
            - einsum("Ik,jk->Ij", g1o.co, hf.foo)
            - einsum("JbKc,ibKc->Ji", g2a.cvcv, hf.ovcv)
        )
        w.ov = (
            - einsum("ij,ja->ia", hf.foo, g1o.ov)
            + einsum("JaKc,iJKc->ia", g2a.cvcv, hf.occv)
        )
        w.cv = (
            - einsum("IJ,Ja->Ia", hf.fcc, g1o.cv)
            + einsum("JaKc,IJKc->Ia", g2a.cvcv, hf.cccv)
        )
        print("\nOMEGA cv:\n", w.cv.evaluate())
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
        #if hf.has_core_occupied_space:
        #    raise NotImplementedError("OrbitalResponseMatrix not implemented "
        #                              "for CVS reference state.")
        self.hf = hf

    @property
    def shape(self):
        no1 = self.hf.mospaces.n_orbs(b.o)
        if self.hf.has_core_occupied_space:
            no1 += self.hf.mospaces.n_orbs(b.c)
        nv1 = self.hf.mospaces.n_orbs(b.v)
        size = no1 * nv1
        return (size, size)

    def __matmul__(self, l_ov):
        if self.hf.has_core_occupied_space:
            # This is the OV block (hijacked from AmplitudeVector)
            # TODO: find a different solution (new class?)
            ret_ov = (
                + einsum("ab,ib->ia", self.hf.fvv, l_ov['pphh'])
                - einsum("ij,ja->ia", self.hf.foo, l_ov['pphh'])
                + einsum("ijab,jb->ia", self.hf.oovv, l_ov['pphh'])
                - einsum("ibja,jb->ia", self.hf.ovov, l_ov['pphh'])
                + einsum("iJab,Jb->ia", self.hf.ocvv, l_ov['ph'])
                - einsum("ibJa,Jb->ia", self.hf.ovcv, l_ov['ph'])
            )
            # This is the CV block (hijacked from AmplitudeVector)
            # TODO: find a different solution (new class?)
            ret_cv = (
                + einsum("ab,Ib->Ia", self.hf.fvv, l_ov['ph'])
                - einsum("IJ,Ja->Ia", self.hf.fcc, l_ov['ph'])
                + einsum("IJab,Jb->Ia", self.hf.ccvv, l_ov['ph'])
                - einsum("IbJa,Jb->Ia", self.hf.cvcv, l_ov['ph'])
                + einsum("Ijab,jb->Ia", self.hf.covv, l_ov['pphh'])
                - einsum("Ibja,jb->Ia", self.hf.cvov, l_ov['pphh'])
            )
            ret = AmplitudeVector(ph=ret_cv, pphh=ret_ov)
        else:
            ret = (
                + einsum("ab,ib->ia", self.hf.fvv, l_ov)
                - einsum("ij,ja->ia", self.hf.foo, l_ov)
                + einsum("ijab,jb->ia", self.hf.oovv, l_ov)
                - einsum("ibja,jb->ia", self.hf.ovov, l_ov)
            )
        # TODO: generalize once other solvent methods are available
        if self.hf.environment == "pe":
            # PE contribution to the orbital Hessian
            ops = self.hf.operators
            dm = OneParticleOperator(self.hf, is_symmetric=True)
            dm.ov = l_ov
            ret += ops.pe_induction_elec(dm).ov
        return evaluate(ret)


class OrbitalResponsePinv:
    def __init__(self, hf):
        self.hf = hf
        if hf.has_core_occupied_space:
            fc = hf.fock(b.cc).diagonal()
            fo = hf.fock(b.oo).diagonal()
            fv = hf.fock(b.vv).diagonal()
            fov = direct_sum("-i+a->ia", fo, fv).evaluate()
            fcv = direct_sum("-I+a->Ia", fc, fv).evaluate()

            # Highjacking AmplitudeVector to store both cv and ov blocks.
            # 'ph' stores the CV block, 'pphh' stores the OV block.
            self.df = AmplitudeVector(ph=fcv, pphh=fov)
        else: 
            fo = hf.fock(b.oo).diagonal()
            fv = hf.fock(b.vv).diagonal()
            self.df = direct_sum("-i+a->ia", fo, fv).evaluate()

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
    l_ov = conjugate_gradient(A, rhs=rhs, x0=x0, Pinv=Pinv,
                              explicit_symmetrisation=None,
                              callback=default_print)
    #print(rhs.space)
    #print(rhs.subspaces)
    ##print(dir(l_ov.solution))
    print("Solution CV:\n", 0.5*l_ov.solution['ph'])
    print("\nSolution OV:\n",0.5*l_ov.solution['pphh'])
    return l_ov.solution
