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

from adcc.solver.conjugate_gradient import conjugate_gradient, default_print


def orbital_response_rhs(hf, g1a, g2a):
    """
    Build the right-hand side for solving the orbital
    response given amplitude-relaxed density matrices (method-specific)
    """
    # TODO: only add non-zero blocks to equations!

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
        if hf.has_core_occupied_space:
            raise NotImplementedError("OrbitalResponseMatrix not implemented "
                                      "for CVS reference state.")
        self.hf = hf

    @property
    def shape(self):
        no1 = self.hf.n_orbs(b.o)
        nv1 = self.hf.n_orbs(b.v)
        size = no1 * nv1
        return (size, size)

    def __matmul__(self, l_ov):
        ret = (
            + einsum("ab,ib->ia", self.hf.fvv, l_ov)
            - einsum("ij,ja->ia", self.hf.foo, l_ov)
            + einsum("ijab,jb->ia", self.hf.oovv, l_ov)
            - einsum("ibja,jb->ia", self.hf.ovov, l_ov)
        )
        # TODO: generalize once other solvent methods are available
        if "pe_induction_elec" in self.hf.operators.density_dependent_operators:
            # PE contribution to the orbital Hessian
            ops = self.hf.operators
            dm = OneParticleOperator(self.hf, is_symmetric=True)
            dm.ov = l_ov
            ret += ops.density_dependent_operators["pe_induction_elec"](dm).ov
        return evaluate(ret)


class OrbitalResponsePinv:
    def __init__(self, hf):
        if hf.has_core_occupied_space:
            raise NotImplementedError("OrbitalResponsePinv not implemented "
                                      "for CVS reference state.")
        fo = hf.fock(b.oo).diagonal()
        fv = hf.fock(b.vv).diagonal()
        self.df = direct_sum("-i+a->ia", fo, fv).evaluate()

    @property
    def shape(self):
        no1 = self.hf.n_orbs(b.o)
        nv1 = self.hf.n_orbs(b.v)
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
    return l_ov.solution
