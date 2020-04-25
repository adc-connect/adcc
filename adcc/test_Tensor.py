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
import unittest
import numpy as np

from .misc import expand_test_templates
from numpy.testing import assert_allclose

from adcc import contract, direct_sum, empty_like, nosym_like
from adcc.testdata.cache import cache


@expand_test_templates(["h2o_sto3g", "cn_sto3g"])
class TestTensor(unittest.TestCase):
    def template_nontrivial_addition(self, case):
        refstate = cache.refstate[case]
        mtcs = [empty_like(refstate.fock("o1v1")).set_random(),
                empty_like(refstate.fock("v1o1")).set_random(),
                empty_like(refstate.fock("o1v1")).set_random()]
        mnps = [m.to_ndarray() for m in mtcs]

        res = -3 * (2 * mtcs[0] + mtcs[1].T) * mtcs[2]
        ref = -3 * (2 * mnps[0] + mnps[1].T) * mnps[2]

        assert res.needs_evaluation
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

    def test_nontrivial_symmetrisation(self):
        refstate = cache.refstate["cn_sto3g"]
        mtcs = [nosym_like(refstate.eri("o1o1v1v1")).set_random(),
                nosym_like(refstate.eri("o1v1o1v1")).set_random(),
                nosym_like(refstate.eri("o1o1v1v1")).set_random()]
        mnps = [m.to_ndarray() for m in mtcs]

        res = (mtcs[0] / mtcs[1].transpose((0, 2, 3, 1)))
        res -= 2 * mtcs[2].antisymmetrise((0, 1))
        res = res.symmetrise((0, 1), (2, 3))

        ref = (mnps[0] / mnps[1].transpose((0, 2, 3, 1)))
        ref -= (mnps[2] - mnps[2].transpose((1, 0, 2, 3)))
        ref = 0.5 * (ref + ref.transpose((1, 0, 3, 2)))

        assert res.needs_evaluation
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

    def template_nontrivial_contraction(self, case):
        refstate = cache.refstate[case]
        f_oo = empty_like(refstate.fock("o1o1")).set_random()
        f_vv = empty_like(refstate.fock("v1v1")).set_random()
        i1 = empty_like(refstate.fock("v1v1")).set_random()
        i2 = empty_like(refstate.fock("o1o1")).set_random()
        oovv = empty_like(refstate.eri("o1o1v1v1")).set_random()
        ovov = empty_like(refstate.eri("o1v1o1v1")).set_random()
        t2 = empty_like(refstate.eri("o1o1v1v1")).set_random()
        u1 = empty_like(refstate.fock("o1v1")).set_random()
        nf_oo = f_oo.to_ndarray()
        nf_vv = f_vv.to_ndarray()
        ni1 = i1.to_ndarray()
        ni2 = i2.to_ndarray()
        noovv = oovv.to_ndarray()
        novov = ovov.to_ndarray()
        nt2 = t2.to_ndarray()
        nu1 = u1.to_ndarray()

        # (Slightly) modified ADC(2) singles block equation
        res = (
            + contract("ib,ab->ia", u1, f_vv + i1)
            - contract("ij,ja->ia", f_oo - i2, u1)
            - contract("jaib,jb->ia", ovov, u1)
            - 0.5 * contract("ijab,jb->ia", t2, contract("jkbc,kc->jb", oovv, u1))
            - 0.5 * contract("ijab,jb->ia", oovv, contract("jkbc,kc->jb", t2, u1))
        )
        ref = (
            + np.einsum("ib,ab->ia", nu1, nf_vv + ni1)
            - np.einsum("ij,ja->ia", nf_oo - ni2, nu1)
            - np.einsum("jaib,jb->ia", novov, nu1)
            - 0.5 * np.einsum("ijab,jkbc,kc->ia", nt2, noovv, nu1)
            - 0.5 * np.einsum("ijab,jkbc,kc->ia", noovv, nt2, nu1)
        )

        assert res.needs_evaluation
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)

    def test_nontrivial_trace(self):
        import libadcc

        refstate = cache.refstate["cn_sto3g"]
        oovv = empty_like(refstate.eri("o1o1v1v1")).set_random()
        ovov = empty_like(refstate.eri("o1v1o1v1")).set_random()
        noovv = oovv.to_ndarray()
        novov = ovov.to_ndarray()

        res = libadcc.trace("iaai", contract("ijab,kcja->icbk", oovv, ovov))
        ref = np.einsum("ijab,ibja->", noovv, novov)
        assert_allclose(res, ref, rtol=1e-10, atol=1e-14)

    def test_nontrivial_direct_sum(self):
        refstate = cache.refstate["cn_sto3g"]
        oeo = nosym_like(refstate.orbital_energies("o1")).set_random()
        oovv = empty_like(refstate.eri("o1o1v1v1")).set_random()
        oev = nosym_like(refstate.orbital_energies("v1")).set_random()

        interm = np.einsum("ijac,ijcb->ab", oovv.to_ndarray(), oovv.to_ndarray())
        ref = (interm[:, :, None, None]
               - oeo.to_ndarray()[None, None, :, None]
               - oev.to_ndarray()[None, None, None, :])
        ref = ref.transpose((2, 0, 3, 1))

        res = direct_sum("ab-i-c->iacb", contract("ijac,ijcb->ab", oovv, oovv),
                         oeo, oev)
        assert_allclose(res.to_ndarray(), ref, rtol=1e-10, atol=1e-14)
