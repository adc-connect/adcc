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
import numpy as np

from adcc.LazyMp import LazyMp
from adcc.Excitation import Excitation
from adcc.timings import Timer
from adcc.functions import einsum, evaluate

from adcc.OneParticleOperator import OneParticleOperator, product_trace
from .TwoParticleDensityMatrix import TwoParticleDensityMatrix
from .orbital_response import (
    orbital_response, orbital_response_rhs, energy_weighted_density_matrix
)
from .amplitude_response import amplitude_relaxed_densities


class NuclearGradientResult:
    def __init__(self, excitation_or_mp, g1, g2):
        self.g1, self.g2 = g1, g2

    @property
    def relaxed_dipole_moment(self):
        elec_dip = -1.0 * np.array(
            [product_trace(self.g1, dip)
             for dip in self.hf.operators.electric_dipole]
        )
        return self.hf.nuclear_dipole + elec_dip


def nuclear_gradient(excitation_or_mp):
    if isinstance(excitation_or_mp, LazyMp):
        mp = excitation_or_mp
    elif isinstance(excitation_or_mp, Excitation):
        mp = excitation_or_mp.ground_state
    else:
        raise TypeError("")

    timer = Timer()
    hf = mp.reference_state
    with timer.record("amplitude_response"):
        g1a, g2a = amplitude_relaxed_densities(excitation_or_mp)

    env_lr_excited = False
    g1_exci_ao = None
    if hf.environment == "pe":
        try:
            if len(excitation_or_mp.matrix.extra_terms):
                env_lr_excited = True
                x = excitation_or_mp.excitation_vector
                d_xi = OneParticleOperator(hf, is_symmetric=False)
                d_xi.vo = x.ph.transpose()
                v_pe = hf.operators.pe_induction_elec(d_xi)
                rhs_pe = (
                    + 2.0 * einsum("ja,ij->ia", x.ph, v_pe.oo)
                    - 2.0 * einsum("ib,ab->ia", x.ph, v_pe.vv)
                )
                g1_exci = OneParticleOperator(hf, is_symmetric=True)
                g1_exci.ov = x.ph
                g1_exci_ao = sum(g1_exci.to_ao_basis()).to_ndarray()
        except AttributeError:
            pass

    with timer.record("orbital_response"):
        rhs = orbital_response_rhs(hf, g1a, g2a).evaluate()
        if env_lr_excited:
            rhs += rhs_pe
            rhs.evaluate()
        l_ov = orbital_response(hf, rhs)

    # orbital-relaxed OPDM (without reference state)
    g1o = g1a.copy()
    g1o.ov = 0.5 * l_ov
    # orbital-relaxed OPDM (including reference state)
    g1 = g1o.copy()
    g1 += hf.density

    with timer.record("energy_weighted_density_matrix"):
        w = energy_weighted_density_matrix(hf, g1o, g2a)
        if env_lr_excited:
            x = excitation_or_mp.excitation_vector
            w.oo -= einsum('jc,ic->ij', x.ph, v_pe.ov)
            w.ov -= einsum("ja,ij->ia", x.ph, v_pe.oo)
            w.vv -= einsum('kb,ka->ab', x.ph, v_pe.ov)

    # build two-particle density matrices for contraction with TEI
    with timer.record("form_tpdm"):
        delta_ij = hf.density.oo
        g2_hf = TwoParticleDensityMatrix(hf)
        g2_hf.oooo = 0.25 * (- einsum("li,jk->ijkl", delta_ij, delta_ij)
                             + einsum("ki,jl->ijkl", delta_ij, delta_ij))

        g2_oresp = TwoParticleDensityMatrix(hf)
        g2_oresp.oooo = einsum("ij,kl->kilj", delta_ij, g1o.oo)
        g2_oresp.ovov = einsum("ij,ab->iajb", delta_ij, g1o.vv)
        g2_oresp.ooov = (- einsum("kj,ia->ijka", delta_ij, g1o.ov)
                         + einsum("ki,ja->ijka", delta_ij, g1o.ov))

        # scale for contraction with integrals
        g2a.oovv *= 0.5
        g2_total = evaluate(g2_hf + g2a + g2_oresp)

    with timer.record("transform_ao"):
        g2_ao_1, g2_ao_2 = g2_total.to_ao_basis()
        g2_ao_1, g2_ao_2 = g2_ao_1.to_ndarray(), g2_ao_2.to_ndarray()
        g1_ao = sum(g1.to_ao_basis(hf)).to_ndarray()
        w_ao = sum(w.to_ao_basis(hf)).to_ndarray()

    with timer.record("contract_integral_derivatives"):
        Gradient = hf.gradient_provider.correlated_gradient(
            g1_ao, w_ao, g2_ao_1, g2_ao_2
        )
        if hf.environment == "pe":
            pe_grad = hf.gradient_provider.pe_gradient(g1_ao, g1_exci_ao)
            Gradient["PE"] = pe_grad
            Gradient["Total"] += pe_grad

    Gradient["timer"] = timer
    return Gradient
