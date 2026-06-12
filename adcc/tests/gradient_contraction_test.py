#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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
from numpy.testing import assert_allclose
import pytest

import adcc
import adcc.backends
from adcc.backends import have_backend
from adcc.gradients.TwoParticleDensityMatrix import TwoParticleDensityMatrix

from .backends.testing import cached_backend_hf


pytestmark = pytest.mark.skipif(
    not have_backend("pyscf"), reason="pyscf not found."
)


def _random_ao_inputs(nao):
    rng = np.random.default_rng(20240611)
    g1_ao = rng.standard_normal((nao, nao))
    g1_ao = 0.5 * (g1_ao + g1_ao.T)
    w_ao = rng.standard_normal((nao, nao))
    w_ao = 0.5 * (w_ao + w_ao.T)
    g2_ao_1 = rng.standard_normal((nao, nao, nao, nao))
    g2_ao_2 = rng.standard_normal((nao, nao, nao, nao))
    return g1_ao, w_ao, g2_ao_1, g2_ao_2


def test_packed_ao_pair_contraction_matches_full_pyscf_reference():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    provider = hf.gradient_provider
    inputs = _random_ao_inputs(hf.n_bas)
    _, _, g2_ao_1, g2_ao_2 = inputs

    full = provider.correlated_gradient(*inputs)
    pair_density = TwoParticleDensityMatrix.ao_pair_density_from_dense(
        g2_ao_1, g2_ao_2
    )
    packed_tei = provider._contract_tei_with_packed_density(
        pair_density, shell_chunk_size=2
    )

    assert_allclose(packed_tei, full.two_electron, atol=1e-10)


def test_direct_mp2_pair_density_matches_dense_ao_transform():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))
    gradient = adcc.nuclear_gradient(mp, tei_contraction="full_ao")

    g2_ao_1, g2_ao_2 = gradient.g2.to_ao_basis()
    dense_pair_density = TwoParticleDensityMatrix.ao_pair_density_from_dense(
        g2_ao_1.to_ndarray(), g2_ao_2.to_ndarray()
    )
    direct_pair_density = gradient.g2.to_ao_pair_density(
        gradient.reference_state, pair_chunk_size=5
    )

    assert_allclose(direct_pair_density, dense_pair_density, atol=1e-10)


def test_direct_mp2_gradient_matches_full_ao_fallback():
    hf = cached_backend_hf("pyscf", "h2o_sto3g", conv_tol=1e-11)
    mp = adcc.LazyMp(adcc.ReferenceState(hf))

    full = adcc.nuclear_gradient(mp, tei_contraction="full_ao")
    direct = adcc.nuclear_gradient(
        mp, tei_contraction="direct", tei_shell_chunk_size=2,
        tei_pair_chunk_size=5
    )

    assert_allclose(direct.total, full.total, atol=1e-10)
    assert_allclose(direct.components.two_electron,
                    full.components.two_electron, atol=1e-10)
