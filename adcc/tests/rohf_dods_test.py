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
import pytest
from numpy.testing import assert_allclose

import adcc
from adcc.exceptions import InputError
from adcc.LazyMp import LazyMp
from adcc.LazyRoDodsMp import LazyRoDodsMp

pyscf = pytest.importorskip("pyscf")

# For a UHF reference F = f and F_ov = 0, so the ROHF DODS variant
# has to reproduce the standard UHF results.
methods = ["adc0", "adc1", "adc2"]
# rohf_dods is not available with CVS
cvs_methods = [
    pytest.param(f"cvs-{m}", marks=pytest.mark.xfail(raises=InputError, strict=True))
    for m in methods
]


@pytest.fixture(scope="module")
def scfres():
    mol = pyscf.gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = pyscf.scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged
    return mf


# OH sto-3g excitation energies on the custom ROHF DODS reference,
# dumped with the rohf_dods branch (ROHF DODS terms in adc_pp/matrix.py)
reference_energies = {
    "adc0": [0.737825758020, 0.859775274534, 1.064157161490, 1.096280411072, 1.143724856191],
    "adc1": [0.0, 0.219491510508, 0.431641817312, 0.496429944602, 0.561056495648],
    "adc2": [-3.988112361e-06, 0.224485885747, 0.431632956371, 0.492261085482, 0.541331867848],
}


@pytest.fixture(scope="module")
def scfres_dods():
    rohf_dods = pytest.importorskip("pyscf.scf.rohf_dods")
    mol = pyscf.gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = rohf_dods.ROHFDoDS(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    assert mf.converged
    return mf.to_uhf_dods()


@pytest.fixture(scope="module")
def refstates(scfres):
    return {
        rohf_dods: adcc.ReferenceState(scfres, rohf_dods=rohf_dods)
        for rohf_dods in (False, True)
    }


class TestRohfDods:
    @pytest.mark.parametrize("method", methods)
    def test_matvec(self, refstates, method):
        uhf = adcc.AdcMatrix(method, refstates[False])
        dods = adcc.AdcMatrix(method, refstates[True])
        assert not uhf.rohf_dods and type(uhf.ground_state) is LazyMp
        assert dods.rohf_dods and type(dods.ground_state) is LazyRoDodsMp
        vec = adcc.guess_zero(uhf)
        for block in vec.blocks:
            vec[block].set_random()
        ref = uhf @ vec
        res = dods @ vec
        for block in ref.blocks:
            assert_allclose(res[block].to_ndarray(), ref[block].to_ndarray(), atol=1e-12)

    @pytest.mark.parametrize("method", methods + cvs_methods)
    def test_excitation_energies(self, scfres, method):
        kwargs = dict(method=method, n_states=5, conv_tol=1e-10, output=None)
        if method.startswith("cvs"):
            # only 3 CVS guesses with one core orbital in sto-3g
            kwargs.update(core_orbitals=1, n_states=2)
        ref = adcc.run_adc(scfres, **kwargs)
        res = adcc.run_adc(scfres, rohf_dods=True, **kwargs)
        assert not ref.rohf_dods and type(ref.ground_state) is LazyMp
        assert res.rohf_dods and type(res.ground_state) is LazyRoDodsMp
        assert_allclose(res.ground_state.energy(2), ref.ground_state.energy(2), atol=1e-12)
        assert res.converged
        assert_allclose(res.excitation_energy, ref.excitation_energy, atol=1e-8)

    @pytest.mark.parametrize("method", methods)
    def test_reference_energies(self, scfres_dods, method):
        res = adcc.run_adc(
            scfres_dods, method=method, n_states=5, conv_tol=1e-10, output=None, rohf_dods=True
        )
        assert res.rohf_dods and type(res.ground_state) is LazyRoDodsMp
        assert res.converged
        assert_allclose(res.excitation_energy, reference_energies[method], atol=1e-8)
