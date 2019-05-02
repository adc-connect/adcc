#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------


import pytest
import adcc
import unittest
import adcc.backends
import numpy as np
from numpy.testing import assert_array_equal
import os
import tempfile

from adcc.tmp_build_reference_state import tmp_build_reference_state

from .eri_build_helper import _eri_phys_asymm_spin_allowed_prefactors

try:
    from mpi4py import MPI
    import veloxchem as vlx
    _vlx = True
except ImportError:
    _vlx = False


@pytest.mark.skipif(not _vlx, reason="Veloxchem not found.")
class TestVeloxchem(unittest.TestCase):
    def run_hf(self, xyz, basis, charge=0, multiplicity=1):
        basis_dir = os.path.abspath(os.path.join(vlx.__path__[-1],
                                                 "..", "..", "..", "basis"))
        with tempfile.TemporaryDirectory() as tmpdir:
            infile = os.path.join(tmpdir, "vlx.in")
            outfile = os.path.join(tmpdir, "vlx.out")
            with open(infile, "w") as fp:
                lines = ["@jobs", "task: hf", "@end", ""]
                lines += ["@method settings",
                          "basis: {}".format(basis),
                          "basis path: {}".format(basis_dir), "@end", ""]
                lines += ["@molecule",
                          "charge: {}".format(charge),
                          "multiplicity: {}".format(multiplicity),
                          "units: bohr",
                          "xyz:\n{}".format("\n".join(xyz.split(";"))),
                          "@end"]
                fp.write("\n".join(lines))
            task = vlx.MpiTask([infile, outfile], MPI.COMM_WORLD)

            scfdrv = vlx.ScfRestrictedDriver()
            scfdrv.conv_thresh = 1e-14
            scfdrv.compute_task(task)
            scfdrv.task = task
        return scfdrv

    def base_test(self, scfdrv):
        hfdata = adcc.backends.import_scf_results(scfdrv)
        assert hfdata.backend == "veloxchem"

        n_orbs_alpha = hfdata.n_orbs_alpha
        if hfdata.restricted:
            assert hfdata.spin_multiplicity != 0
            assert hfdata.n_alpha >= hfdata.n_beta

            # Check SCF type fits
            assert isinstance(scfdrv, (vlx. ScfRestrictedDriver))

            assert hfdata.n_orbs_alpha == hfdata.n_orbs_beta
            assert_array_equal(hfdata.orben_f[:n_orbs_alpha],
                               hfdata.orben_f[n_orbs_alpha:])

            n_alpha = int(scfdrv.task.molecule.number_of_electrons() / 2)
            n_beta = n_alpha
            mo_energy = (scfdrv.mol_orbs.ea_to_numpy(),
                         scfdrv.mol_orbs.ea_to_numpy())
        else:
            raise NotImplementedError()

        # Check n_alpha and n_beta
        assert hfdata.n_alpha == n_alpha
        assert hfdata.n_beta == n_beta
        assert_array_equal(mo_energy[0], hfdata.orben_f[:n_orbs_alpha])
        assert_array_equal(mo_energy[1], hfdata.orben_f[n_orbs_alpha:])

        # Check the lowest n_alpha / n_beta orbitals are occupied
        assert_array_equal(np.sort(mo_energy[0]), mo_energy[0])
        assert_array_equal(np.sort(mo_energy[1]), mo_energy[1])

        # test symmetry of the ERI tensor
        ii, jj, kk, ll = 0, 1, 2, 3
        allowed_permutations = [
            (kk, ll, ii, jj),
            (jj, ii, ll, kk),
            (ll, kk, jj, ii),
            (jj, ii, kk, ll),
            (jj, ii, ll, kk),
        ]
        eri = np.empty((hfdata.n_orbs, hfdata.n_orbs,
                        hfdata.n_orbs, hfdata.n_orbs))
        sfull = slice(hfdata.n_orbs)
        hfdata.fill_eri_ffff((sfull, sfull, sfull, sfull), eri)
        for perm in allowed_permutations:
            eri_perm = np.transpose(eri, perm)
            np.testing.assert_almost_equal(eri_perm, eri)

        # TODO Many more tests
        #      Compare against reference data

        # TODO checks for values of
        # orben_f
        # fock_ff
        # orbcoeff_fb
        # Perhaps by transforming the fock matrix of the scfdrv object or so

    def eri_asymm_construction_test(self, scfres):
        adcc.memory_pool.initialise(max_memory=1024 * 1024 * 1024,
                                    tensor_block_size=16, allocator="standard")
        hfdata = adcc.backends.import_scf_results(scfres)
        assert hfdata.backend == "veloxchem"

        refstate = tmp_build_reference_state(
            hfdata
        )

        eri_chem = np.empty((hfdata.n_orbs, hfdata.n_orbs,
                             hfdata.n_orbs, hfdata.n_orbs))
        sfull = slice(hfdata.n_orbs)
        hfdata.fill_eri_ffff((sfull, sfull, sfull, sfull), eri_chem)
        eri_phys = eri_chem.transpose(0, 2, 1, 3)
        eri_asymm = eri_phys - eri_phys.transpose(1, 0, 2, 3)

        n_orbs = hfdata.n_orbs
        n_alpha = hfdata.n_alpha
        n_beta = hfdata.n_beta
        n_orbs_alpha = hfdata.n_orbs_alpha

        aro = slice(0, n_alpha, 1)
        bro = slice(n_orbs_alpha, n_orbs_alpha + n_beta, 1)
        arv = slice(n_alpha, n_orbs_alpha, 1)
        brv = slice(n_orbs_alpha + n_beta, n_orbs, 1)

        space_names = [
            "o1o1o1o1",
            # "o1o1o1v1",
            # "o1o1v1v1",
            # "o1v1o1v1",
            "o1v1v1v1",
            "v1v1v1v1",
        ]
        lookuptable = {
            "o": {
                "a": aro,
                "b": bro,
            },
            "v": {
                "a": arv,
                "b": brv,
            }
        }

        lookuptable_prelim = {
            "o": {
                "a": slice(0, n_alpha, 1),
                "b": slice(n_alpha, n_alpha + n_beta, 1),
            },
            "v": {
                "a": slice(0, n_orbs_alpha - n_alpha, 1),
                "b": slice(n_orbs_alpha - n_alpha,
                           n_orbs - (n_alpha + n_beta), 1),
            }
        }

        for s in space_names:
            print(s)
            imported_asymm = refstate.eri(s).to_ndarray()
            s_clean = s.replace("1", "")
            for allowed_spin in _eri_phys_asymm_spin_allowed_prefactors:
                sl = [lookuptable[x] for x in list(s_clean)]
                sl = [sl[x][y] for x, y in enumerate(list(allowed_spin.transposition))]
                sl2 = [lookuptable_prelim[x] for x in list(s_clean)]
                sl2 = [sl2[x][y] for x, y in enumerate(list(allowed_spin.transposition))]
                sl = tuple(sl)
                sl2 = tuple(sl2)
                np.testing.assert_almost_equal(eri_asymm[sl],
                                               imported_asymm[sl2],
                                               err_msg="""ERIs wrong in """
                                               """space {} and spin """
                                               """block {}""".format(s, allowed_spin))

    def test_water_sto3g_rhf(self):
        scfdrv = self.run_hf(xyz='O 0 0 0;'
                             'H 0 0 1.795239827225189;'
                             'H 1.693194615993441 0 -0.599043184453037',
                             basis="sto-3g")
        self.eri_asymm_construction_test(scfdrv)
        # self.base_test(scfdrv)
