#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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

import numpy as np

import adcc
from .eri_build_helper import _eri_phys_asymm_spin_allowed_prefactors


def eri_asymm_construction_test(scfres):
    hfdata = adcc.backends.import_scf_results(scfres)
    refstate = adcc.ReferenceState(hfdata)

    # build the full ERI tensor
    eri_chem = np.empty((hfdata.n_orbs, hfdata.n_orbs,
                         hfdata.n_orbs, hfdata.n_orbs))
    sfull = slice(hfdata.n_orbs)
    hfdata.fill_eri_ffff((sfull, sfull, sfull, sfull), eri_chem)
    eri_phys = eri_chem.transpose(0, 2, 1, 3)
    # full anti-symmetrized ERI tensor
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
        "o1o1o1v1",
        "o1o1v1v1",
        "o1v1o1v1",
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

    n_elec = n_alpha + n_beta
    n_virt_a = n_orbs_alpha - n_alpha
    lookuptable_prelim = {
        "o": {
            "a": slice(0, n_alpha, 1), "b": slice(n_alpha, n_elec, 1)
        },
        "v": {
            "a": slice(0, n_virt_a, 1), "b": slice(n_virt_a, n_orbs - n_elec, 1)
        }
    }
    # loop over all spaces and compare imported
    # tensor to full tensor
    for s in space_names:
        imported_asymm = refstate.eri(s).to_ndarray()
        s_clean = s.replace("1", "")
        for allowed_spin in _eri_phys_asymm_spin_allowed_prefactors:
            sl = [lookuptable[x] for x in list(s_clean)]
            sl = [sl[x][y] for x, y in
                  enumerate(list(allowed_spin.transposition))]
            sl2 = [lookuptable_prelim[x] for x in list(s_clean)]
            sl2 = [sl2[x][y] for x, y in
                   enumerate(list(allowed_spin.transposition))]
            sl = tuple(sl)
            sl2 = tuple(sl2)
            np.testing.assert_almost_equal(
                eri_asymm[sl], imported_asymm[sl2],
                err_msg="""ERIs wrong in space {} """
                        """and spin block {}""".format(s, allowed_spin)
            )
