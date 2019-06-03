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

import adcc
import libadcc

import numpy as np


def operator_import_test(scfres, ao_dict):
    refstate = adcc.ReferenceState(scfres)
    occa = refstate.orbital_coefficients_alpha("o1b").to_ndarray()
    occb = refstate.orbital_coefficients_beta("o1b").to_ndarray()
    virta = refstate.orbital_coefficients_alpha("v1b").to_ndarray()
    virtb = refstate.orbital_coefficients_beta("v1b").to_ndarray()

    for k in ao_dict:
        dip_oo = np.einsum('ib,ba,ja->ij', occa, ao_dict[k], occa)
        dip_oo += np.einsum('ib,ba,ja->ij', occb, ao_dict[k], occb)

        dip_ov = np.einsum('ib,ba,ja->ij', occa, ao_dict[k], virta)
        dip_ov += np.einsum('ib,ba,ja->ij', occb, ao_dict[k], virtb)

        dip_vv = np.einsum('ib,ba,ja->ij', virta, ao_dict[k], virta)
        dip_vv += np.einsum('ib,ba,ja->ij', virtb, ao_dict[k], virtb)

        dip_mock = {"o1o1": dip_oo, "o1v1": dip_ov, "v1v1": dip_vv}

        dip_imported = refstate.operator_integrals.electric_dipole(k)
        for b in dip_imported.blocks:
            symmetry_ref = libadcc.make_symmetry_operator(
                refstate.mospaces, b, True, k
            )
            ref = symmetry_ref.describe()
            imported_sym = dip_imported[b].describe_symmetry()
            assert ref == imported_sym
            np.testing.assert_allclose(
                dip_mock[b], dip_imported[b].to_ndarray(),
                atol=refstate.conv_tol
            )
