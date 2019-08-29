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
import sys

sys.path.insert(0, "../examples/water/")
sys.path.insert(0, "..")

import adcc
import numpy as np

import import_data

refstate = adcc.ReferenceState(import_data.import_data())
adc1 = adcc.AdcMatrix("adc1", refstate)

print("#\n#-- ADC(1)\n#")
n_states = 10
state1 = adcc.adc1(refstate, n_states=n_states)
dense1 = adc1.to_dense_matrix()
np.testing.assert_almost_equal(dense1, dense1.T)
spectrum1 = np.linalg.eigvalsh(dense1)

n_decimals = 10
atol = 1e-6
ref = np.round(state1.excitation_energies, n_decimals)
test = np.unique(np.round(spectrum1, n_decimals))[:n_states]
np.testing.assert_allclose(ref, test, atol=atol)
del state1
del dense1
del spectrum1

print()
print()
print()

print("#\n#-- ADC(2)\n#")
n_states = 3  # TODO Test for a few more ...
adc2 = adcc.AdcMatrix("adc2", refstate)
state2 = adcc.adc2(refstate, n_states=n_states)
dense2 = adc2.to_dense_matrix()
np.testing.assert_almost_equal(dense2, dense2.T)
spectrum2 = np.linalg.eigvalsh(dense2)

n_decimals = 10
atol = 1e-6
ref = np.round(state2.excitation_energies, n_decimals)
test = np.unique(np.round(spectrum2, n_decimals))[:n_states]
np.testing.assert_allclose(ref, test, atol=atol)
del state2
del dense2
del spectrum2

print()
print()
print()

print("#\n#-- ADC(2)-x\n#")
n_states = 3  # TODO Test for a few more ...
adc2x = adcc.AdcMatrix("adc2x", refstate)
state2x = adcc.adc2x(refstate, n_states=n_states)
dense2x = adc2x.to_dense_matrix()
np.testing.assert_almost_equal(dense2x, dense2x.T)
spectrum2x = np.linalg.eigvalsh(dense2x)

n_decimals = 10
atol = 1e-6
ref = np.round(state2x.excitation_energies, n_decimals)
test = np.unique(np.round(spectrum2x, n_decimals))[:n_states]
np.testing.assert_allclose(ref, test, atol=atol)
del state2x
del dense2x
del spectrum2x

print()
print()
print()

print("#\n#-- ADC(3)\n#")
n_states = 3  # TODO Test for a few more ...
adc3 = adcc.AdcMatrix("adc3", refstate)
state3 = adcc.adc3(refstate, n_states=n_states)
dense3 = adc3.to_dense_matrix()
np.testing.assert_almost_equal(dense3, dense3.T)
spectrum3 = np.linalg.eigvalsh(dense3)

n_decimals = 10
atol = 1e-6
ref = np.round(state3.excitation_energies, n_decimals)
test = np.unique(np.round(spectrum3, n_decimals))[:n_states]
np.testing.assert_allclose(ref, test, atol=atol)
del state3
del dense3
del spectrum3
