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
import adcc
import numpy as np

from pyscf import gto, scf

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='sto-3g',
    unit="Bohr"
)

# Compute dipole moment integrals
with mol.with_common_orig((0, 0, 0)):
    dip_ao = mol.intor_symmetric('int1e_r', comp=3)

# Run RHF SCF
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-14
scfres.grad_conv_tol = 1e-10
scfres.kernel()

# Initialise ADC memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc1 calculation:
state = adcc.adc1(scfres, n_singlets=5)[0]

# Attach state densities
state = adcc.attach_state_densities(state, state_diffdm=False)

# Print results in a nice way
print()
print("st   ex.ene. (au)         f  transition dipole moment (au)")
for i, ampl in enumerate(state.eigenvectors):
    tdm = state.ground_to_excited_tdms[i]
    tdm_a, tdm_b = tdm.transform_to_ao_basis(state.reference_state)
    tdm_ao = (tdm_a + tdm_b).to_ndarray()

    # Compute transition dipole moment
    tdip = np.einsum('xij,ij->x', dip_ao, tdm_ao)
    osc = 2. / 3. * np.linalg.norm(tdip)**2 * np.abs(state.eigenvalues[i])

    fmt = "{0:2d}   {1:12.8g} {2:9.3g}  [{3:9.3g}, {4:9.3g}, {5:9.3g}]"
    print(fmt.format(i, state.eigenvalues[i], osc, *tdip))
