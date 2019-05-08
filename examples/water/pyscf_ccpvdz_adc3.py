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

from pyscf import gto, scf

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvdz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()

#
# Some more advanced memory tampering options
#
# Initialise ADC memory (512 MiB)
# Use a tensor block size parameter of 16 and
# a specific allocator (in this case std::allocator)
adcc.memory_pool.initialise(max_memory=512 * 1024 * 1024,
                            tensor_block_size=16, allocator="standard")
# Adjust the contraction_batch_size to a slightly larger value
adcc.memory_pool.contraction_batch_size = 8 * 1024

# Run an adc3 calculation:
pyscf_result = adcc.backends.import_scf_results(scfres)
singlets = adcc.adc3(pyscf_result, n_singlets=3)
triplets = adcc.adc3(singlets.matrix, n_triplets=3)

print(singlets.describe())
print()
print(triplets.describe())
