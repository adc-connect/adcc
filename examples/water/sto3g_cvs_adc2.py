#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
from import_data import import_data

import adcc

# Gather preliminary data
data = import_data()

# Run an cvs-adc2 calculation:
singlets = adcc.cvs_adc2(data, n_core_orbitals=1, n_singlets=1, conv_tol=1e-8)
triplets = adcc.cvs_adc2(singlets.matrix, n_triplets=2, conv_tol=1e-8)
# Note: Above n_core_orbitals is not required again, since the precise CVS
#       splitting is already encoded in the matrix.

# Attach state densities
singlets = adcc.attach_state_densities(singlets)
triplets = adcc.attach_state_densities(triplets)

print(singlets.describe())
print(triplets.describe())
