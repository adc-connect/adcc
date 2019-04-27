#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
from import_data import import_data

import adcc

# Gather precomputed data
data = import_data()

# Run an adc2 calculation:
singlets = adcc.adc2(data, n_singlets=5, conv_tol=1e-8)
triplets = adcc.adc2(singlets.matrix, n_triplets=5, conv_tol=1e-8)

# Attach state densities
singlets = adcc.attach_state_densities(singlets)
triplets = adcc.attach_state_densities(triplets)

print(singlets.describe())
print(triplets.describe())
