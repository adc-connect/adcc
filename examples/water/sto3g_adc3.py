#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data

# Gather precomputed data
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc3 calculation:
singlets = adcc.adc3(data, n_singlets=3, conv_tol=1e-8)
triplets = adcc.adc3(singlets.matrix, n_triplets=4, conv_tol=1e-8)

# Attach state densities
singlets = adcc.attach_state_densities(singlets, method="adc2")
triplets = adcc.attach_state_densities(triplets, method="adc2")

print(singlets.describe())
print(triplets.describe())
