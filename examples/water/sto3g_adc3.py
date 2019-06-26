#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data

# Gather precomputed data
data = import_data()

# Run an adc3 calculation:
singlets = adcc.adc3(data, n_singlets=3, conv_tol=1e-8)
triplets = adcc.adc3(singlets.matrix, n_triplets=4, conv_tol=1e-8)
print(singlets.describe())
print(triplets.describe())
