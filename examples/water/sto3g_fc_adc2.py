#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data

# Gather precomputed data
data = import_data()

# Run an adc2 calculation:
singlets = adcc.adc2(data, frozen_core=1, n_singlets=5, conv_tol=1e-8)
triplets = adcc.adc2(singlets.matrix, frozen_core=1, n_triplets=5, conv_tol=1e-8)
print(singlets.describe())
print(triplets.describe())
