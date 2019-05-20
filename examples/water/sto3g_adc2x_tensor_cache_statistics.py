#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data
from adcc.caching_policy import GatherStatisticsPolicy

# Gather precomputed data
data = import_data()

# Initialise the caching policy:
statistics_policy = GatherStatisticsPolicy()

mp = adcc.LazyMp(adcc.ReferenceState(data), statistics_policy)

# Run an adc2 calculation:
singlets = adcc.adc2x(mp, n_singlets=5, conv_tol=1e-8)
triplets = adcc.adc2x(mp, n_triplets=5, conv_tol=1e-8)

# Attach state densities
singlets = adcc.attach_state_densities(singlets)
triplets = adcc.attach_state_densities(triplets)

print(singlets.describe())
print(triplets.describe())

print("Tensor cache statistics:")
print(statistics_policy.call_count)
