#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data
from adcc.caching_policy import GatherStatisticsPolicy

# Gather precomputed data
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Initialise the caching policy:
statistics_policy = GatherStatisticsPolicy()

refstate = adcc.tmp_build_reference_state(data)
mp = adcc.LazyMp(refstate, statistics_policy)

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
