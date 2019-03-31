#!/usr/bin/env python3
import adcc

from import_data import import_data
from adcc.caching_policy import GatherStatisticsPolicy

import IPython

# Gather precomputed data
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Initialise the caching policy:
statistics_policy = GatherStatisticsPolicy()

# Run an adc2 calculation:
state = adcc.adc2x(data, n_singlets=2, n_triplets=2,
                   caching_policy=statistics_policy)

# Attach state densities
state = [adcc.attach_state_densities(kstate) for kstate in state]

print()
print(state[0].describe())
print()
print(state[1].describe())
print()

print("Tensor cache statistics:")
print(statistics_policy.call_count)

IPython.embed()
