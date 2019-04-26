#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data

# Gather precomputed data
data = import_data()

# Make it unrestricted
data["restricted"] = False

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an unrestricted adc2 calculation:
states = adcc.adc2(data, n_states=10, conv_tol=1e-8)

# Attach state densities
states = adcc.attach_state_densities(states)

print(states.describe())
