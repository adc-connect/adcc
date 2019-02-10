#!/usr/bin/env python3

import adcc
from import_data import import_data
import IPython

# Gather precomputed data
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc2 calculation:
state = adcc.adc2(data, n_singlets=5, n_triplets=5)

# Attach state densities
state = [adcc.attach_state_densities(kstate) for kstate in state]

IPython.embed()
