#!/usr/bin/env python3

import adcc
from import_data import import_data
import IPython

# Gather precomputed data
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc3 calculation:
state = adcc.adc3(data, n_singlets=3, n_triplets=3)

# Attach state densities
state = [adcc.attach_state_densities(kstate, method="adc2")
         for kstate in state]

IPython.embed()
