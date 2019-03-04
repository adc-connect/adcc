#!/usr/bin/env python3
import adcc
import IPython

from adcc import tmp_run_prelim
from import_data import import_data

# Gather precomputed data
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc1 calculation:
state = adcc.adc1(data, n_singlets=5, n_triplets=5)

# Attach state densities
state = [adcc.attach_state_densities(kstate) for kstate in state]

IPython.embed()
