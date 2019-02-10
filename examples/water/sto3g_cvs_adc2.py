#!/usr/bin/env python3

import adcc
from import_data import import_data
import IPython

# Gather preliminary data
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an cvs-adc2 calculation:
state = adcc.cvs_adc2(data, n_core_orbitals=1, n_singlets=2, n_triplets=2)

# Attach state densities
state = [adcc.attach_state_densities(kstate) for kstate in state]

IPython.embed()
