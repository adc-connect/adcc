#!/usr/bin/env python3

import adcc
from import_data import import_data
import IPython

# Gather preliminary data and import it into an HfData object
data = import_data()
hfdata = adcc.HfData.from_dict(data)

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run an adc2 calculation:
state = adcc.adc2(hfdata, n_singlets=6, n_triplets=6, solver_method="jacobi_diis",
                  conv_tol=1e-7)

# Attach state densities
# state = [adcc.attach_state_densities(kstate, method="adc2")
#          for kstate in state]

# IPython.embed()
