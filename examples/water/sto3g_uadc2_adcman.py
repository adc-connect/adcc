#!/usr/bin/env python3

from adcc.solver.adcman import jacobi_davidson
from import_data import import_data
import adcc
import IPython

# Gather preliminary data and import it into an HfData object
data = import_data()
data["restricted"] = False

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run the initial preparation (MP2, intermediates, ...)
res = adcc.tmp_run_prelim(data, "adc2", n_guess_singles=10)

# Setup the matrix
matrix = adcc.AdcMatrix(res.method, res.ground_state)

# Solve for 6 states with some default output
state = jacobi_davidson(matrix, print_level=100, n_states=10)
IPython.embed()
