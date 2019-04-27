#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
from import_data import import_data

import adcc

# Gather precomputed data
data = import_data()

# Make it unrestricted
data["restricted"] = False

# Run an unrestricted adc2 calculation:
states = adcc.adc2(data, n_states=10, conv_tol=1e-8)

# Attach state densities
states = adcc.attach_state_densities(states)

print(states.describe())
