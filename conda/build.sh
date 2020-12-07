#!/bin/bash
set -eu

# First run tests
${PYTHON} setup.py test

# Check adcc finds pyscf and psi4
# TODO Installing psi4 or pyscf currently does not work, so disabled here
# ${PYTHON} <<- EOF
#     import adcc
#     assert "psi4" in adcc.backends.available()
#     assert "pyscf" in adcc.backends.available()
# EOF

# Now install adcc
${PYTHON} setup.py install --prefix=${PREFIX}
