#!/bin/bash
set -eu

# Setup wrapper scripts for ccache during adcc builds
ccache -M 1Gi

rm -rf $HOME/bin/cc
rm -rf $HOME/bin/cxx
echo -e '#!/bin/sh\n' "ccache $CC \$@" > $HOME/bin/cc
echo -e '#!/bin/sh\n' "ccache $CXX \$@" > $HOME/bin/cxx
chmod +x $HOME/bin/cc
chmod +x $HOME/bin/cxx
export CC="$HOME/bin/cc"
export CXX="$HOME/bin/cxx"
echo ++CC=$CC
echo ++CXX=$CXX

# First run tests
${PYTHON} setup.py test

# Check adcc finds pyscf and psi4
# TODO Installing psi4 currently does not work, so disabled here
${PYTHON} <<- EOF
    import adcc
    import sys

    # assert "psi4" in adcc.backends.available()
    if sys.version_info.minor == 7:
        assert "pyscf" in adcc.backends.available()
EOF

# Now install adcc
${PYTHON} setup.py install --prefix=${PREFIX}
