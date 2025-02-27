#!/bin/bash

if ! which sha256sum &> /dev/null; then
    echo "sha256sum not installed" >&2
    exit 1
fi

# move in the folder of the script
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPT_DIR"

# Take the filename as positional argument
# or default to SHA256SUMS
SHAFILE=${1:-SHA256SUMS}
if [ -f "$SHAFILE" ] && [ -s "$SHAFILE" ]; then
    echo "The file $SHAFILE already exists" >&2
    exit 1
fi

# compute the sha256sums for all hdf5 files in the workdir
for file in *.hdf5; do
    [ -f "$file" ] || break
    sha256sum "$file" >> "$SHAFILE"
done
