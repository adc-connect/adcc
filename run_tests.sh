#!/bin/bash

PYDIR="adcc"

if [ ! -d "$PYDIR" ]; then
	echo "Could not find python adcc module directory $PYDIR" >&2
	exit 1
fi

PART="$1"
shift

echo "Notice: This script is poorly maintained. It should be replaced by setup.py stuff."
if [ "$PART" == "gdb_pytest" ]; then
	gdb --tui --ex "catch throw" --ex run --args python3 -m pytest "$PYDIR" "$@"
elif [ "$PART" == "valgrind_pytest" ]; then
	valgrind python3 -m pytest "$PYDIR" "$@"
	echo "Unknown part: $PART, known are: gdb_pytest valgrind_pytest" >&2
	exit 1
fi
