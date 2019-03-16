#!/bin/bash

PYDIR="adcc"

if [ ! -d "$PYDIR" ]; then
	echo "Could not find python adcc module directory $PYDIR" >&2
	exit 1
fi

PART="$1"
shift

if [ "$PART" == "testdata" ]; then
	$PYDIR/testdata/0_download_testdata.sh
elif [ "$PART" == "pytest" ]; then
	python3 -m pytest "$PYDIR" "$@"
elif [ "$PART" == "gdb_pytest" ]; then
	gdb --tui --ex "catch throw" --ex run --args python3 -m pytest "$PYDIR" "$@"
elif [ "$PART" == "valgrind_pytest" ]; then
	valgrind python3 -m pytest "$PYDIR" "$@"
elif [ "$PART" == "" ]; then
	$0 "testdata" || exit 1
	$0 "pytest"
else
	echo "Unknown part: $PART, known are: testdata pytest gdb_pytest valgrind_pytest" >&2
	exit 1
fi
