#!/bin/sh -e

rm -rf adcc/tests/*/
find adcc/tests -type f -not -name "smoke_test.py" -a -name "*test*.py" -exec rm {} \;
