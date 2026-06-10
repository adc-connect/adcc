#!/bin/bash
set -e

cd ..
pybind11-stubgen libadcc -o .
ruff format libadcc.pyi
ruff check --fix libadcc.pyi
