#!/bin/bash
set -e

pybind11-stubgen libadcc -o .
ruff format libadcc.pyi
ruff check --fix libadcc.pyi
