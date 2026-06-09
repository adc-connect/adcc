#!/bin/bash

pybind11-stubgen libadcc -o .
ruff format libadcc.pyi
