#!/bin/bash
set -e

cd ..
pybind11-stubgen libadcc -o .
# Remove the __backend__ module-level variable added by ExportAdcc.cc
# Unions currently produce 'libadcc.Tensor | T', which is not correctly
# resolved to 'Tensor | T' by pybind11-stubgen.
python3 -c """
import re, pathlib
p = pathlib.Path('libadcc.pyi')
text = p.read_text()
text = re.sub(r'\n__backend__\s*:.*?}\n', '', text, flags=re.DOTALL)
text = text.replace('libadcc.Tensor', 'Tensor')
p.write_text(text)
"""
# Format the stub with the ruff version pinned in .pre-commit-config.yaml, so that
# the result is identical to what the commit hook produces.
# The first run is expected to modify the file, thus try a second time.
# If the second run also fails, there is some problem.
pre-commit run --files libadcc.pyi || pre-commit run --files libadcc.pyi
