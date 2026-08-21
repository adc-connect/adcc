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
ruff format libadcc.pyi
ruff check --fix libadcc.pyi
