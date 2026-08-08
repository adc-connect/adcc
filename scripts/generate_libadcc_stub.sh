#!/bin/bash
set -e

cd ..
pybind11-stubgen --numpy-array-use-type-var libadcc -o .
# Remove the __backend__ module-level variable added by ExportAdcc.cc
python3 -c """
import re, pathlib
p = pathlib.Path('libadcc.pyi')
text = p.read_text()
text = re.sub(r'\n__backend__\s*:.*?}\n', '', text, flags=re.DOTALL)
p.write_text(text)
"""
ruff format libadcc.pyi
ruff check --fix libadcc.pyi
