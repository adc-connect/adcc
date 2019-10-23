#!/bin/sh -e
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

if [ ! -f scripts/upload_to_pypi.sh -o ! -f setup.py ]; then
	echo "Please run from top dir of repository" >&2
	exit 1
fi
if [ ! -f extension/adccore/adccore_config.json ]; then
	echo "Please compile and install / download adccore first" >&2
	exit 1
fi

rm -rf dist
python3 setup.py sdist bdist_wheel
twine upload -s dist/*
