#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
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
import os
import subprocess


def build_docs():
    subprocess.run("rm -r docs/api".split())
    subprocess.run("./setup.py build_docs".split(), check=True)
    return "build/sphinx/html"


def upload_docs(outdir):
    import json

    with open(os.path.dirname(__file__) + "/config.json") as fp:
        target = json.load(fp)["documentation"]
    subprocess.run("rsync -P -rvzc --exclude .buildinfo --exclude objects.inv "
                   "--delete {}/ {} --cvs-exclude".format(outdir, target).split(),
                   check=True)


def main():
    if not os.path.isfile("scripts/upload_documentation.py") or \
       not os.path.isfile("setup.py"):
        raise SystemExit("Please run from top dir of repository")

    outdir = build_docs()
    upload_docs(outdir)


if __name__ == "__main__":
    main()
