#!/usr/bin/env python3
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
import re
import ast
import sys
import glob
import subprocess


def get_adccore_data():
    """Get a class providing info about the adccore library"""
    abspath = os.path.abspath("extension")
    if abspath not in sys.path:
        sys.path.insert(0, abspath)

    from AdcCore import AdcCore

    return AdcCore()


def extract_postfix(fn):
    with open(fn, "r") as fp:
        for line in fp:
            match = re.match(r"^ *adccore_version *= *(\([^()]*\))", line)
            if match:
                return ast.literal_eval(match.group(1))[1]
        else:
            raise RuntimeError("Could not extract adccore version from " + fn)


def make_tarball(adccore, postfix=None):
    filename = adccore.get_tarball_name(postfix=postfix)
    fullpath = os.path.abspath("dist/" + filename)

    os.makedirs("dist", exist_ok=True)
    olddir = os.getcwd()
    os.chdir(adccore.top_dir)
    filelist = []
    for globstr in adccore.file_globs:
        relglob = os.path.relpath(globstr, adccore.top_dir)
        filelist += glob.glob(relglob)
    subprocess.run(["tar", "cvzf", fullpath] + filelist)
    os.chdir(olddir)
    return "dist/" + filename


def print_input(text, interactive=sys.stdin.isatty()):
    if interactive:
        try:
            input(text)
        except KeyboardInterrupt:
            raise SystemExit("... aborted.")
    else:
        print(text, "... yes")


def upload_tarball(filename):
    import json

    with open(os.path.dirname(__file__) + "/config.json") as fp:
        target = json.load(fp)["adccore"]

    print()
    print_input("Press enter to upload {} -> {}".format(filename, target))

    host, tdir = target.split(":")
    command = "put {} {}/".format(filename, tdir)
    subprocess.run(["sftp", "-b", "-", host], input=command.encode(), check=True)


def main():
    if not os.path.isfile("scripts/upload_core.py") or \
       not os.path.isfile("setup.py"):
        raise SystemExit("Please run from top dir of repository")

    # Get postfix from setup.py
    postfix = extract_postfix("setup.py")

    # Build adccore and pack tarball
    adccore = get_adccore_data()
    adccore.build()
    filename = make_tarball(adccore, postfix=postfix)
    upload_tarball(filename)


if __name__ == "__main__":
    main()
