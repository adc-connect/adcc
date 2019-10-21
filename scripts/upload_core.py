#!/usr/bin/env python3
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
    os.chdir(adccore.install_dir)
    filelist = []
    for globstr in adccore.file_globs:
        filelist += glob.glob(globstr)
    subprocess.run(["tar", "cvzf", fullpath] + filelist)
    os.chdir(olddir)
    return "dist/" + filename


def upload_tarball(filename):
    import json

    with open(os.path.dirname(__file__) + "/config.json") as fp:
        target = json.load(fp)["adccore"]

    print()
    try:
        input("Press enter to upload {} -> {}".format(filename, target))
    except KeyboardInterrupt:
        raise SystemExit("... aborted.")
    subprocess.run(["scp", filename, target], check=True)


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
