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

"""Setup for adcc"""
import os
import sys
import setuptools

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

import setupext

#
# Building sphinx documentation
#

try:
    from sphinx.setup_command import BuildDoc as BuildSphinxDoc
except ImportError:
    # No sphinx found -> make a dummy class
    class BuildSphinxDoc(setuptools.Command):
        user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class BuildDocs(BuildSphinxDoc):
    def run(self):
        try:
            import breathe  # noqa F401

            import sphinx  # noqa F401
        except ImportError:
            raise SystemExit("Sphinx or or one of its required plugins not "
                             "found.\nTry 'pip install -U adcc[build_docs]'")
        super().run()


class PyTest(TestCommand):
    user_options = [
        ("mode=", "m", "Mode for the testsuite (fast or full)"),
        ("skip-update", "s", "Skip updating testdata"),
        ("pytest-args=", "a", "Arguments to pass to pytest"),
    ]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""
        self.mode = "fast"
        self.skip_update = False

    def finalize_options(self):
        if self.mode not in ["fast", "full"]:
            raise Exception("Only test modes 'fast' and 'full' are supported")

    def run_tests(self):
        import shlex

        # import here, cause outside the eggs aren't loaded
        import pytest

        if not os.path.isdir("adcc/testdata"):
            raise RuntimeError("Can only test from git repository, "
                               "not from installation tarball.")

        args = ["adcc"]
        args += ["--mode", self.mode]
        if self.skip_update:
            args += ["--skip-update"]
        args += shlex.split(self.pytest_args)
        errno = pytest.main(args)
        sys.exit(errno)


#
# Main setup code
#
def strip_readme():
    with open("README.md") as fp:
        return "".join([line for line in fp if not line.startswith("<img")])


def adccsetup(*args, **kwargs):
    """Wrapper around setup, displaying a link to adc-connect.org on any error."""
    if setupext.is_conda_build():
        kwargs.pop("install_requires")
        kwargs.pop("tests_require")
    try:
        setup(*args, **kwargs)
    except Exception as e:
        url = kwargs["url"] + "/installation.html"
        raise RuntimeError("Unfortunately adcc setup.py failed.\n"
                           "For hints how to install adcc, see {}."
                           "".format(url)) from e


def main():
    if not os.path.isfile("adcc/__init__.py"):
        raise RuntimeError("Running setup.py is only supported "
                           "from top level of repository as './setup.py <command>'")

    adccsetup(
        name="adcc",
        description="adcc:  Seamlessly connect your host program to ADC",
        long_description=strip_readme(),
        long_description_content_type="text/markdown",
        keywords=[
            "ADC", "algebraic-diagrammatic", "construction", "excited", "states",
            "electronic", "structure", "computational", "chemistry", "quantum",
            "spectroscopy",
        ],
        #
        author="Michael F. Herbst, Maximilian Scheurer",
        author_email="developers@adc-connect.org",
        license="GPL v3",
        url="https://adc-connect.org",
        project_urls={
            "Source": "https://github.com/adc-connect/adcc",
            "Issues": "https://github.com/adc-connect/adcc/issues",
        },
        #
        version="0.15.4",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
            "License :: Free For Educational Use",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Chemistry",
            "Topic :: Education",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: POSIX :: Linux",
        ],
        #
        packages=find_packages(exclude=["*.test*", "test"]),
        package_data={"adcc": ["lib/*.so", "lib/*.dylib",
                               "lib/*.so.*",
                               "lib/libadccore_LICENSE"],
                      "": ["LICENSE*"]},
        ext_modules=setupext.get_extensions(),
        zip_safe=False,
        #
        platforms=["Linux", "Mac OS-X"],
        python_requires=">=3.6",
        install_requires=[
            "opt_einsum >= 3.0",
            "pybind11 >= 2.6",
            "numpy >= 1.14",
            "scipy >= 1.2",
            "matplotlib >= 3.0",
            "h5py >= 2.9",
            "tqdm >= 4.30",
            "pandas >= 0.25.0",
        ],
        tests_require=["pytest", "pytest-cov", "pyyaml"],
        extras_require={
            "build_docs": ["sphinx>=2", "breathe", "sphinxcontrib-bibtex",
                           "sphinx-automodapi"],
        },
        #
        cmdclass={"build_ext": setupext.BuildExt, "pytest": PyTest,
                  "build_docs": BuildDocs},
    )


if __name__ == "__main__":
    main()
