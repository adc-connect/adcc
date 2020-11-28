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
import glob
import warnings
import setuptools

from os.path import join

from setuptools import Extension, find_packages, setup
from setuptools.command.test import test as TestCommand
from setuptools.command.build_ext import build_ext as BuildCommand

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

# Version of the python bindings and adcc python package.
__version__ = "0.15.4"
adccore_version = ("0.14.4", "")  # (base version, unstable postfix)


def is_conda_build():
    return (
        os.environ.get("CONDA_BUILD", None) == "1"
        or os.environ.get("CONDA_EXE", None)
    )


def get_adccore_data():
    """Get a class providing info about the adccore library"""
    abspath = os.path.abspath("extension")
    if abspath not in sys.path:
        sys.path.insert(0, abspath)

    import AdcCore

    # TODO: find a more sustainable and clean solution ASAP
    if is_conda_build():
        os.environ["LDFLAGS_LD"] = \
            os.environ["LDFLAGS_LD"].replace("-dead_strip_dylibs", "")
        os.environ["LDFLAGS"] = \
            os.environ["LDFLAGS"].replace("-Wl,-dead_strip_dylibs", "")

    adccore = AdcCore.AdcCore()
    if not adccore.is_config_file_present \
       or adccore.version != adccore_version[0] \
       or not adccore.agrees_with_os_platform:
        # Is a checkout of adccore ok?
        disable_checkout = "DISABLE_ADCCORE_CHECKOUT" in os.environ

        # Get this version by building it or downloading it
        adccore.obtain(*adccore_version, allow_checkout=not disable_checkout)

    if adccore.version != adccore_version[0]:
        raise RuntimeError(
            "Version mismatch between requested adccore version (== {}) "
            "and available adccore version (== {})"
            "".format(adccore_version[0], adccore.version)
        )
    if not adccore.agrees_with_os_platform:
        raise RuntimeError(
            "Platform mismatch between this os (== {}) and adccore (== {})"
            "".format(AdcCore.get_platform(), adccore.platform)
        )
    return adccore


class LinkerDynamic:
    """A poor-man's ld-like resolver for shared libraries"""
    def __init__(self):
        # Initialise by environment and add default search locations:
        self.library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        self.library_paths += ["/usr/lib", "/lib"]

        for conf in ["/etc/ld.so.conf"] + glob.glob("/etc/ld.so.conf.d/*.conf"):
            if not os.path.isfile(conf):
                warnings.warn("Resolving configuration file {} failed. Probably "
                              "the file has been remove during distribution "
                              "upgrade and only a symbolic link to the removed "
                              "file is left.".format(conf))
                continue
            with open(conf, "r") as fp:
                for line in fp:
                    line = line.strip()
                    if line and not line.startswith("#") \
                       and os.path.isdir(line):
                        self.library_paths.append(line)

        if is_conda_build():
            # If we are in a CONDA environment, add conda library prefix
            libpath = os.path.join(os.environ["CONDA_PREFIX"], "lib")
            if os.path.isdir(libpath):
                self.library_paths.append(libpath)

    def find(self, library):
        for path in self.library_paths:
            name = join(path, library)
            if os.path.isfile(name):
                return name
        return None


def strip_readme():
    with open("README.md") as fp:
        return "".join([line for line in fp if not line.startswith("<img")])


def adccsetup(*args, **kwargs):
    """Wrapper around setup, displaying a link to adc-connect.org on any error."""
    if is_conda_build():
        kwargs.pop("install_requires")
        kwargs.pop("tests_require")
    try:
        setup(*args, **kwargs)
    except Exception as e:
        url = kwargs["url"] + "/installation.html"
        raise RuntimeError("Unfortunately adcc setup.py failed.\n"
                           "For hints how to install adcc, see {}."
                           "".format(url)) from e


#
# Pybind11 BuildExt
#
class GetPyBindInclude:
    """Helper class to determine the pybind11 include path

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname, opts=[]):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".cpp") as f:
        f.write("int main (int argc, char **argv) { return 0; }")
        try:
            extra_postargs = ["-Werror", flagname] + opts
            compiler.compile([f.name], extra_postargs=extra_postargs)
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler, opts=[]):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is preferred over c++11 (when it is available).
    """
    if has_flag(compiler, "-std=c++14", opts):
        return "-std=c++14"
    elif has_flag(compiler, "-std=c++11", opts):
        return "-std=c++11"
    else:
        raise RuntimeError("Unsupported compiler -- at least C++11 support "
                           "is needed!")


class BuildExt(BuildCommand):
    """A custom build extension for adding compiler-specific options."""
    def build_extensions(self):
        adccore = get_adccore_data()
        if adccore.has_source:
            adccore.build()  # Update adccore if required

        opts = []
        potential_opts = []
        if is_conda_build():
            newopt = "-Wno-error=unused-command-line-argument"
            if has_flag(self.compiler, newopt, opts):
                opts += [newopt]
        if sys.platform == "darwin":
            potential_opts += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
        if self.compiler.compiler_type == "unix":
            opts.append(cpp_flag(self.compiler, opts))
            potential_opts += ["-fvisibility=hidden", "-Wall", "-Wextra"]
        opts.extend([newopt for newopt in potential_opts
                     if has_flag(self.compiler, newopt, opts)])

        for ext in self.extensions:
            ext.extra_compile_args = opts
        BuildCommand.build_extensions(self)


#
# Building sphinx documentation
#
class BuildDocs(BuildSphinxDoc):
    def run(self):
        adccore = get_adccore_data()
        if adccore.has_source:
            adccore.build_documentation()
        elif not adccore.is_documentation_present:
            raise SystemExit("adccore documentation could not be found and "
                             "could not be installed.")

        try:
            import sphinx  # noqa F401
            import breathe  # noqa F401
        except ImportError:
            raise SystemExit("Sphinx or or one of its required plugins not "
                             "found.\nTry 'pip install -U adcc[build_docs]'")
        super().run()


#
# Pytest integration
#
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
if not os.path.isfile("adcc/__init__.py"):
    raise RuntimeError("Running setup.py is only supported "
                       "from top level of repository as './setup.py <command>'")

adccore = get_adccore_data()
# Check we have all dynamic library dependencies:
if sys.platform == "linux":
    ld = LinkerDynamic()
    for lib in adccore.required_dynamic_libraries:
        if not ld.find(lib):
            raise RuntimeError("Required dynamic library '{}' not found "
                               "on your system. Please install it first. "
                               "Consult adcc documentation for further details."
                               "".format(lib))

# Setup RPATH on Linux and MacOS
if sys.platform == "darwin":
    extra_link_args = ["-Wl,-rpath,@loader_path",
                       "-Wl,-rpath,@loader_path/adcc/lib"]
    runtime_library_dirs = []
elif sys.platform == "linux":
    extra_link_args = []
    runtime_library_dirs = ["$ORIGIN", "$ORIGIN/adcc/lib"]
else:
    raise OSError("Unsupported platform: {}".format(sys.platform))

# Setup build of the libadcc extension
ext_modules = [
    Extension(
        "libadcc", glob.glob("extension/*.cc"),
        include_dirs=[
            # Path to pybind11 headers
            GetPyBindInclude(),
            GetPyBindInclude(user=True),
            adccore.include_dir
        ],
        libraries=adccore.libraries,
        library_dirs=[adccore.library_dir],
        extra_link_args=extra_link_args,
        runtime_library_dirs=runtime_library_dirs,
        language="c++",
    ),
]

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
    version=__version__,
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
    ext_modules=ext_modules,
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
    cmdclass={"build_ext": BuildExt, "pytest": PyTest,
              "build_docs": BuildDocs},
)
