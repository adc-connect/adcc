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
import json
import shlex
import shutil
import tempfile
import functools
import sysconfig
import setuptools
import subprocess

from distutils import log

from setuptools import find_packages, setup
from setuptools.command.test import test as TestCommand

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    have_pybind11 = True
except ImportError:
    # Failing for the first time is ok because of setup_requires
    from setuptools import Extension as Pybind11Extension
    from setuptools.command.build_ext import build_ext

    have_pybind11 = False

#
# Custom commands
#

try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    # No sphinx found -> make a dummy class
    class BuildDoc(setuptools.Command):
        user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        raise SystemExit("Sphinx not found. Try 'pip install -U adcc[build_docs]'")


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
# Setup libadcc extension
#
@functools.lru_cache()
def get_pkg_config():
    """
    Get path to pkg-config and set up the PKG_CONFIG environment variable.
    """
    pkg_config = os.environ.get('PKG_CONFIG', 'pkg-config')
    if shutil.which(pkg_config) is None:
        log.warn("WARNING: Pkg-config is not installed. Adcc may not be "
                 "able to find some dependencies.")
        return None

    # Some default places to search for pkg-config files:
    pkg_config_paths = [sysconfig.get_config_var('LIBDIR'),
                        os.path.expanduser("~/.local/lib")]
    for path in pkg_config_paths:
        if path is not None:
            path = os.path.join(path, 'pkgconfig')
            try:
                os.environ['PKG_CONFIG_PATH'] += ':' + path
            except KeyError:
                os.environ['PKG_CONFIG_PATH'] = path
    return pkg_config


def search_with_pkg_config(library, minversion=None, define_prefix=True):
    """
    Search the OS with pkg-config for a library and return the resulting
    cflags and libs stored inside the pc file. Also checks for a minimal
    version if `minversion` is not `None`.
    """
    pkg_config = get_pkg_config()
    if pkg_config:
        cmd = [pkg_config, "libtensorlight"]
        if define_prefix:
            cmd.append("--define-prefix")

        try:
            if minversion:
                subprocess.check_call([*cmd, f"--atleast-version={minversion}"])

            cflags = shlex.split(os.fsdecode(
                subprocess.check_output([*cmd, "--cflags"])))
            libs = shlex.split(os.fsdecode(
                subprocess.check_output([*cmd, "--libs"])))

            return cflags, libs
        except (OSError, subprocess.CalledProcessError):
            pass
    return None, None


def extract_library_dirs(libs):
    """
    From the `libs` flags returned by `search_with_pkg_config` extract
    the existing library directories.
    """
    libdirs = []
    for flag in libs:
        if flag.startswith("-L") and os.path.isdir(flag[2:]):
            libdirs.append(flag[2:])
    return libdirs


def request_urllib(url, filename):
    """Download a file from the net"""
    import urllib.request

    try:
        resp = urllib.request.urlopen(url)
    except urllib.request.HTTPError as e:
        return e.code

    if 200 <= resp.status < 300:
        with open(filename, 'wb') as fp:
            fp.write(resp.read())
    return resp.status


def assets_most_recent_release(project):
    """Return the assert urls attached to the most recent release of a project."""
    url = f"https://api.github.com/repos/{project}/releases"
    with tempfile.TemporaryDirectory() as tmpdir:
        fn = tmpdir + "/releases.json"
        request_urllib(url, fn)
        with open(fn) as fp:
            ret = json.loads(fp.read())
        assets = ret[0]["assets"]
        return [asset["browser_download_url"] for asset in assets]


def install_libtensor(url, destination):
    """
    Download libtensor from `url` and install at `destination`, removing possibly
    existing files from a previous installation.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading libtensorlight from {url} to {destination} ...")
        fn = os.path.basename(url)
        local = tmpdir + "/" + fn

        status_code = request_urllib(url, local)
        if status_code < 200 or status_code >= 300:
            raise RuntimeError(
                "Could not download libtensorlight. Check for a network issue "
                "and if in doubt see the adcc installation instructions "
                "(https://adc-connect.org/latest/installation.html)."
            )

        file_globs = [
            destination + "/include/libtensorlight/**/*.hh",
            destination + "/include/libtensor/**/*.hh",
            destination + "/include/libutil/**/*.hh",
            destination + "/lib/libtensorlight.so",
            destination + "/lib/libtensorlight.so.*",
            destination + "/lib/libtensorlight.dylib",
            destination + "/lib/libtensorlight.*.dylib",
            destination + "/lib/pkgconfig/libtensorlight.pc",
        ]

        for fglob in file_globs:
            for fn in glob.glob(fglob, recursive=True):
                log.info(f"Removing old libtensor file: {fn}")
                os.remove(fn)

        # Change to installation directory
        olddir = os.getcwd()
        os.chdir(destination)
        subprocess.run(["tar", "xf", local], check=True)
        os.chdir(olddir)


@functools.lru_cache()
def libadcc_extension():
    thisdir = os.path.dirname(__file__)

    # Initial lot of flags
    libraries = []
    library_dirs = []
    include_dirs = [os.path.join(thisdir, "libadcc")]
    extra_link_args = []
    extra_compile_args = ["-Wall", "-Wextra", "-Werror", "-O3"]
    runtime_library_dirs = []
    extra_objects = []
    define_macros = []
    search_system = True

    if sys.platform == "darwin" and is_conda_build():
        extra_compile_args += ["-Wno-unused-command-line-argument",
                               "-Wno-undefined-var-template"]

    platform_autoinstall = (
        sys.platform.startswith("linux") or sys.platform.startswith("darwin")
    )
    if platform_autoinstall and not is_conda_build():
        libtensor_autoinstall = "~/.local"
    else:
        # Not yet supported on other platforms and disabled
        # for conda builds
        libtensor_autoinstall = None

    # User-provided config
    adcc_config = os.environ.get('ADCC_CONFIG')
    if adcc_config and not os.path.isfile(adcc_config):
        raise FileNotFoundError(adcc_config)
    for siteconfig in [adcc_config, "siteconfig.py", "~/.adcc/siteconfig.py"]:
        if siteconfig is not None:
            siteconfig = os.path.expanduser(siteconfig)
            if os.path.isfile(siteconfig):
                log.info("Reading siteconfig file:", siteconfig)
                exec(open(siteconfig, "r").read())
                break

    # Keep track whether libtensor has been found
    found_libtensor = "tensorlight" in libraries
    lt_min_version = "2.9.9"

    if not found_libtensor:
        if search_system:  # Try to find libtensor on the OS using pkg-config
            log.info("Searching OS for libtensorlight using pkg-config")
            cflags, libs = search_with_pkg_config("libtensorlight", lt_min_version)

        # Try to download libtensor if not on the OS
        if (cflags is None or libs is None) and libtensor_autoinstall:
            assets = assets_most_recent_release("adc-connect/libtensor")
            url = []
            if sys.platform == "linux":
                url = [asset for asset in assets if "-linux_x86_64" in asset]
            elif sys.platform == "darwin":
                url = [asset for asset in assets
                       if "-macosx_" in asset and "_x86_64" in asset]
            else:
                raise AssertionError("Should not get to download for "
                                     "unspported platform.")
            if len(url) != 1:
                raise RuntimeError(
                    "Could not find a libtensor version to download. "
                    "Check your platform is supported and if in doubt see the "
                    "adcc installation instructions "
                    "(https://adc-connect.org/latest/installation.html)."
                )

            destdir = os.path.expanduser(libtensor_autoinstall)
            install_libtensor(url[0], destdir)
            os.environ['PKG_CONFIG_PATH'] += f":{destdir}/lib/pkg_config"
            cflags, libs = search_with_pkg_config("libtensorlight", lt_min_version)
            assert cflags is not None and libs is not None

        if cflags is not None and libs is not None:
            found_libtensor = True
            extra_compile_args.extend(cflags)
            extra_link_args.extend(libs)
            print(f"Using libtensorlight libraries: {libs}.")
            if sys.platform == "darwin":
                extra_link_args.append("-Wl,-rpath,@loader_path")
                for path in extract_library_dirs(libs):
                    extra_link_args.append(f"-Wl,-rpath,{path}")
            else:
                runtime_library_dirs.extend(extract_library_dirs(libs))

    if not found_libtensor:
        raise RuntimeError("Did not find the libtensorlight library.")

    sourcefiles = set(glob.glob("libadcc/**/*.cc", recursive=True))
    testfiles = glob.glob("libadcc/**/tests/*.cc", recursive=True)
    sourcefiles = sorted(sourcefiles.difference(testfiles))

    # This is needed on the first pass where pybind11 is not yet installed
    cxx_stdargs = dict()
    if have_pybind11:
        cxx_stdargs["cxx_std"] = 14
    return Pybind11Extension(
        "libadcc",
        sourcefiles,
        libraries=libraries,
        library_dirs=library_dirs,
        include_dirs=include_dirs,
        extra_link_args=extra_link_args,
        extra_compile_args=extra_compile_args,
        runtime_library_dirs=runtime_library_dirs,
        extra_objects=extra_objects,
        define_macros=define_macros,
        language="c++",
        **cxx_stdargs,
    )


#
# Main setup code
#
def is_conda_build():
    return (
        os.environ.get("CONDA_BUILD", None) == "1"
        or os.environ.get("CONDA_EXE", None)
    )


def adccsetup(*args, **kwargs):
    """Wrapper around setup, displaying a link to adc-connect.org on any error."""
    if is_conda_build():
        kwargs.pop("install_requires")
        kwargs.pop("setup_requires")
        kwargs.pop("tests_require")
    try:
        setup(*args, **kwargs)
    except Exception as e:
        url = kwargs["url"] + "/installation.html"
        raise RuntimeError("Unfortunately adcc setup.py failed.\n"
                           "For hints how to install adcc, see {}."
                           "".format(url)) from e


def read_readme():
    with open("README.md") as fp:
        return "".join([line for line in fp if not line.startswith("<img")])


if not os.path.isfile("adcc/__init__.py"):
    raise RuntimeError("Running setup.py is only supported "
                       "from top level of repository as './setup.py <command>'")

adccsetup(
    name="adcc",
    description="adcc:  Seamlessly connect your host program to ADC",
    long_description=read_readme(),
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
    version="0.15.5",
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
    ext_modules=[libadcc_extension()],
    zip_safe=False,
    #
    platforms=["Linux", "Mac OS-X"],
    python_requires=">=3.6",
    setup_requires=["pybind11 >= 2.6"],
    install_requires=[
        "opt_einsum >= 3.0",
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
    cmdclass={"build_ext": build_ext, "pytest": PyTest,
              "build_docs": BuildDoc},
)
