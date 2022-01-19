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
import time
import shlex
import shutil
import tempfile
import functools
import sysconfig
import subprocess

from distutils import log

from setuptools import Command, find_packages, setup
from setuptools.command.test import test as TestCommand

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    have_pybind11 = True
except ImportError:
    # Failing for the first time is ok because of setup_requires
    from setuptools import Extension as Pybind11Extension
    from setuptools.command.build_ext import build_ext

    have_pybind11 = False

try:
    from sphinx.setup_command import BuildDoc as SphinxBuildDoc

    have_sphinx = True
except ImportError:
    have_sphinx = False

#
# Custom commands
#

if have_sphinx:
    class BuildDocs(SphinxBuildDoc):
        def run(self):
            subprocess.check_call(["doxygen"], cwd="docs")
            super().run()
else:
    # No sphinx found -> make a dummy class
    class BuildDocs(Command):
        user_options = []

        def initialize_options(self):
            pass

        def finalize_options(self):
            pass

        def run(self):
            raise RuntimeError(
                "Sphinx not found. Try 'pip install -U adcc[build_docs]'"
            )


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


class CppTest(Command):
    description = "Build and run C++ tests"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        test_executable = self.compile_test_executable()
        subprocess.check_call([test_executable])

    def compile_test_executable(self):
        from distutils.ccompiler import new_compiler
        from distutils.sysconfig import customize_compiler

        output_dir = os.path.abspath("build/cpptest")
        test_executable = output_dir + "/libadcc_tests"
        if os.path.isfile(test_executable):
            print(f"Skipping recompilation of {os.path.relpath(test_executable)}. "
                  "Delete file if recompilation desired.")
            return test_executable  # Don't recompile

        # Download catch
        if not os.path.isfile(output_dir + "/catch2/catch.hpp"):
            os.makedirs(output_dir + "/catch2", exist_ok=True)
            base = "https://github.com/catchorg/Catch2/releases/download/"
            request_urllib(base + "v2.7.0/catch.hpp",
                           output_dir + "/catch2/catch.hpp")

        # Adapt stuff from libadcc extension
        libadcc = libadcc_extension()
        include_dirs = libadcc.include_dirs + [output_dir]
        sources = libadcc_sources("cpptest")
        extra_compile_args = [arg for arg in libadcc.extra_compile_args
                              if not any(arg.startswith(start)
                                         for start in ("-fvisibility", "-g", "-O"))]

        # Reduce optimisation a bit to ensure that the debugging experience is good
        if "--coverage" in extra_compile_args:
            extra_compile_args += ["-O0", "-g"]
        else:
            extra_compile_args += ["-O1", "-g"]

        # Bring forward stuff from libadcc
        compiler = new_compiler(verbose=self.verbose)
        customize_compiler(compiler)
        objects = compiler.compile(
            sources, output_dir, libadcc.define_macros, include_dirs,
            debug=True, extra_postargs=extra_compile_args
        )

        if libadcc.extra_objects:
            objects.extend(libadcc.extra_objects)

        compiler.link_executable(
            objects, "libadcc_tests", output_dir, libadcc.libraries,
            libadcc.library_dirs, libadcc.runtime_library_dirs, debug=True,
            extra_postargs=libadcc.extra_link_args, target_lang=libadcc.language
        )
        return test_executable


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
        for _ in range(10):
            status = request_urllib(url, fn)
            if 200 <= status < 300:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Error downloading asset list from {url} "
                               f"... Response: {status}")

        with open(fn) as fp:
            ret = json.loads(fp.read())
        assets = [ret[i]["assets"] for i in range(len(ret))][0]
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
        os.makedirs(destination, exist_ok=True)
        os.chdir(destination)
        subprocess.run(["tar", "xf", local], check=True)
        os.chdir(olddir)


def libadcc_sources(target):
    sourcefiles = set(glob.glob("libadcc/**/*.cc", recursive=True))
    unittests = glob.glob("libadcc/tests/*.cc", recursive=True)
    exportfiles = glob.glob("libadcc/pyiface/*.cc")
    if target == "extension":
        return sorted(sourcefiles.difference(unittests))
    elif target == "cpptest":
        return sorted(sourcefiles.difference(exportfiles))
    else:
        raise ValueError(f"Unknown target: {target}")


@functools.lru_cache()
def libadcc_extension():
    # Initial lot of flags
    flags = dict(
        libraries=[],
        library_dirs=[],
        include_dirs=[],
        extra_link_args=[],
        extra_compile_args=["-Wall", "-Wextra", "-Werror", "-O3"],
        runtime_library_dirs=[],
        extra_objects=[],
        define_macros=[],
        search_system=True,
        coverage=False,
        libtensor_autoinstall="~/.local",
        libtensor_url=None,
    )

    if sys.platform == "darwin" and is_conda_build():
        flags["extra_compile_args"] += ["-Wno-unused-command-line-argument",
                                        "-Wno-undefined-var-template"]

    platform_autoinstall = (
        sys.platform.startswith("linux") or sys.platform.startswith("darwin")
    )
    if platform_autoinstall and not is_conda_build():
        flags["libtensor_autoinstall"] = "~/.local"
    else:
        # Not yet supported on other platforms and disabled
        # for conda builds
        flags["libtensor_autoinstall"] = None

    # User-provided config
    adcc_config = os.environ.get('ADCC_CONFIG')
    if adcc_config and not os.path.isfile(adcc_config):
        raise FileNotFoundError(adcc_config)
    for siteconfig in [adcc_config, "siteconfig.py", "~/.adcc/siteconfig.py"]:
        if siteconfig is not None:
            siteconfig = os.path.expanduser(siteconfig)
            if os.path.isfile(siteconfig):
                log.info("Reading siteconfig file:", siteconfig)
                exec(open(siteconfig, "r").read(), flags)
                flags.pop("__builtins__")
                break

    # Keep track whether libtensor has been found
    found_libtensor = "tensorlight" in flags["libraries"]
    lt_min_version = "3.0.1"

    if not found_libtensor:
        if flags["search_system"]:  # Find libtensor on the OS using pkg-config
            log.info("Searching OS for libtensorlight using pkg-config")
            cflags, libs = search_with_pkg_config("libtensorlight", lt_min_version)

        # Try to download libtensor if not on the OS
        if (cflags is None or libs is None) and flags["libtensor_autoinstall"]:
            if flags["libtensor_url"]:
                url = flags["libtensor_url"]
            else:
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
                url = url[0]

            destdir = os.path.expanduser(flags["libtensor_autoinstall"])
            install_libtensor(url, destdir)
            os.environ['PKG_CONFIG_PATH'] += f":{destdir}/lib/pkgconfig"
            cflags, libs = search_with_pkg_config("libtensorlight", lt_min_version)
            assert cflags is not None and libs is not None

        if cflags is not None and libs is not None:
            found_libtensor = True
            flags["extra_compile_args"].extend(cflags)
            flags["extra_link_args"].extend(libs)
            log.info(f"Using libtensorlight libraries: {libs}.")
            if sys.platform == "darwin":
                flags["extra_link_args"].append("-Wl,-rpath,@loader_path")
                for path in extract_library_dirs(libs):
                    flags["extra_link_args"].append(f"-Wl,-rpath,{path}")
            else:
                flags["runtime_library_dirs"].extend(extract_library_dirs(libs))

    if not found_libtensor:
        raise RuntimeError("Did not find the libtensorlight library.")

    # Filter out the arguments to pass to Pybind11Extension
    extargs = {k: v for k, v in flags.items()
               if k in ("libraries", "library_dirs", "include_dirs",
                        "extra_link_args", "extra_compile_args",
                        "runtime_library_dirs", "extra_objects",
                        "define_macros")}
    if have_pybind11:
        # This is needed on the first pass where pybind11 is not yet installed
        extargs["cxx_std"] = 14

    ext = Pybind11Extension("libadcc", libadcc_sources("extension"),
                            language="c++", **extargs)
    if flags["coverage"]:
        ext.extra_compile_args += ["--coverage", "-O0", "-g"]
        ext.extra_link_args += ["--coverage"]
    return ext


#
# Main setup code
#
def is_conda_build():
    return (
        os.environ.get("CONDA_BUILD", None) == "1"
        or os.environ.get("CONDA_EXE", None) is not None
    )


def adccsetup(*args, **kwargs):
    """Wrapper around setup, displaying a link to adc-connect.org on any error."""
    if is_conda_build():
        kwargs.pop("install_requires")
        kwargs.pop("setup_requires")
        kwargs.pop("tests_require")
        kwargs.pop("extras_require")
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
    version="0.15.11",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "License :: Free For Educational Use",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Education",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    #
    packages=find_packages(exclude=["*.test*", "test"]),
    ext_modules=[libadcc_extension()],
    zip_safe=False,
    #
    platforms=["Linux", "Mac OS-X"],
    python_requires=">=3.7",
    setup_requires=["pybind11 >= 2.6"],
    install_requires=[
        "opt_einsum >= 3.0",
        "numpy >= 1.14",
        "scipy >= 1.2",
        "h5py >= 2.9",
        "tqdm >= 4.30",
    ],
    tests_require=["pytest", "pytest-cov", "pyyaml", "pandas >= 0.25.0"],
    extras_require={
        "build_docs": ["sphinx>=2", "breathe", "sphinxcontrib-bibtex",
                       "sphinx-automodapi", "sphinx-rtd-theme"],
        "analysis": ["matplotlib >= 3.0", "pandas >= 0.25.0"],
    },
    #
    cmdclass={"build_ext": build_ext, "pytest": PyTest,
              "build_docs": BuildDocs, "cpptest": CppTest},
)
