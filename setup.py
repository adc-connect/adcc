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
import logging
import platform
import tempfile
import functools
import sysconfig
import subprocess

from pathlib import Path

from setuptools import Command, setup

from pybind11.setup_helpers import Pybind11Extension, build_ext


log = logging.getLogger()


#
# Custom commands
#
class BuildDocs(Command):
    description = "Build the C++ and python documentation"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build_folder = Path(__file__).parent / "build"
        docs_folder = Path(__file__).parent / "docs"
        if not docs_folder.is_dir():
            raise RuntimeError("setup.py is expected to be called from the "
                               "top level project directory: ./setup.py build_docs")

        try:  # generate the documentation for libadcc
            # we need to create the folder for doxygen
            (build_folder / "libadcc_docs").mkdir(parents=True, exist_ok=True)
            doxyfile = docs_folder / "Doxyfile"
            subprocess.check_call(["doxygen", str(doxyfile)])
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError(
                f"Could not build C++ documentation with doxygen: {e}"
            )
        try:  # generate full documentation
            output = build_folder / "docs"
            subprocess.check_call([
                "sphinx-build", "-M", "html", str(docs_folder), str(output)
            ])
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError(
                f"Could not build adcc documentation with sphinx: {e}"
            )


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
        from setuptools._distutils.ccompiler import new_compiler
        from setuptools._distutils.sysconfig import customize_compiler

        output_dir = Path(__file__).parent / "build" / "cpptest"
        test_executable = output_dir / "libadcc_tests"
        if test_executable.is_file():
            rel_path = test_executable.relative_to(Path.cwd())
            print(f"Skipping recompilation of {rel_path}. "
                  "Delete file if recompilation desired.")
            return test_executable  # Don't recompile

        # Download catch
        catch_header = output_dir / "catch2" / "catch.hpp"
        if not catch_header.is_file():
            (output_dir / "catch2").mkdir(parents=True, exist_ok=True)
            base = "https://github.com/catchorg/Catch2/releases/download/"
            request_urllib(base + "v2.13.9/catch.hpp", str(catch_header))

        # Adapt stuff from libadcc extension
        libadcc = libadcc_extension()
        include_dirs = libadcc.include_dirs + [str(output_dir)]
        sources = libadcc_sources("cpptest")
        extra_compile_args = [
            arg for arg in libadcc.extra_compile_args
            if not any(arg.startswith(start) for
                       start in ("-fvisibility", "-g", "-O"))
        ]

        # Reduce optimisation a bit to ensure that the debugging experience is good
        if "--coverage" in extra_compile_args:
            extra_compile_args += ["-O0", "-g"]
        else:
            extra_compile_args += ["-O1", "-g"]

        # Bring forward stuff from libadcc
        compiler = new_compiler()
        customize_compiler(compiler)
        objects = compiler.compile(
            sources, str(output_dir), libadcc.define_macros, include_dirs,
            debug=True, extra_postargs=extra_compile_args
        )

        if libadcc.extra_objects:
            objects.extend(libadcc.extra_objects)

        compiler.link_executable(
            objects, "libadcc_tests", str(output_dir), libadcc.libraries,
            libadcc.library_dirs, libadcc.runtime_library_dirs, debug=True,
            extra_postargs=libadcc.extra_link_args, target_lang=libadcc.language
        )
        return test_executable


#
# Setup libadcc extension
#
def append_to_pkg_config_path(*paths: str):
    """Append the given paths to the `PKG_CONFIG_PATH`."""
    for path in paths:
        if "PKG_CONFIG_PATH" in os.environ:
            os.environ["PKG_CONFIG_PATH"] += os.pathsep + path
        else:
            os.environ["PKG_CONFIG_PATH"] = path


@functools.lru_cache()
def get_pkg_config():
    """
    Get path to pkg-config and set up the PKG_CONFIG environment variable.
    """
    pkg_config = os.environ.get("PKG_CONFIG", "pkg-config")
    if shutil.which(pkg_config) is None:
        raise RuntimeError("Pkg-config is not installed. Adcc is not able to "
                           "find or autoinstall the libtensorlight library.")

    # Some default places to search for pkg-config files
    pkg_config_paths = [
        sysconfig.get_config_var("LIBDIR"), Path.home() / ".local" / "lib"
    ]
    for path in pkg_config_paths:
        if path is None:
            continue
        elif not isinstance(path, Path):
            path = Path(path)
        path = str(path / "pkgconfig")
        append_to_pkg_config_path(path)
    return pkg_config


def search_with_pkg_config(library: str, minversion=None,
                           define_prefix: bool = True):
    """
    Search the OS with pkg-config for a library and return the resulting
    cflags and libs stored inside the pc file. Also checks for a minimal
    version if `minversion` is not `None`.
    """
    pkg_config = get_pkg_config()

    cmd = [pkg_config, library]
    if define_prefix:
        cmd.append("--define-prefix")
    try:
        # first verify that the lib has the correct version
        if minversion:
            subprocess.check_call([*cmd, f"--atleast-version={minversion}"])
        # then get the include path for the headers for the compiler
        cflags = shlex.split(os.fsdecode(
            subprocess.check_output([*cmd, "--cflags"])
        ))
        # and the path to the library file and its name for the linker
        libs = shlex.split(os.fsdecode(
            subprocess.check_output([*cmd, "--libs"])
        ))
        return cflags, libs
    except (OSError, subprocess.CalledProcessError):
        return None, None


def extract_library_dirs(libs: list[str]) -> list[str]:
    """
    From the `libs` flags returned by `search_with_pkg_config` extract
    the existing library directories.
    """
    libdirs = []
    for flag in libs:
        if flag.startswith("-L") and Path(flag[2:]).is_dir():
            libdirs.append(flag[2:])
    return libdirs


def request_urllib(url: str, filename: str, token=None):
    """Download a file from the net"""
    import urllib.request

    # optionally add an auth token to the request
    request = urllib.request.Request(url)
    if token is not None:
        request.add_header("Authorization", token)

    try:
        resp = urllib.request.urlopen(request)
    except urllib.request.HTTPError as e:
        return e.code

    if 200 <= resp.status < 300:
        with open(filename, "wb") as fp:
            fp.write(resp.read())
    return resp.status


def assets_most_recent_release(project: str) -> list[str]:
    """
    Return the asset urls attached to the most recent release of a github project.
    """
    url = f"https://api.github.com/repos/{project}/releases"
    # look for the github token in the environment
    # and use it for the request to avoid the github API rate limits
    token = os.environ.get("GITHUB_TOKEN", None)
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = str(Path(tmpdir) / "releases.json")
        status = None
        for _ in range(10):
            status = request_urllib(url, filename, token=token)
            if 200 <= status < 300:
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"Error downloading asset list from {url} "
                               f"... Response: {status}")

        with open(filename) as fp:
            ret = json.loads(fp.read())
        assets = [ret[i]["assets"] for i in range(len(ret))][0]
        return [asset["browser_download_url"] for asset in assets]


def url_most_recent_release(project: str) -> str:
    """
    Return the download url for the most recent release of a github project."""
    assets = assets_most_recent_release(project)
    if sys.platform == "linux":
        url = [asset for asset in assets if "-linux_x86_64" in asset]
    elif sys.platform == "darwin":
        # platform.machine() gives e.g., arm64
        # we want to match macosx_X_arm64, where X is the Version
        url = [
            asset for asset in assets
            if "-macosx_" in asset and f"_{platform.machine()}" in asset
        ]
    else:
        raise AssertionError("Can't download {project} from github"
                             " releases for unspported platform "
                             f"{sys.platform}.")
    if not url:
        raise RuntimeError(
            f"Could not find a version of project {project} to download. "
            "Check your platform is supported and if in doubt see the "
            "adcc installation instructions "
            "(https://adc-connect.org/latest/installation.html)."
        )
    assert len(url) == 1  # found more than 1 url for the current system??
    return url[0]


def install_libtensor(url: str, destination: str):
    """
    Download libtensor from `url` and install at `destination`, removing possibly
    existing files from a previous installation.
    """
    dest_folder = Path(destination)
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Downloading libtensorlight from {url} to {destination} ...")
        tmpdir = Path(tmpdir).resolve()
        archive_file = tmpdir / Path(url).name

        # download the archive
        status_code = request_urllib(url, str(archive_file))  # add token?
        if status_code < 200 or status_code >= 300:
            raise RuntimeError(
                "Could not download libtensorlight. Check for a network issue "
                "and if in doubt see the adcc installation instructions "
                "(https://adc-connect.org/latest/installation.html)."
            )
        # remove old installation files
        file_globs = [
            "include/libtensorlight/**/*.h",
            "include/libtensor/**/*.h",
            "include/libutil/**/*.h",
            "lib/libtensorlight.so",
            "lib/libtensorlight.so.*",
            "lib/libtensorlight.dylib",
            "lib/libtensorlight.*.dylib",
            "lib/pkgconfig/libtensorlight.pc",
        ]
        for fglob in file_globs:
            for file in dest_folder.rglob(fglob):
                log.info(f"Removing old libtensor file: {str(file)}")
                assert file.is_file()
                file.unlink()
        # Change to installation directory and unpack the downloaded archive
        olddir = Path.cwd()
        dest_folder.mkdir(parents=True, exist_ok=True)
        os.chdir(destination)
        subprocess.run(["tar", "xf", archive_file], check=True)
        os.chdir(olddir)


def libadcc_sources(target: str) -> list[str]:
    sourcefiles = set(glob.glob("libadcc/**/*.cc", recursive=True))
    unittests = glob.glob("libadcc/tests/*.cc", recursive=True)
    exportfiles = glob.glob("libadcc/pyiface/*.cc")
    if target == "extension":
        return sorted(sourcefiles.difference(unittests))
    elif target == "cpptest":
        return sorted(sourcefiles.difference(exportfiles))
    else:
        raise ValueError(f"Unknown target: {target}")


def update_flags_from_config(config_file: Path, *flags: dict):
    """
    Reads and executes the content of the config file and updates
    the prodided flags accordingly
    """
    log.info(f"Reading siteconfig file: {str(config_file)}")
    assert config_file.is_file()
    # merge the flags into a single dict: can't have the same keys!
    combined_flags = {}
    for flag_subset in flags:
        assert not flag_subset.keys() & combined_flags.keys()
        combined_flags.update(flag_subset)
    # read and execute the config and update the flags
    exec(open(config_file, "r").read(), combined_flags)
    # split the flags up in the original dicts only keeping
    # known keys that existed in the original dicts
    for key, val in combined_flags.items():
        for flag_subset in flags:
            if key in flag_subset:
                flag_subset[key] = val


@functools.lru_cache()
def libadcc_extension():
    # flags that are passed to the compiler
    build_flags: dict[str, list[str]] = {
        "libraries": [],
        "library_dirs": [],
        "include_dirs": [],
        "extra_link_args": [],
        "extra_compile_args": ["-Wall", "-Wextra", "-Werror", "-O3"],
        "runtime_library_dirs": [],
        "extra_objects": [],
        "define_macros": [],
    }
    # flags relevant for configuration of the extension (e.g. setting
    # the compiler flags and finding libtensor)
    config_flags = {
        "coverage": False,  # generate a test coverage report
        "search_system": True,  # search system for libtensorlight using pkg-config
        # install libtensor to folder if missing in the system
        "libtensor_autoinstall": None,
        # try to download libtensor from the url for the autoinstall
        "libtensor_url": None,
    }
    # only checked when we search the system or download the library!
    libtensorlight_min_version = "3.0.1"
    # The config files to check
    if "ADCC_CONFIG" in os.environ:
        if not Path(os.environ["ADCC_CONFIG"]).is_file():
            raise FileNotFoundError(os.environ["ADCC_CONFIG"])
        config_files: list[Path] = [Path(os.environ["ADCC_CONFIG"])]
    else:
        config_files: list[Path] = [
            Path("siteconfig.py"), Path.home() / ".adcc" / "siteconfig.py"
        ]

    # set specific compile args depending on the operating system
    if sys.platform == "darwin":
        build_flags["extra_compile_args"] += [
            "-Wno-unused-command-line-argument",
            "-Wno-undefined-var-template", "-Wno-bitwise-instead-of-logical"
        ]
        build_flags["extra_compile_args"].extend(["-arch", platform.machine()])
        build_flags["extra_link_args"].extend(["-arch", platform.machine()])
    elif sys.platform.startswith("linux"):
        # otherwise fails with -O3 on gcc>=12
        build_flags["extra_compile_args"] += ["-Wno-array-bounds"]

    # folder to look for libtensor install: for linux and mac-os and if we are not
    # running in a conda environment (avoid creating a folder in home in conda)
    platform_autoinstall = (
        sys.platform.startswith("linux") or sys.platform.startswith("darwin")
    )
    if platform_autoinstall and not is_conda_build():
        # Keep '~' here and expand later to not change the behaviour!
        config_flags["libtensor_autoinstall"] = "~/.local"

    # User-provided config: modify build and config flags
    for siteconfig in config_files:
        if not siteconfig.is_file():
            continue
        # in-place update the flags from the config file
        update_flags_from_config(siteconfig, build_flags, config_flags)
        break  # only read a single config file!

    # Keep track whether libtensor has been found
    found_libtensor = "tensorlight" in build_flags["libraries"]

    if not found_libtensor:
        # Once we enter this if statement we require pkg-config to find
        # libtensorlight and generate the compiler and linker args!
        cflags, libs = None, None
        # first try to search the system with pkg-config
        if config_flags["search_system"]:
            log.info("Searching OS for libtensorlight using pkg-config")
            cflags, libs = search_with_pkg_config(
                "libtensorlight", libtensorlight_min_version
            )
        # if this was not successful we try to download libensorlight
        if (cflags is None or libs is None) and \
                config_flags["libtensor_autoinstall"]:
            # - get the url from where to download
            if config_flags["libtensor_url"]:
                url = config_flags["libtensor_url"]
            else:
                url = url_most_recent_release("adc-connect/libtensor")
            # download libtensor from the url and install it in the given folder
            install_dir = os.path.expanduser(config_flags["libtensor_autoinstall"])
            install_libtensor(url, install_dir)
            # add the installdir to the PKG_CONFIG_PATH so we can find the package
            append_to_pkg_config_path(
                str(Path(install_dir) / "lib" / "pkgconfig")
            )
            cflags, libs = search_with_pkg_config(
                "libtensorlight", libtensorlight_min_version
            )
            assert cflags is not None and libs is not None

        if cflags is not None and libs is not None:
            found_libtensor = True
            build_flags["extra_compile_args"].extend(cflags)
            build_flags["extra_link_args"].extend(libs)
            log.info(f"Using libtensorlight libraries: {libs}.")
            if sys.platform == "darwin":
                build_flags["extra_link_args"].append("-Wl,-rpath,@loader_path")
                for path in extract_library_dirs(libs):
                    build_flags["extra_link_args"].append(f"-Wl,-rpath,{path}")
            else:
                build_flags["runtime_library_dirs"].extend(
                    extract_library_dirs(libs)
                )

    if not found_libtensor:
        raise RuntimeError("Did not find the libtensorlight library.")

    ext = Pybind11Extension("libadcc", libadcc_sources("extension"),
                            language="c++", cxx_std=14, **build_flags)
    if config_flags["coverage"]:
        ext.extra_compile_args += [
            "--coverage", "-O0", "-g", "-fprofile-update=atomic"
        ]
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


def read_readme():
    with open("README.md") as fp:
        return "".join([line for line in fp if not line.startswith("<img")])


if not os.path.isfile("adcc/__init__.py"):
    raise RuntimeError("Running setup.py is only supported "
                       "from top level of repository as './setup.py <command>'")

setup(
    # content of readme can't be modified from within pyproject.toml
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    #
    ext_modules=[libadcc_extension()],
    #
    cmdclass={
        "build_ext": build_ext, "build_docs": BuildDocs, "cpptest": CppTest
    },
)
