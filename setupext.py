#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import sys
import glob

from setuptools import Extension
from setuptools.distutils.errors import CompileError
from setuptools.command.build_ext import build_ext as BuildCommand


def is_conda_build():
    return (
        os.environ.get("CONDA_BUILD", None) == "1"
        or os.environ.get("CONDA_EXE", None)
    )


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
        except CompileError:
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


def get_extensions():
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
