#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

"""Setup for adcc"""
import os
import sys
import glob
import json
import setuptools

from os.path import join

from setuptools import Extension, find_packages, setup
from setuptools.command.test import test as TestCommand
from setuptools.command.build_ext import build_ext as BuildCommand

# Version of the python bindings and adcc python package.
__version__ = '0.11.0'


#
# Compile and install adccore library
#
def trigger_adccore_build():
    """
    Trigger a build of the adccore library, if it exists in source form.
    """
    if os.path.isfile("adccore/build_adccore.py"):
        abspath = os.path.abspath("adccore")
        if abspath not in sys.path:
            sys.path.insert(0, abspath)

        import build_adccore

        build_dir = "adccore/build"
        install_dir = "extension/adccore"
        build_adccore.build_install(build_dir, install_dir)


#
# Find AdcCore
#
class AdcCore:
    def __init__(self):
        this_dir = os.path.dirname(__file__)
        self.library_dir = join(this_dir, "adcc", "lib")

        base_path = join(this_dir, "extension", "adccore")
        self.include_dir = join(base_path, "include")
        self.config_path = join(base_path, "adccore_config.json")

    @property
    def is_config_file_present(self):
        """
        Is the config file present on disk
        """
        return os.path.isfile(self.config_path)

    @property
    def config(self):
        if not self.is_config_file_present:
            raise RuntimeError(
                "Did not find adccore_config.json file in the directory tree."
                + " Did you download or install adccore properly? See the adcc "
                + "documentation for help."
            )
        else:
            with open(self.config_path, "r") as fp:
                return json.load(fp)

    def __getattr__(self, key):
        try:
            return self.config[key]
        except KeyError:
            raise AttributeError

    @property
    def libraries_full(self):
        """Return the full path to all libraries"""
        def get_full(lib):
            for prefix in ["lib", ""]:
                for ext in [".so", ".dylib"]:
                    name = join(self.library_dir, prefix + lib + ext)
                    if os.path.isfile(name):
                        return name
            raise RuntimeError("Could not find full path of library '"
                               + lib + '"')
        return [get_full(lib) for lib in self.libraries]

    @property
    def required_dynamic_libraries(self):
        """
        Return a list of all dynamically loaded shared libraries
        the OS needs to provide to function with this package.
        """
        import subprocess

        if sys.platform != "linux":
            raise OSError("required_dynamic_libraries is not "
                          "supported on this OS.")
        needed = []
        for lib in self.libraries_full:
            try:
                ret = subprocess.check_output(["objdump", "-p", lib],
                                              universal_newlines=True)
            except subprocess.CalledProcessError as cpe:
                if cpe.returncode == 127:
                    raise OSError("Could not find objdump binary")
                else:
                    raise RuntimeError("Could not determine required "
                                       "dynamic libraries")

            for line in ret.split("\n"):
                line = line.split()
                if line and line[0] == "NEEDED":
                    needed.append(line[1])
        return needed

    @property
    def feature_macros(self):
        return [("ADCC_WITH_" + feat.upper(), 1) for feat in self.features]


#
# Poor-man's ld-like resolver for shared libraries
#
class LinkerDynamic:
    def __init__(self):
        # Initialise by environment and add default search locations:
        self.library_paths = os.environ.get("LD_LIBRARY_PATH", "").split(":")
        self.library_paths += ["/usr/lib", "/lib"]

        for conf in ["/etc/ld.so.conf"] + glob.glob("/etc/ld.so.conf.d/*.conf"):
            with open(conf, "r") as fp:
                for line in fp:
                    line = line.strip()
                    if line and not line.startswith("#") \
                       and os.path.isdir(line):
                        self.library_paths.append(line)

    def find(self, library):
        for path in self.library_paths:
            name = join(path, library)
            if os.path.isfile(name):
                return name
        return None


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
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile

    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.

    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(BuildCommand):
    """A custom build extension for adding compiler-specific options."""
    def build_extensions(self):
        trigger_adccore_build()

        opts = []
        if sys.platform == "darwin":
            potential_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.7']
            opts.extend([opt for opt in potential_opts
                         if has_flag(self.compiler, opt)])
        if self.compiler.compiler_type == 'unix':
            opts.append(cpp_flag(self.compiler))
            potential_opts = [
                "-fvisibility=hidden", "-Werror", "-Wall", "-Wextra",
                "-pedantic", "-Wnon-virtual-dtor", "-Woverloaded-virtual",
                "-Wcast-align", "-Wconversion", "-Wsign-conversion",
                "-Wmisleading-indentation", "-Wduplicated-cond",
                "-Wduplicated-branches", "-Wlogical-op",
                "-Wdouble-promotion", "-Wformat=2",
                "-Wno-error=deprecated-declarations",
            ]
            opts.extend([opt for opt in potential_opts
                         if has_flag(self.compiler, opt)])

        for ext in self.extensions:
            ext.extra_compile_args = opts
        BuildCommand.build_extensions(self)


#
# Pytest integration
#
class PyTest(TestCommand):
    user_options = [
        ('mode=', 'm', 'Mode for the testsuite (fast or full)'),
        ('skip-update', 's', 'Skip updating testdata'),
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

adccore = AdcCore()
if not adccore.is_config_file_present:
    # Trigger a build of adccore if the source code can be found
    trigger_adccore_build()
if adccore.version != __version__:
    # Try to see if a simple adccore build solves this issue
    trigger_adccore_build()

    if adccore.version != __version__:
        raise RuntimeError(
            "Version mismatch between adcc (== {}) and adccore (== {})"
            "".format(__version__, adccore.version)
        )

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
        'libadcc', glob.glob("extension/*.cc"),
        define_macros=adccore.feature_macros,
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
        language='c++',
    ),
]

long_description = """
adcc is a python-based framework for performing quantum-chemical simulations
based upon the algebraic-diagrammatic construction (ADC) approach.

As of now PP-ADC and CVS-PP-ADC methods are available to compute excited
states on top of an MP2 ground state. The underlying Hartree-Fock reference
is not computed inside adcc, much rather external packages should be used
for this purpose. Interfaces to seamlessly interact with pyscf, VeloxChem
or molsturm are available, but other SCF codes or even statically computed
data can be easily used as well.

Notice, that only the adcc python and C++ source code are released under the
terms of the GNU Lesser General Public License v3 (LGPLv3) license. This
license does not apply to the libadccore.so binary file contained inside
the directory '/adcc/lib/' of the distributed tarball. For further details
see the file LICENSE_adccore.
""".strip()  # TODO extend
setup(
    name='adcc',
    description='adcc:  Seamlessly connect your host program to ADC',
    long_description=long_description,
    #
    url='https://github.com/mfherbst/adcc',
    author='adcc developers',
    author_email='adcc+developers@michael-herbst.com',
    maintainer='Michael F. Herbst',
    maintainer_email='info@michael-herbst.com',
    license="LGPL v3",
    #
    version=__version__,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: '
        'GNU Lesser General Public License v3 (LGPLv3)',
        'License :: Free For Educational Use',
        'Intended Audience :: Science/Research',
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Education",
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: Unix',
    ],
    #
    packages=find_packages(exclude=["*.test*", "test"]),
    package_data={'adcc': ["lib/*.so", "lib/*.dylib",
                           "lib/libadccore_LICENSE",
                           "lib/libadccore_thirdparty/ctx/*"],
                  '': ["LICENSE*"]},
    ext_modules=ext_modules,
    zip_safe=False,
    #
    platforms=["Linux", "Mac OS-X", "Unix"],
    python_requires='>=3.5',
    install_requires=[
        'pybind11 (>= 2.2)',
        'numpy (>= 1.13)',  # Maybe even higher?
        'scipy',
    ],
    tests_require=["pytest", "h5py"],
    #
    cmdclass={'build_ext': BuildExt, "pytest": PyTest},
)
