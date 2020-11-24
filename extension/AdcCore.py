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
import sys
import glob
import json
import tempfile
import subprocess
import distutils.util

from distutils import log
from os.path import join


def get_platform():
    """Return our platform name 'win32', 'linux_x86_64'"""
    # Copied from https://github.com/pypa/wheel/blob/master/wheel/pep425tags.py
    result = distutils.util.get_platform().replace('.', '_').replace('-', '_')
    if result == "linux_x86_64" and sys.maxsize == 2147483647:
        result = "linux_i686"
    return result


def has_mkl_numpy():
    """Has numpy been installed and linked against MKL"""
    try:
        from numpy.__config__ import get_info

        return any("mkl" in lib for lib in
                   get_info("blas_mkl").get("libraries", {}))
    except ImportError as e:
        if "mkl" in str(e):
            # numpy seems to be installed and linked against MKL,
            # but mkl was not found.
            raise ImportError(
                "Trying to import numpy for MKL check, but obtained an "
                "import error indicating a missing MKL dependency. "
                "Did you load the MKL modules properly?"
            ) from e

        # This indicates a missing numpy or a big error in numpy. It's best
        # to assume MKL is not there and (potentially) install the non-mkl
        # version from pypi
        return False


def request_urllib(url, filename):
    """Download a file from the net using requests, displaying
    a nice progress bar along the way"""
    import urllib.request

    print("Downloading {} ... this may take a while.".format(url))
    try:
        resp = urllib.request.urlopen(url)
    except urllib.request.HTTPError as e:
        return e.code

    if 200 <= resp.status < 300:
        with open(filename, 'wb') as fp:
            fp.write(resp.read())
    return resp.status


class AdcCore:
    def __init__(self):
        this_dir = os.path.dirname(__file__)
        self.top_dir = os.path.abspath(join(this_dir, ".."))
        self.source_dir = join(self.top_dir, "adccore")
        self.install_dir = join(this_dir, "adccore")
        self.library_dir = join(self.top_dir, "adcc", "lib")
        self.include_dir = join(self.install_dir, "include")
        self.config_path = join(self.install_dir, "adccore_config.json")

    @property
    def is_config_file_present(self):
        """
        Is the config file present on disk
        """
        return os.path.isfile(self.config_path)

    @property
    def is_documentation_present(self):
        """Is the doxygen documentation available on disk"""
        doc_dir = join(self.install_dir, "share/adccore/docs")
        return os.path.isfile(join(doc_dir, "xml", "index.xml"))

    @property
    def upstream(self):
        """Return upstream remote or None if unknown"""
        jsonfile = os.path.expanduser("~/.adccore.json")
        if not os.path.isfile(jsonfile):
            return None
        try:
            with open(jsonfile, "r") as fp:
                return json.load(fp).get("upstream", None)
        except json.JSONDecodeError:
            return None

    def checkout(self, version):
        """Checkout adccore source code in the given version if possible"""
        if not self.upstream:
            raise RuntimeError("Cannot checkout adccore, since upstream "
                               "not known.")
        subprocess.check_call(["git", "clone", self.upstream, self.source_dir])

        olddir = os.getcwd()
        os.chdir(self.source_dir)
        subprocess.check_call(["git", "checkout", "v" + version])
        subprocess.check_call("git submodule update --init --recursive".split())
        os.chdir(olddir)

        assert self.has_source

    @property
    def has_source(self):
        """Can adccore be build from source"""
        return os.path.isfile(join(self.source_dir, "build_adccore.py"))

    def build(self, features=None):
        """Build adccore from source. Only valid if has_source is true"""
        if not self.has_source:
            return RuntimeError("Cannot build adccore from source, "
                                "since source not available.")
        abspath = os.path.abspath(self.source_dir)
        if abspath not in sys.path:
            sys.path.insert(0, abspath)

        import build_adccore

        if features is None:
            if has_mkl_numpy():
                features = ["mkl"]
            else:
                features = []

        build_dir = join(self.source_dir, "build")
        build_adccore.build_install(build_dir, self.install_dir,
                                    features=features)

    def build_documentation(self):
        """Build adccore documentation. Only valid if has_source is true"""
        if not self.has_source:
            return RuntimeError("Cannot build adccore documentation, "
                                "since source not available.")
        abspath = os.path.abspath(self.source_dir)
        if abspath not in sys.path:
            sys.path.insert(0, abspath)

        import build_adccore

        doc_dir = join(self.install_dir, "share/adccore/docs")
        build_adccore.build_documentation(doc_dir, latex=False,
                                          html=False, xml=True)

    @property
    def file_globs(self):
        """
        Return the file globs to be applied relative to the top directory of the
        repository in order to obtain all files relevant for the binary
        distribution of adccore.
        """
        return [
            self.install_dir + "/adccore_config.json",
            self.install_dir + "/include/adcc/*.hh",
            self.install_dir + "/include/adcc/*/*.hh",
            self.library_dir + "/libadccore.so",
            self.library_dir + "/libadccore.*.dylib",
            self.library_dir + "/libadccore.dylib",
            self.library_dir + "/libstdc++.so.*",
            self.library_dir + "/libc++.so.*",
            self.library_dir + "/libadccore_LICENSE",
        ]

    def get_tarball_name(self, version=None, postfix=None):
        """
        Get the platform-dependent name of the adccore tarball
        of the specified version.
        """
        if version is None:
            version = self.version
        if postfix is None:
            postfix = ""
        return "adccore-{}{}-{}.tar.gz".format(version, postfix, get_platform())

    def download(self, version, postfix=None):
        """Download a particular version of adccore from the internet"""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_url = "https://get.adc-connect.org/adccore"
            fn = self.get_tarball_name(version, postfix)
            local = tmpdir + "/" + fn
            status_code = request_urllib(base_url + "/" + fn, local)
            if status_code < 200 or status_code >= 300:
                msg = ("Could not download adccore version {} for platform {} "
                       "from {}.".format(version, get_platform(), base_url))
                if 400 <= status_code < 500:
                    # Either an unsupported version or an error on our end
                    msg += (" This should not have happened and either this means"
                            " your platform / OS / architecture is unsupported or"
                            " that there is a bug in adcc. Please check the adcc "
                            " installation instructions"
                            " (https://adc-connect.org/latest/installation.html)"
                            " and if in doubt, please open an issue on github.")
                raise RuntimeError(msg)

            # Delete the old files
            for fglob in self.file_globs:
                for fn in glob.glob(fglob):
                    log.info("Removing old adccore file {}".format(fn))
                    os.remove(fn)

            # Change to installation directory
            olddir = os.getcwd()
            os.chdir(self.top_dir)
            subprocess.run(["tar", "xf", local], check=True)
            os.chdir(olddir)

    def obtain(self, version, postfix=None, allow_checkout=True):
        """Obtain the library in some way."""
        if self.has_source:
            self.build()
        elif allow_checkout and self.upstream:
            self.checkout(version)
            self.build()
        else:
            self.download(version, postfix)

    @property
    def agrees_with_os_platform(self):
        if sys.platform == "linux":
            return get_platform() == self.platform
        elif sys.platform == "darwin":
            os_version = tuple([int(x) for x in get_platform().split('_')[1:3]])

            if not self.platform.startswith("macosx"):
                return False

            core_platform_split = self.platform.split('_')[1:3]
            try:
                core_version = tuple([int(x) for x in core_platform_split])
            except ValueError:
                return False
            # MacOS is fine with things being compiled on earlier versions
            return core_version <= os_version
        else:
            raise OSError("Unsupported platform: {}".format(sys.platform))

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
