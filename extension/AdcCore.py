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
import os
import sys
import json

from os.path import join


class AdcCore:
    def __init__(self):
        this_dir = os.path.dirname(__file__)
        top_dir = os.path.abspath(join(this_dir, ".."))
        self.source_dir = join(top_dir, "adccore")
        self.install_dir = join(this_dir, "adccore")
        self.library_dir = join(top_dir, "adcc", "lib")
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
    def has_source(self):
        """Can adccore be build from source"""
        return os.path.isfile(join(self.source_dir, "build_adccore.py"))

    def build(self):
        """Build adccore from source. Only valid if has_source is true"""
        if not self.has_source:
            return RuntimeError("Cannot build adccore from source, "
                                "since source not available.")
        abspath = os.path.abspath(self.source_dir)
        if abspath not in sys.path:
            sys.path.insert(0, abspath)

        import build_adccore

        build_dir = join(self.source_dir, "build")
        build_adccore.build_install(build_dir, self.install_dir)

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

    def download(self, version):
        """Download a particular version of adccore from the internet"""
        raise NotImplementedError("Downloading binaries not yet implemented.")
        # TODO Idea is to delete the current files in this folder
        #      and download the tarball of the new adccore and unpack it
        pass

    def obtain(self, version):
        """Obtain the library in some way."""
        if self.has_source:
            self.build()
        else:
            self.download(version)

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
