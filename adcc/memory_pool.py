#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import glob
import atexit
import shutil
import libadcc
import tempfile


class MemoryPool(libadcc.AdcMemory):
    def initialise(self, scratch_directory="/tmp", max_block_size=16,
                   allocator="standard"):
        """Initialise the adcc memory management.

        Parameters
        ----------
        scratch_directory : str, optional
            Directory for storing temporary pagefiles. This should be a fast
            storage location as tensor data will be mapped to asynchronously
            to this directory for the "libxc" allocator.

        max_block_size : int, optional
            The maximal size a tensor block may have along each axis.

        allocator : str, optional
            The allocator to be used. Valid values are "libxm" or "standard"
            (libstc++ allocator).
        """
        pagefile_directory = tempfile.mkdtemp(prefix="adcc_", dir=scratch_directory)
        super().initialise(pagefile_directory, max_block_size, allocator)
        atexit.register(MemoryPool.cleanup, self)

    def cleanup(self):
        if os.path.isdir(self.pagefile_directory):
            shutil.rmtree(self.pagefile_directory)

    @property
    def scratch_directory(self):
        return os.path.dirname(self.pagefile_directory)

    @property
    def page_files(self):
        """The list of all page files."""
        globs = ["pagefile.*", "xmpagefile"]
        globs = [os.path.join(self.pagefile_directory, pat) for pat in globs]
        return [c for pat in globs for c in glob.glob(pat)]

    @property
    def total_size_page_files(self):
        """The total size of all page files."""
        return sum(os.path.getsize(f) for f in self.page_files)


# The actual memory object to use
memory_pool = MemoryPool()
