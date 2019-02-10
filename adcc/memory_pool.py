#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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

import atexit
import glob
import libadcc
import os
import shutil
import tempfile


class StdAllocatorWarning(Warning):
    pass


class MemoryPool(libadcc.AdcMemory):
    def initialise(self, max_memory, tensor_block_size=16,
                   pagefile_directory=None):
        """
        Initialise the adcc virtual memory management. Calling this function
        will direclty allocate all storage of max_memory.

        @param   max_memory   Estimate for the maximally employed memory
        @param tensor_block_size   This parameter roughly has the meaning
                                   of how many indices are handled together
                                   on operations. A good value is 16 for most
                                   nowaday CPU cachelines.
        @param pagefile_prefix     Directory prefix for storing temporary
                                   cache files.
        """
        if not pagefile_directory:
            pagefile_directory = tempfile.mkdtemp(prefix="adcc_", dir="/tmp")
        super().initialise(pagefile_directory, max_memory, tensor_block_size)
        atexit.register(MemoryPool.cleanup, self)

    def cleanup(self):
        if os.path.isdir(self.pagefile_directory):
            shutil.rmtree(self.pagefile_directory)

    @property
    def is_initialised(self):
        """
        Is the adcc virtual memory management initialised? If the returns
        False, the std::allocator of C++ is used.
        """
        # The vmm is initialised exactly when do not
        # use the std::allocator.
        return not self.use_std_allocator

    @property
    def page_files(self):
        """
        The list of all page files.
        """
        return glob.glob(os.path.join(self.pagefile_directory, "pagefile.*"))

    @property
    def total_size_page_files(self):
        """
        The total size of all page files
        """
        return sum(os.path.getsize(f) for f in self.page_files)


# The actual memory object to use
memory_pool = MemoryPool()