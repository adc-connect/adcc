//
// Copyright (C) 2017 by the adcc authors
//
// This file is part of adcc.
//
// adcc is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// adcc is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with adcc. If not, see <http://www.gnu.org/licenses/>.
//

#include "adcc/config.hh"
#include <string>
#pragma once

namespace adcc {
/**
 *  \addtogroup Utilities
 */
///@{

/** Manages memory for adc calculations.
 *
 * This class has to remain in scope for as long as the allocated
 * memory is needed. Normally this is achieved by keeping some sort
 * of keep-alive shared pointer in the private context of all classes
 * which require access to this memory.
 * All changes to the environment are undone when the class goes out of scope.
 *
 * \note Undefined behaviour may result if this class is ever initialised twice.
 */
class AdcMemory {
 public:
  /** Setup the class and directly call initialise with the default allocator */
  AdcMemory(std::string pagefile_directory, size_t max_memory, size_t tbs_param = 16)
        : AdcMemory{} {
    initialise(pagefile_directory, max_memory, tbs_param);
  }

  /** Setup the AdcMemory class using the std::allocator for memory management */
  AdcMemory();

  /** Destroy the AdcMemory class and free all the allocated memory */
  ~AdcMemory() { shutdown(); }

  // Disallow copying of this class.
  AdcMemory& operator=(const AdcMemory&) = delete;
  AdcMemory(const AdcMemory&)            = delete;

  /** Return the allocator to which the class is initialised. */
  std::string allocator() const { return m_allocator; }

  /** Return the max_memory parameter value to which the class was initialised.
   *  \note This value is only a meaningful upper bound if
   *  allocator() != "standard" */
  size_t max_memory() const { return m_max_memory; }

  /** Return the pagefileprefix value
   *
   * \note This value is only meaningful if allocator() != "standard" */
  std::string pagefile_directory() const { return m_pagefile_directory; }

  /** Return the tbs_param value */
  size_t tbs_param() const { return m_tbs_param; }

  /** Get the contraction batch size, this is the number of
   *  tensor blocks, which are processed in a batch in a tensor contraction. */
  size_t contraction_batch_size() const;

  /** Set the contraction batch size */
  void set_contraction_batch_size(size_t bsize);

  /** Setup the environment for the memory management.
   *
   * \param pagefile_directory  File prefix for page files
   * \param max_memory          The maximal memory adc makes use of (in bytes).
   * \param tbs_param           The tensor block size parameter.
   *                            This parameter roughly has the meaning of how many indices
   *                            are handled together on operations. A good value seems to
   *                            be 16.
   * \param allocator   The allocator to be used. Valid values are "libxm", "libvmm",
   *                    "standard" and "default", where "default" uses a default
   *                    chosen from the first three.
   *
   * \note For some allocators \c pagefile_directory and \c max_memory are not supported
   *       and thus ignored.
   * \note "libxm" and "libvmm" are extra features, which are not available in a
   *       default setup.
   **/
  void initialise(std::string pagefile_directory, size_t max_memory,
                  size_t tbs_param = 16, std::string allocator = "default");

 protected:
  /** Shutdown the allocator, i.e. cleanup all memory currently held. */
  void shutdown();

 private:
  /** The allocator this object is currently initialised to. */
  std::string m_allocator;

  /** Has the initialise function been called by the user */
  bool m_initialise_called;

  /** Configured maximal memory in bytes */
  size_t m_max_memory;

  /** Configured file prefix for pagefiles */
  std::string m_pagefile_directory;

  /** Configured tensor block size parameter */
  size_t m_tbs_param;
};

///@}
}  // namespace adcc
