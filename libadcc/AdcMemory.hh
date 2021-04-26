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

#include "config.hh"
#include <string>
#pragma once

namespace libadcc {
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
  /** Setup the class and directly call initialise with the standard allocator */
  AdcMemory(std::string pagefile_directory, size_t max_block_size = 16) : AdcMemory{} {
    initialise(pagefile_directory, max_block_size);
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

  /** Return the pagefileprefix value
   *
   * \note This value is only meaningful if allocator() != "standard" */
  std::string pagefile_directory() const { return m_pagefile_directory; }

  /** Get the contraction batch size, this is the number of
   *  tensor elements, which are processed in a batch in a tensor contraction. */
  size_t contraction_batch_size() const;

  /** Set the contraction batch size */
  void set_contraction_batch_size(size_t bsize);

  /** Get the maximal block size a tensor may have along each axis */
  size_t max_block_size() const { return m_max_block_size; }

  /** Setup the environment for the memory management.
   *
   * \param pagefile_directory  File prefix for page files
   * \param max_block_size      Maximal block size a tensor may have along each axis.
   * \param allocator   The allocator to be used. Valid values are "libxm" or
   *                    "standard" where "standard" is just the standard C++ allocator.
   *
   * \note For some allocators \c pagefile_directory is not supported and thus ignored.
   * \note "libxm" is an extra features, which might not be available in a default setup.
   **/
  void initialise(std::string pagefile_directory, size_t max_block_size = 16,
                  std::string allocator = "standard");

 protected:
  /** Shutdown the allocator, i.e. cleanup all memory currently held. */
  void shutdown();

 private:
  /** Maximal size a tensor block may have along any axis. */
  size_t m_max_block_size;

  /** The allocator this object is currently initialised to. */
  std::string m_allocator;

  /** Has the initialise function been called by the user */
  bool m_initialise_called;

  /** Configured file prefix for pagefiles */
  std::string m_pagefile_directory;
};

///@}
}  // namespace libadcc
