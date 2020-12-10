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

#include "AdcMemory.hh"
#include "exceptions.hh"
#include <algorithm>
#include <libtensor/metadata.h>

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/core/allocator.h>
#include <libtensor/core/batching_policy_base.h>
#pragma GCC visibility pop

namespace libadcc {

AdcMemory::AdcMemory()
      : m_allocator("none"),
        m_initialise_called(false),
        m_max_memory(0),
        m_pagefile_directory{""},
        m_tbs_param(0) {
  // Just some value: >4GB is quite usual today and should get the setup right.
  // ... and anyway this limit is not really a hard limit for the std::allocator
  // anyway, since it plainly allocates memory on the system until we run out.
  const size_t max_memory = 4ul * 1024ul * 1024ul * 1024ul;

  // A tbs_param of 16 should be a good default.
  const size_t tbs_param = 16;

  initialise("", max_memory, tbs_param, "standard");

  // Make sure we can call initialise once more
  m_initialise_called = false;
}

void AdcMemory::initialise(std::string pagefile_directory, size_t max_memory,
                           size_t tbs_param, std::string allocator) {
  if (max_memory == 0) {
    throw invalid_argument("A max_memory value of 0 is not valid.");
  }
  if (tbs_param == 0) {
    throw invalid_argument("A tbs_param value of 0 is not valid.");
  }
  if (m_initialise_called) {
    throw invalid_argument("Cannot initialise AdcMemory object twice.");
  }

  m_initialise_called  = true;
  m_max_memory         = max_memory;
  m_pagefile_directory = pagefile_directory;
  m_tbs_param          = tbs_param;

  //
  // Determine initialisation parameters
  //
  // Our principle unit of memory is the employed scalar type,
  // so our memory calculations will be based on that.
  const size_t memunit = sizeof(scalar_type);

  // Ideally our largest tensor block should be large enough to fit a few of the
  // block size parameters. Our rationale here is that we have typically
  // no more than tensors of rank 6
  const size_t tbs3               = tbs_param * tbs_param * tbs_param;
  const size_t largest_block_size = tbs3 * tbs3;

  if (largest_block_size * memunit > max_memory) {
    throw invalid_argument("At least " + std::to_string(largest_block_size * memunit) +
                           " bytes of memory need to be requested.");
  }

  const bool has_libxm = [] {
    const std::vector<std::string>& libtensor_features = libtensor::metadata::features();
    const auto it =
          std::find(libtensor_features.begin(), libtensor_features.end(), "libxm");
    return it != libtensor_features.end();
  }();

  if (allocator == "default") {
    // Preference is 1. libxm 2. standard
    allocator = has_libxm ? "libxm" : "standard";
  }

  if (allocator == "standard") {
  } else if (allocator == "libxm" && has_libxm) {
  } else {
    throw invalid_argument("A memory allocator named '" + allocator +
                           "' is not known to adcc. Perhaps it is not compiled in.");
  }

  shutdown();  // Shutdown previously initialised allocator (if any)
  libtensor::allocator<double>::init(
        allocator,
        tbs_param,                     // Exponential base of data block size
        tbs_param * memunit,           // Smallest data block size in bytes
        largest_block_size * memunit,  // Largest data block size in bytes
        max_memory,                    // Memory limit in bytes
        pagefile_directory.c_str()     // Prefix to page file path.
  );
  m_allocator = allocator;

  // Set initial batch size
  set_contraction_batch_size(max_memory / tbs3 / tbs_param / 3);
}

size_t AdcMemory::contraction_batch_size() const {
  return libtensor::batching_policy_base::get_batch_size();
}

void AdcMemory::set_contraction_batch_size(size_t bsize) {
  libtensor::batching_policy_base::set_batch_size(bsize);
}

void AdcMemory::shutdown() {
  if (m_allocator != "none") {
    libtensor::allocator<double>::shutdown();
    m_allocator = "none";
  }
}

}  // namespace libadcc
