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

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/core/allocator.h>
#include <libtensor/core/batching_policy_base.h>
#include <libtensor/expr/btensor/eval_btensor.h>
#include <libtensor/metadata.h>
#pragma GCC visibility pop

namespace libadcc {
namespace {
bool libtensor_has_libxm() {
  const std::vector<std::string>& libtensor_features = libtensor::metadata::features();
  const auto it =
        std::find(libtensor_features.begin(), libtensor_features.end(), "libxm");
  return it != libtensor_features.end();
}
}  // namespace

AdcMemory::AdcMemory()
      : m_max_block_size(0),
        m_allocator("none"),
        m_initialise_called(false),
        m_pagefile_directory{""} {
  initialise("", 16, "standard");

  // Make sure we can call initialise once more
  m_initialise_called = false;
}

void AdcMemory::initialise(std::string pagefile_directory, size_t max_block_size,
                           std::string allocator) {
  if (m_initialise_called) {
    throw invalid_argument("Cannot initialise AdcMemory object twice.");
  }

  m_initialise_called  = true;
  m_pagefile_directory = pagefile_directory;
  m_max_block_size     = max_block_size;

  if (allocator == "standard") {
  } else if (allocator == "libxm" && libtensor_has_libxm()) {
  } else {
    throw invalid_argument("A memory allocator named '" + allocator +
                           "' is not known to adcc. Perhaps it is not compiled in.");
  }

  shutdown();  // Shutdown previously initialised allocator (if any)
  libtensor::allocator<double>::init(allocator, pagefile_directory.c_str());
  m_allocator = allocator;

  // Set initial batch size ... value empirically determined to be reasonable
  set_contraction_batch_size(21870);

  // Enable libxm contraction backend if libxm allocator is used
  libtensor::expr::eval_btensor<double>::use_libxm(allocator == "libxm");
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
