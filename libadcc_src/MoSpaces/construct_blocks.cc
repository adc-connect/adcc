//
// Copyright (C) 2019 by the adcc authors
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

#include "construct_blocks.hh"
#include "../exceptions.hh"

namespace libadcc {

std::vector<size_t> construct_blocks(std::vector<size_t> block_starts_crude,
                                     size_t length, size_t max_block_size) {
  std::vector<size_t> ret;

  // Lambda to insert extra block starts into ret. It gets the last position of a block
  // start and the next planned one and checks whether this is so large such that extra
  // starts need to be inserted. If that is the case these are inserted to ret.
  auto insert_extra_starts = [](std::vector<size_t>& ret, size_t last_start,
                                size_t next_start, size_t max_block_size) {
    const size_t len = next_start - last_start;
    if (len > max_block_size) {
      // Split into this number of blocks:
      const size_t n_blocks = (len + max_block_size - 1) / max_block_size;

      // The first few blocks might need to be a little larger to
      // make sure to completely tile the length len.
      // The block size of the small blocks
      const size_t block_size = len / n_blocks;

      // The remainder gives the number of "large blocks" we need.
      const size_t n_large_blocks = len % n_blocks;

      // The current position of block starts
      size_t pos = last_start;
      for (size_t ib = 0; ib < n_large_blocks; ++ib) {
        if (ib != 0) ret.push_back(pos);
        pos += block_size + 1;
      }
      for (size_t ib = n_large_blocks; ib < n_blocks; ++ib) {
        if (ib != 0) ret.push_back(pos);
        pos += block_size;
      }
      if (pos != next_start) {
        throw runtime_error("Internal error: Block tiling failed.");
      }
    }
    return next_start;
  };

  // Make sure at least a block start at 0 is in the block_starts_crude
  if (block_starts_crude.empty()) {
    block_starts_crude.push_back(0);
  }
  size_t last_start = block_starts_crude.front();
  ret.push_back(last_start);
  for (auto it = block_starts_crude.begin() + 1; it != block_starts_crude.end(); ++it) {
    last_start = insert_extra_starts(ret, last_start, *it, max_block_size);
    ret.push_back(*it);
  }
  insert_extra_starts(ret, last_start, length, max_block_size);
  return ret;
}

}  // namespace libadcc
