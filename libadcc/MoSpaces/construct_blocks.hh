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

#pragma once
#include <string>
#include <vector>

namespace libadcc {
/**
 *  \addtogroup ReferenceObjects
 */
///@{

/** Construct a list of Tensor blocks (by identifying their starting indices)
 *  from a list of indices that identify places where a block definitely
 *  has to start and a maximal block size.
 *
 *  The function checks the block sizes resulting from block_starts_crude
 *  and if they are beyond max_block_size tries to split them evenly.
 *
 *  \param block_starts_crude list of crude block starts
 *  \param length   Total number of indices
 *  \param max_block_size  Maximal size of a block
 */
std::vector<size_t> construct_blocks(std::vector<size_t> block_starts_crude,
                                     size_t length, size_t max_block_size);

///@}
}  // namespace libadcc
