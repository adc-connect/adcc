//
// Copyright (C) 2020 by the adcc authors
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
#include "../config.hh"
#include <vector>

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/expr/btensor/btensor.h>
#pragma GCC visibility pop

namespace libadcc {
namespace lt = libtensor;

/** Extract the block starts from a libtensor block-index space */
template <size_t N>
std::vector<std::vector<size_t>> get_block_starts(
      const libtensor::block_index_space<N>& bis);

/** Parse the symmetry object contained in btensor get the block starts */
template <size_t N>
std::vector<std::vector<size_t>> get_block_starts(lt::btensor<N, scalar_type>& btensor) {
  lt::block_tensor_ctrl<N, scalar_type> ctrl_from(btensor);
  return get_block_starts(ctrl_from.req_const_symmetry().get_bis());
}

}  // namespace libadcc
