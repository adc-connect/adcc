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

#include "get_block_starts.hh"
#include "../exceptions.hh"

namespace libadcc {
namespace lt = libtensor;

template <size_t N>
std::vector<std::vector<size_t>> get_block_starts(
      const libtensor::block_index_space<N>& bis) {
  std::vector<std::vector<size_t>> ret(N);

  for (size_t i = 0; i < N; ++i) {
    const size_t dim_type      = bis.get_type(i);
    const lt::split_points& sp = bis.get_splits(dim_type);
    ret[i].push_back(0);
    for (size_t isp = 0; isp < sp.get_num_points(); ++isp) {
      ret[i].push_back(sp[isp]);
    }
  }
  return ret;
}

//
// Explicit instantiation
//

#define INSTANTIATE(DIM)                                      \
  template std::vector<std::vector<size_t>> get_block_starts( \
        const libtensor::block_index_space<DIM>& bis);

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)

#undef INSTANTIATE

}  // namespace libadcc
