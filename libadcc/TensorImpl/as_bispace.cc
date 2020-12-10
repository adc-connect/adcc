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

#include "as_bispace.hh"
#include "../exceptions.hh"

namespace libadcc {
namespace lt = libtensor;

namespace {

/** Struct to logically combine a libtensor::bispace<N>
 *  and a string array of size N, which contains the ids
 *  of the bispace<1> objects which make up the bispace<N>.
 */
template <size_t N>
struct IdedBispace : public libtensor::bispace<N> {
  typedef libtensor::bispace<N> base_type;

  /** The elementary spaces */
  std::array<std::string, N> spaces;

  /** The spaces as a concatenated string */
  std::string spaces_string() const {
    std::string s;
    for (auto& id : spaces) s.append(id);
    return s;
  }

  IdedBispace(libtensor::bispace<N> inner, std::array<std::string, N> spaces)
        : base_type(std::move(inner)), spaces(std::move(spaces)) {}
};

/** Make a libtensor bispace from some block starting indices and a space string
 * describing the space for which the bispace object should be made */
IdedBispace<1> make_bispace(const std::string& space, std::vector<size_t> block_starts,
                            size_t length) {
  // Split at the subspace splits, noting the difference in convention:
  // libtensor does not keep the first zero around
  //
  // Get the subspace splits
  const std::vector<size_t>& starts = block_starts;
  if (starts[0] != 0) {
    throw runtime_error(
          "Internal error: Expected starts vector to start with a 0 element.");
  }

  lt::bispace<1> bispace(length);
  for (size_t i = 1; i < starts.size(); ++i) {
    bispace.split(starts[i]);
  }
  return IdedBispace<1>(bispace, {{space}});
}

/** Form higher spaces by taking the tensor products of 2 same-sized bispaces
 *
 * This function takes the product between two IdedBispace objects `x` and `y` by
 * appending the string identifiers and taking the tensor product between the
 * bispace<N> objects.
 */
template <size_t N>
IdedBispace<2 * N> bispace_product(const IdedBispace<N>& x, const IdedBispace<N>& y) {
  // Build new spaces array
  std::array<std::string, 2 * N> ret_spaces;
  std::copy(std::begin(x.spaces), std::end(x.spaces), std::begin(ret_spaces));
  std::copy(std::begin(y.spaces), std::end(y.spaces), std::begin(ret_spaces) + N);

  if (x.spaces == y.spaces) {  // Symmetric tensor product of spaces
    return IdedBispace<2 * N>(libtensor::bispace<2 * N>(x & y), std::move(ret_spaces));
  } else {  // No symmetry
    return IdedBispace<2 * N>(libtensor::bispace<2 * N>(x | y), std::move(ret_spaces));
  }
}

template <size_t N>
IdedBispace<3 * N> bispace_product(const IdedBispace<N>& x, const IdedBispace<N>& y,
                                   const IdedBispace<N>& z) {
  // Build new spaces array
  std::array<std::string, 3 * N> ret_spaces;
  std::copy(std::begin(x.spaces), std::end(x.spaces), std::begin(ret_spaces));
  std::copy(std::begin(y.spaces), std::end(y.spaces), std::begin(ret_spaces) + N);
  std::copy(std::begin(z.spaces), std::end(z.spaces), std::begin(ret_spaces) + 2 * N);

  if (x.spaces == y.spaces && y.spaces == z.spaces) {  // All symmetric
    return IdedBispace<3 * N>(libtensor::bispace<3 * N>(x & y & z),
                              std::move(ret_spaces));
  } else if (x.spaces == y.spaces) {  // First product symmetric, second not symmetric
    return IdedBispace<3 * N>(libtensor::bispace<3 * N>((x & y) | z),
                              std::move(ret_spaces));
  } else if (y.spaces == z.spaces) {  // First not symmetric, second symmetric
    return IdedBispace<3 * N>(libtensor::bispace<3 * N>(x | (y & z)),
                              std::move(ret_spaces));
  } else if (x.spaces == z.spaces) {  // First and third symmetric
    return IdedBispace<3 * N>(libtensor::bispace<3 * N>(x | y | z, (x & z) | y),
                              std::move(ret_spaces));
  } else {  // No symmetry
    return IdedBispace<3 * N>(libtensor::bispace<3 * N>(x | y | z),
                              std::move(ret_spaces));
  }
}

template <size_t N>
lt::bispace<N> make_sym_bispace_aux(std::vector<IdedBispace<1>>) {
  throw not_implemented_error("Not implemented for N != 1, 2, 3, 4");
}

template <>
lt::bispace<1> make_sym_bispace_aux(std::vector<IdedBispace<1>> spaces1d) {
  return spaces1d[0];
}

template <>
lt::bispace<2> make_sym_bispace_aux(std::vector<IdedBispace<1>> spaces1d) {
  return bispace_product(spaces1d[0], spaces1d[1]);
}

template <>
lt::bispace<3> make_sym_bispace_aux(std::vector<IdedBispace<1>> spaces1d) {
  return bispace_product(spaces1d[0], spaces1d[1], spaces1d[2]);
}

template <>
lt::bispace<4> make_sym_bispace_aux(std::vector<IdedBispace<1>> spaces1d) {
  auto pair01 = bispace_product(spaces1d[0], spaces1d[1]);
  auto pair23 = bispace_product(spaces1d[2], spaces1d[3]);
  return bispace_product(pair01, pair23);
}

}  // namespace

template <size_t N>
lt::bispace<N> as_bispace(const std::vector<AxisInfo>& axes) {
  if (axes.size() != N) {
    throw invalid_argument("N (== " + std::to_string(N) +
                           ") and number of passed AxisInfos (== " +
                           std::to_string(axes.size()) + " does not agree.");
  }
  std::vector<IdedBispace<1>> spaces1d;
  for (size_t i = 0; i < N; ++i) {
    spaces1d.push_back(make_bispace(axes[i].label, axes[i].block_starts, axes[i].size()));
  }
  return make_sym_bispace_aux<N>(spaces1d);
}

//
// Explicit instantiation
//

#define INSTANTIATE(DIM) \
  template lt::bispace<DIM> as_bispace(const std::vector<AxisInfo>& axes);

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)

#undef INSTANTIATE

}  // namespace libadcc
