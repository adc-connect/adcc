//
// Copyright (C) 2018 by the adcc authors
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

#include "random_tensor.hh"
#include "../exceptions.hh"
#include "wrap_libtensor.hh"
#include <random>

namespace libadcc {
namespace tests {
namespace lt = libtensor;

template <size_t N>
std::shared_ptr<Tensor> random_tensor(std::shared_ptr<AdcMemory> adcmem_ptr,
                                      std::array<size_t, N> dimension,
                                      std::array<std::string, N> subspaces) {
  size_t size = 1;
  for (size_t i = 0; i < N; ++i) {
    size *= dimension[i];
  }
  std::vector<scalar_type> buffer(size);

  // Fill with random data
  std::random_device rd;
  std::default_random_engine dre(rd());
  std::uniform_real_distribution<double> uid(-1., 1.);
  std::generate(buffer.begin(), buffer.end(), [&]() { return uid(dre); });

  if (N == 1) {
    lt::bispace<1> bispace_0(dimension[0]);
    lt::bispace<1> bispace(bispace_0);
    std::vector<AxisInfo> axes{{subspaces[0], dimension[0]}};

    auto t_ptr = std::make_shared<lt::btensor<1>>(bispace);
    lt::btod_import_raw<1>(buffer.data(), bispace.get_bis().get_dims()).perform(*t_ptr);
    return wrap_libtensor(adcmem_ptr, axes, t_ptr);
  } else if (N == 2) {
    lt::bispace<1> bispace_0(dimension[0]);
    lt::bispace<1> bispace_1(dimension[1]);
    lt::bispace<2> bispace(bispace_0 | bispace_1);
    std::vector<AxisInfo> axes{
          {subspaces[0], dimension[0]},
          {subspaces[1], dimension[1]},
    };

    auto t_ptr = std::make_shared<lt::btensor<2>>(bispace);
    lt::btod_import_raw<2>(buffer.data(), bispace.get_bis().get_dims()).perform(*t_ptr);
    return wrap_libtensor(adcmem_ptr, axes, t_ptr);
  } else if (N == 3) {
    lt::bispace<1> bispace_0(dimension[0]);
    lt::bispace<1> bispace_1(dimension[1]);
    lt::bispace<1> bispace_2(dimension[2]);
    lt::bispace<3> bispace(bispace_0 | bispace_1 | bispace_2);
    std::vector<AxisInfo> axes{
          {subspaces[0], dimension[0]},
          {subspaces[1], dimension[1]},
          {subspaces[2], dimension[2]},
    };

    auto t_ptr = std::make_shared<lt::btensor<3>>(bispace);
    lt::btod_import_raw<3>(buffer.data(), bispace.get_bis().get_dims()).perform(*t_ptr);
    return wrap_libtensor(adcmem_ptr, axes, t_ptr);
  } else if (N == 4) {
    lt::bispace<1> bispace_0(dimension[0]);
    lt::bispace<1> bispace_1(dimension[1]);
    lt::bispace<1> bispace_2(dimension[2]);
    lt::bispace<1> bispace_3(dimension[3]);
    lt::bispace<4> bispace(bispace_0 | bispace_1 | bispace_2 | bispace_3);
    std::vector<AxisInfo> axes{
          {subspaces[0], dimension[0]},
          {subspaces[1], dimension[1]},
          {subspaces[2], dimension[2]},
          {subspaces[3], dimension[3]},
    };

    auto t_ptr = std::make_shared<lt::btensor<4>>(bispace);
    lt::btod_import_raw<4>(buffer.data(), bispace.get_bis().get_dims()).perform(*t_ptr);
    return wrap_libtensor(adcmem_ptr, axes, t_ptr);
  } else {
    throw not_implemented_error("random_tensor not implemented for dimensionality " +
                                std::to_string(N) + ".");
  }
}

//
// Explicit instantiation
//

#define INSTANTIATE(DIM)                                                          \
  template std::shared_ptr<Tensor> random_tensor(                                 \
        std::shared_ptr<AdcMemory> adcmem_ptr, std::array<size_t, DIM> dimension, \
        std::array<std::string, DIM> subspaces);

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)

#undef INSTANTIATE

}  // namespace tests
}  // namespace libadcc
