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

#pragma once
#include "../Tensor.hh"
#include "../TensorImpl.hh"

namespace libadcc {
namespace tests {

template <size_t N>
std::shared_ptr<Tensor> random_tensor(std::shared_ptr<AdcMemory> adcmem_ptr,
                                      std::array<size_t, N> dimension,
                                      std::array<std::string, N> subspaces);

}  // namespace tests
}  // namespace libadcc
