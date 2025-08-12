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
#include "../TensorImpl.hh"

namespace libadcc {

template <size_t N>
std::shared_ptr<Tensor> wrap_libtensor(
      std::shared_ptr<const AdcMemory> adcmem_ptr, std::vector<AxisInfo> axes,
      std::shared_ptr<libtensor::btensor<N, scalar_type>> libtensor_ptr) {
  return std::make_shared<TensorImpl<N>>(adcmem_ptr, axes, libtensor_ptr);
}

}  // namespace libadcc
