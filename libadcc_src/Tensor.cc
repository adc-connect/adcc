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

#include "Tensor.hh"

namespace libadcc {

Tensor::Tensor(std::shared_ptr<const AdcMemory> adcmem_ptr, std::vector<AxisInfo> axes)
      : m_size(1), m_shape(axes.size(), 0), m_axes{axes}, m_adcmem_ptr{adcmem_ptr} {
  for (size_t i = 0; i < axes.size(); ++i) {
    m_shape[i] = axes[i].size();
    m_size *= axes[i].size();
  }
}

std::vector<std::string> Tensor::subspaces() const {
  std::vector<std::string> ret;
  for (const auto& ax : axes()) {
    ret.push_back(ax.label);
  }
  return ret;
}

std::string Tensor::space() const {
  std::string ret = "";
  for (const auto& ax : axes()) ret.append(ax.label);
  return ret;
}

void Tensor::fill(scalar_type value) {
  const std::string alphabet = "abcdefgh";
  if (ndim() > alphabet.size()) {
    throw not_implemented_error(
          "zeros_like and empty_like only implemented up to tensor dimensionality 8.");
  }
  std::string mask = alphabet.substr(0, ndim());
  set_mask(mask, value);
}

std::shared_ptr<Tensor> Tensor::zeros_like() const {
  auto res = empty_like();
  res->fill(0.0);
  return res;
}

std::shared_ptr<Tensor> Tensor::ones_like() const {
  auto res = empty_like();
  res->fill(1.0);
  return res;
}

std::shared_ptr<Tensor> make_tensor_zero(std::shared_ptr<Symmetry> symmetry) {
  auto res = make_tensor(symmetry);
  res->fill(0.0);
  return res;
}

}  // namespace libadcc
