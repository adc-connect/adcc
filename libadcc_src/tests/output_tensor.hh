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
#include "../Tensor.hh"
#include <ostream>

namespace libadcc {

std::ostream& operator<<(std::ostream& o, const Tensor& t) {
  std::vector<double> exported;
  t.export_to(exported);
  o << "(";
  for (size_t i = 0; i < t.ndim(); ++i) {
    o << i << ", ";
  }
  o << ")\n";
  for (size_t i = 0; i < exported.size(); ++i) {
    o << exported[i] << " ";
    if (i % 3 == 2) o << "\n";
  }
  o << "\n";
  return o;
}

}  // namespace libadcc
