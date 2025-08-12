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

#include "shape_to_string.hh"
#include <sstream>

namespace libadcc {

std::string shape_to_string(const std::vector<size_t>& shape) {
  std::stringstream ss;
  bool first = true;

  ss << "(";
  for (auto& s : shape) {
    ss << (first ? "" : ",") << s;
    first = false;
  }
  ss << ")";
  return ss.str();
}

}  // namespace libadcc
