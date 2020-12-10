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
#include <exception>
#include <stdexcept>

namespace libadcc {
/**
 *  \defgroup Utilities Utilities and misc. stuff
 */
///@{

using std::invalid_argument;
using std::out_of_range;
using std::runtime_error;

struct dimension_mismatch : public invalid_argument {
  using invalid_argument::invalid_argument;
};

struct not_implemented_error : public runtime_error {
  using runtime_error::runtime_error;
};

///@}
}  // namespace libadcc
