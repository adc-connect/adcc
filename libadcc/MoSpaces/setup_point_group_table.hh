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

#pragma once

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/symmetry/product_table_container.h>
#pragma GCC visibility pop

namespace libadcc {
/**
 *  \addtogroup TensorLibtensor
 */
///@{

/** Setup point group symmetry table inside libtensor and return the irrep mapping as
 * libtensor needs it. */
std::map<size_t, std::string> setup_point_group_table(
      libtensor::product_table_container& ptc, const std::string& point_group);

///@}
}  // namespace libadcc
