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
#include "../AxisInfo.hh"
#include <array>

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/expr/bispace/bispace.h>
#pragma GCC visibility pop

namespace libadcc {
/**
 *  \addtogroup TensorLibtensor
 */
///@{

/** Form a bispace from a list of AxisInfo representing the axes of a tensor */
template <size_t N>
libtensor::bispace<N> as_bispace(const std::vector<AxisInfo>& axes);

///@}
}  // namespace libadcc
