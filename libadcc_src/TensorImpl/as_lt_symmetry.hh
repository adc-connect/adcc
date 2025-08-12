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
#include "../Symmetry.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/core/scalar_transf_double.h>  // Note: This header is needed here
#include <libtensor/core/symmetry.h>
#pragma GCC visibility pop

namespace libadcc {
/**
 *  \addtogroup TensorLibtensor
 */
///@{

/** Translate a adcc::Symmetry object into a shared pointer of the libtensor symmetry
 * object */
template <size_t N>
std::shared_ptr<libtensor::symmetry<N, scalar_type>> as_lt_symmetry(const Symmetry& sym);

///@}
}  // namespace libadcc
