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
#include <cstddef>
#include <vector>

namespace libadcc {
namespace tests {

struct TensorTestData {
  /** The size of each dimension */
  static size_t N;

  /** A rank-4 tensor of test data */
  static std::vector<double> a;

  /** The symmetrisation of above tensor along axis 0 and 1 */
  static std::vector<double> a_sym_01;

  /** The anti-symmetrisation of above tensor along axis 0 and 1 */
  static std::vector<double> a_asym_01;

  /** The symmetrisation of above tensor by permuting simultaneously
   *  axes 0 and 1 and 2 / 3 */
  static std::vector<double> a_sym_01_23;

  /** The anti-symmetrisation of above tensor by permuting simultaneously
   *  axes 0 and 1 and 2 / 3 */
  static std::vector<double> a_asym_01_23;
};

}  // namespace tests
}  // namespace libadcc
