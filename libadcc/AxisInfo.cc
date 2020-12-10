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

#include "AxisInfo.hh"

namespace libadcc {

bool operator==(const AxisInfo& lhs, const AxisInfo& rhs) {
  if (lhs.label != rhs.label) return false;
  if (lhs.n_orbs_alpha != rhs.n_orbs_alpha) return false;
  if (lhs.n_orbs_beta != rhs.n_orbs_beta) return false;
  if (lhs.block_starts != rhs.block_starts) return false;
  return true;
}

}  // namespace libadcc
