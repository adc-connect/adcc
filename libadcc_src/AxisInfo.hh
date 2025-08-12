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
#include <string>
#include <vector>

namespace libadcc {

/** Info data structure for tensor axes */
struct AxisInfo {
  std::string label;    //!< The label for this axis
  size_t n_orbs_alpha;  //!< The number of (alpha) orbitals
  size_t n_orbs_beta;   //!< The number of beta orbitals (or zero for axes without spin)
  std::vector<size_t> block_starts;  //!< The split indices when a new block starts

  AxisInfo(std::string label_, size_t n_orbs_alpha_, size_t n_orbs_beta_ = 0,
           std::vector<size_t> block_starts_ = {0})
        : label(label_),
          n_orbs_alpha(n_orbs_alpha_),
          n_orbs_beta(n_orbs_beta_),
          block_starts(block_starts_) {}

  /** Does the axis have a spin-splitting, or can no spin be identified */
  bool has_spin() const { return n_orbs_beta > 0; }

  /** Size of the axis, i.e. sum of number of alpha and beta orbitals */
  size_t size() const { return n_orbs_alpha + n_orbs_beta; }
};

bool operator==(const AxisInfo& lhs, const AxisInfo& rhs);

inline bool operator!=(const AxisInfo& lhs, const AxisInfo& rhs) { return !(lhs == rhs); }

}  // namespace libadcc
