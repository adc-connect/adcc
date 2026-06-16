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
/**
 *  \addtogroup AdcGuess
 */
///@{

/**
 * Apply the spin projection required to make the doubles part of an amplitude
 * vector consist of components for a singlet state only.
 *
 * \note This function assumes a restricted reference with a closed-shell
 * ground state (e.g. RHF).
 *
 * @param tensor  The tensor to apply the symmetrisation to
 * @param block   The block of an amplitude this tensor represents
 * @param spin_kind   The kind of spin to enforce
 */
void amplitude_vector_enforce_spin_kind(std::shared_ptr<Tensor> tensor, std::string block,
                                        std::string spin_kind);
///@}
}  // namespace libadcc
