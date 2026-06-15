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
#include "Tensor.hh"

namespace libadcc {

/** Fill the passed vector of doubles blocks with doubles guesses using
 * the delta-Fock matrices df1 and df2, which are the two delta-Fock matrices
 * involved in the doubles block.
 *
 * guesses_d    Vectors of guesses, all elements are assumed to be initialised to zero
 *              and the symmetry is assumed to be properly set up.
 * mospaces     Mospaces object
 * df02         Delta-Fock between spaces 0 and 2 of the ADC matrix
 * df13         Delta-Fock between spaces 1 and 3 of the ADC matrix
 * spin_change_twice   Twice the value of the spin change to enforce in an excitation.
 * degeneracy_tolerance  Tolerance for two entries of the diagonal to be considered
 *                       degenerate, i.e. identical.
 *
 * \returns  The number of guess vectors which have been properly initialised
 *           (the others are invalid and should be discarded).
 */
size_t fill_pp_doubles_guesses(std::vector<std::shared_ptr<Tensor>> guesses_d,
                               std::shared_ptr<const MoSpaces> mospaces,
                               std::shared_ptr<Tensor> df1, std::shared_ptr<Tensor> df2,
                               int spin_change_twice, scalar_type degeneracy_tolerance);

}  // namespace libadcc
