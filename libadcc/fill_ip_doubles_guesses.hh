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

// TODO modify docstring

/** Fill the passed vector of doubles blocks with doubles guesses using
 * the O and V matrices.
 * 
 *
 * guesses_d             Vectors of guesses, all elements are assumed to be initialised to zero
 *                       and the symmetry is assumed to be properly set up.
 * mospaces              Mospaces object
 * d_o                   Matrix to construct guesses from (occ.)
 * d_v                   Matrix to construct guesses from (virt.)
 * a_spin                If alpha ionization (false: beta)
 * restricted            Is this a restricted calculation
 * spin_change_twice     Twice the value of the spin change to enforce in an excitation.
 * degeneracy_tolerance  Tolerance for two entries of the diagonal to be considered
 *                       degenerate, i.e. identical.
 *
 * \returns  The number of guess vectors which have been properly initialised
 *           (the others are invalid and should be discarded).
 */
size_t fill_ip_doubles_guesses(std::vector<std::shared_ptr<Tensor>> guesses_d,
                               std::shared_ptr<const MoSpaces> mospaces,
                               std::shared_ptr<Tensor> d_o, 
                               std::shared_ptr<Tensor> d_v,
                               bool a_spin, bool restricted, int spin_change_twice, 
                               scalar_type degeneracy_tolerance);

}  // namespace libadcc