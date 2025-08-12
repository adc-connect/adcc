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
#include "Symmetry.hh"

namespace libadcc {
/**
 *  \addtogroup Tensor
 */
///@{

/** Return the Symmetry object for the orbital energies tensor for the passed orbital
 *  subspace. */
std::shared_ptr<Symmetry> make_symmetry_orbital_energies(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space);

/** Return the Symmetry object for the orbital coefficient tensor for the passed orbital
 *  subspace.
 *
 *   \param  mospaces_ptr   MoSpaces pointer
 *   \param  space    Space to use (e.g. fb or o1b)
 *   \param  n_bas    Number of AO basis functions
 *   \param  blocks   Which blocks of the coefficients to return. Valid values are "ab" to
 *                    return a tensor for both alpha and beta block as a block-diagonal
 *                    tensor, "a" to only return a tensor for only alpha block
 *                    and "b" for only the beta block. The fourth option is "abstack",
 *                    which returns the symmetry for a alpha block stacked on top of
 *                    a beta block (This is the convention used by adcman).
 * */
std::shared_ptr<Symmetry> make_symmetry_orbital_coefficients(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space,
      size_t n_bas, const std::string& blocks = "ab");

/** Return the Symmetry object for the (antisymmetrised) electron-repulsion integral
 * tensor in the physicists' indexing convention for the passed orbital subspace. */
std::shared_ptr<Symmetry> make_symmetry_eri(std::shared_ptr<const MoSpaces> mospaces_ptr,
                                            const std::string& space);

/** Return the Symmetry object for the (non-antisymmetrised) electron-repulsion integral
 *  tensor in the physicists' indexing convention for the passed orbital subspace.
 *
 * \note This is *not* the symmetry of the eri objects in adcc. This is only needed for
 * some special routines importing the ERI tensor into adcc. Use it only if you know
 * you need it.
 **/
std::shared_ptr<Symmetry> make_symmetry_eri_symm(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space);

/** Return the Symmetry object for a MO spaces block of a one-particle operator
 *
 * \param mospaces_ptr     MoSpaces pointer
 * \param space            Space to use (e.g. o1o1 or o1v1)
 * \param symmetric        Is the tensor symmetric (only in effect if both space
 *                         axes identical). false disables a setup of permutational
 *                         symmetry.
 * \param cartesian_transformation
 *  The cartesian function accordung to which the operator transforms.
 *  Valid values are:
 *     "1"                   Totally symmetric (default)
 *     "x", "y", "z"         Coordinate axis
 *     "xx", "xy", "yz" ...  Products of two coordinate axis
 *     "Rx", "Ry", "Rz"      Rotations about the coordinate axis
 *
 *     Note: This can be used for the Fock matrix with operator_class = "1"
 */
std::shared_ptr<Symmetry> make_symmetry_operator(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space,
      bool symmetric, const std::string& cartesian_transformation);

/** Return the symmetry object for an operator in the AO basis. The object will
 *  represent a block-diagonal matrix of the form
 *      ( M 0 )
 *      ( 0 M ).
 *  where M is an n_bas x n_bas block and is indentical in upper-left
 *  and lower-right.
 *
 * \param mospaces_ptr     MoSpaces pointer
 * \param n_bas            Number of AO basis functions
 * \param symmetric        Is the tensor symmetric (only in effect if both space
 *                         axes identical). false disables a setup of permutational
 *                         symmetry.
 */
std::shared_ptr<Symmetry> make_symmetry_operator_basis(
      std::shared_ptr<const MoSpaces> mospaces_ptr, size_t n_bas, bool symmetric);

///@}
}  // namespace libadcc
