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
#include "MoSpaces.hh"
#include "ReferenceState.hh"
#include "Tensor.hh"
#include <algorithm>
#include <map>
#include <memory>

namespace adcc {
/**
 *  \defgroup Properties ADC properties
 */
///@{

/** Container for a one-particle operator
 *  in MO representation. This container is used both for state density
 *  matrices as well as for transition-density matrices.
 *
 *  This object is a map from a block label (like o1o1) to the actual
 *  tensor object.
 */
struct OneParticleOperator {
  /** Construct an empty OneParticleOperator object. Will construct
   * an object with all blocks initialised as zero blocks.
   *
   * \param mospaces      MoSpaces object
   * \param is_symmetric  Is the operator symmetric
   * \param cartesian_transformation
   *                      The cartesian function according to which the operator
   *                      transforms. See make_symmetry_operator for details.
   */
  OneParticleOperator(std::shared_ptr<const MoSpaces> mospaces_ptr,
                      bool is_symmetric = true, std::string cartesian_transform = "1");

  /** Number of dimensions */
  size_t ndim() const { return 2; }

  /** Shape of each dimension */
  std::vector<size_t> shape() const;

  /** Number of elements */
  size_t size() const;

  /** Return a deep copy of this OneParticleOperator.
   *
   *  This can be used to conveniently add tensor corrections on top of existing
   *  OneParticleOperator objects.
   */
  std::shared_ptr<OneParticleOperator> copy() const;

  /** Get a particular block of the one-particle operator, throwing an invalid_argument
   *  in case the block is a known zero block */
  std::shared_ptr<Tensor> block(std::string block) const;

  /** Get a particular block of the one-particle operator. It allocates and stores
   *  a tensor representing a zero block on the fly if required. */
  std::shared_ptr<Tensor> operator[](std::string label);

  /** Set a particular block of the one-particle operator */
  void set_block(std::string block, std::shared_ptr<Tensor> value);

  /** Set the list of specified blocks to zero without storing an explicit zero
   *  tensor for them. If previously a tensor has been stored for them it
   *  will be purged. */
  void set_zero_block(std::vector<std::string> blocks) {
    for (auto& b : blocks) set_zero_block(b);
  }

  /** The version of \c set_zero_block for only a single block */
  void set_zero_block(std::string block);

  /** Return whether the specified block is a block flagged to be exactly zero
   *  using the set_zero_block function. */
  bool is_zero_block(std::string block) const {
    return m_map.find(block) != m_map.end() && m_map.at(block) == nullptr;
  }

  /** Check weather a particular block exists */
  bool has_block(std::string block) const { return m_map.find(block) != m_map.end(); }

  /** Return the list of all registered blocks */
  std::vector<std::string> blocks() const;

  /** Return the list of all registered not-zero blocks */
  std::vector<std::string> blocks_nonzero() const;

  /** Return the list of all orbital subspaces contained in the operator */
  std::vector<std::string> orbital_subspaces() const { return m_orbital_subspaces; }

  /** Is the operator represented by this object symmetric
   *  If true then certain density matrices (like v1o1) might be
   *  missing, since they are equivalent to already existing
   *  blocks (such as o1v1 in our example)
   * */
  bool is_symmetric() const { return m_is_symmetric; }

  /** The cartesian function according to which the operator
   *  transforms. See make_symmetry_operator for details.
   */
  std::string cartesian_transform() const { return m_cartesian_transform; }

  /** Transform this operator object to the AO basis, which was used
   *  to obtain the provided reference state.
   *
   *  \returns  A pair of matrices in the AO basis. The first
   *            is the matrix for alpha spin, the second for beta spin.
   **/
  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> to_ao_basis(
        std::shared_ptr<const ReferenceState> reference_state_ptr) const;

  /** Transform this operator object to the AO basis using the coefficient
   *  tensors provided for each molecular orbital subspace. The coefficient tensors
   *  should be provided via a mapping from the subspace identifier to the actual
   *  coefficient for each spin part. E.g. "o1_a" should map to the
   *  coefficient "c_o1b" which are the alpha coefficients for the first
   *  occupied orbital subspace.
   *
   *  \returns  A pair of matrices in the AO basis. The first
   *            is the matrix for alpha spin, the second for beta spin.
   **/
  std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> to_ao_basis(
        std::map<std::string, std::shared_ptr<Tensor>> coefficient_ptrs) const;

  /** Extract the operator to plain memory provided
   *  by the given pointer.
   *
   *  \note This will return a full, *dense* tensor.
   *        At least size elements of space are assumed at the provided memory location.
   *        The data is stored in row-major (C-like) format
   */
  void export_to(scalar_type* memptr, size_t size) const;

  std::shared_ptr<const MoSpaces> mospaces_ptr() const { return m_mospaces_ptr; }

 private:
  std::vector<std::string> parse_split_block(const std::string& block) const;

  /** Build all orbital subspaces to be contained in this OneParticleOperator */
  std::vector<std::string> build_orbital_subspaces(
        std::shared_ptr<const MoSpaces> mospaces_ptr) const;

  std::map<std::string, std::shared_ptr<Tensor>> m_map;
  bool m_is_symmetric;
  std::vector<std::string> m_orbital_subspaces;
  std::shared_ptr<const MoSpaces> m_mospaces_ptr;
  std::string m_cartesian_transform;
};

///@}
}  // namespace adcc
