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
#include "AxisInfo.hh"
#include "MoSpaces.hh"

namespace libadcc {
/**
 *  \addtogroup Tensor
 */
///@{

/** Container for Tensor symmetry */
class Symmetry {
 public:
  /** Create an empty symmetry object for a Tensor with the provided space string */
  Symmetry(std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space)
        : Symmetry(mospaces_ptr, space, {}) {}

  /** Create an empty symmetry object for a Tensor with the provided space string,
   * optionally using a map to supply the number of orbitals for some additional axes.
   *
   * For the additional axis the pair contains either two numbers (for the number
   * of alpha and beta orbitals in this axis) or only one number and a zero (for an axis,
   * which as only one spin kind, alpha or beta).
   **/
  Symmetry(std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space,
           std::map<std::string, std::pair<size_t, size_t>> extra_axes_orbs);

  //
  // Access to shape and sizes
  //
  /** Return the MoSpaces to which this Symmetry is set up */
  std::shared_ptr<const MoSpaces> mospaces_ptr() const { return m_mospaces_ptr; }

  /** Return the adc memory keepalive object */
  std::shared_ptr<const AdcMemory> adcmem_ptr() const {
    return m_mospaces_ptr->adcmem_ptr();
  }

  /** Return the info about each of the contained tensor axes in order */
  const std::vector<AxisInfo> axes() const { return m_axes; }

  /** The space used to initialise the object */
  std::string space() const;

  /** The space splitup into subspaces along each dimension */
  const std::vector<std::string>& subspaces() const { return m_subspaces; }

  /** Number of dimensions */
  size_t ndim() const { return m_subspaces.size(); }

  /** Shape of each dimension */
  std::vector<size_t> shape() const;

  // A nice descriptive string for the Symmetry object
  std::string describe() const;

  //
  // Modifying symmetry elements
  //
  /** Clear all symmetry elements stored in the datastructure */
  void clear();

  /** Is the datastructure empty, i.e. are there no symmetry elements stored in here */
  bool empty() const;

  //
  // Allowed irreps
  //
  /** Set the list of irreducible representations, for which the symmetry
   *  shall be non-zero. If this is *not* set all irreps will be allowed */
  void set_irreps_allowed(std::vector<std::string> irreps);

  /** Get the list of allowed irreducible representations.
   *  If this empty, all irreps are allowed. */
  std::vector<std::string> irreps_allowed() const { return m_irreps_allowed; }

  /** Is a restriction to irreps set */
  bool has_irreps_allowed() const { return !m_irreps_allowed.empty(); }

  /** Clear the restriction to particular irreps, i.e. allow all irreps */
  void clear_irreps_allowed() { m_irreps_allowed.clear(); }

  //
  // Permutational symmetry
  //
  /** Set a list of index permutations, which do not change the tensor.
   *  A minus may be used to indicate anti-symmetric
   *  permutations with respect to the first (reference) permutation.
   *
   * For example the vector {"ij", "ji"} defines a symmetric matrix
   * and {"ijkl", "-jikl", "-ijlk", "klij"} the symmetry of the ERI
   * tensor. Not all permutations need to be given to fully describe
   * the symmetry. The check for errors and conflicts is only
   * rudimentary at the moment, however.
   */
  void set_permutations(std::vector<std::string> permutations);

  /** Return the list of equivalent permutations */
  std::vector<std::string> permutations() const;

  /** Are there equivalent permutations set up */
  bool has_permutations() const { return !m_permutations.empty(); }

  /** Clear the equivalent permutations */
  void clear_permutations() {
    m_permutations.clear();
    m_permutations_factor.clear();
  }

  /** Return the list of equivalent permutations in a parsed internal format.
   *
   * \note Internal function. May go or change at any time
   */
  std::vector<std::pair<std::vector<size_t>, scalar_type>> permutations_parsed() const;

  //
  // Spin symmetry
  //
  /** Set lists of tuples of the form {"aaaa", "bbbb", -1.0}, i.e.
   *  two spin blocks followed by a factor. This maps the second onto the first
   *  with a factor of -1.0 between them.
   */
  void set_spin_block_maps(
        std::vector<std::tuple<std::string, std::string, double>> spin_maps);

  /** Return the list of equivalent spin blocks */
  std::vector<std::tuple<std::string, std::string, double>> spin_block_maps() const {
    return m_spin_block_maps;
  }

  /** Are there equivalent spin blocks set up? */
  bool has_spin_block_maps() const { return !m_spin_block_maps.empty(); }

  /** CLear the equivalent spin blocks */
  void clear_spin_block_maps() { m_spin_block_maps.clear(); }

  /** Mark spin-blocks as forbidden (i.e. enforce them to stay zero).
   *  Blocks are give as a list in the letters 'a' and 'b', e.g. {"aaaa", "abba"} */
  void set_spin_blocks_forbidden(std::vector<std::string> forbidden);

  /** Return the list of currently forbidden spin blocks */
  std::vector<std::string> spin_blocks_forbidden() const {
    return m_spin_blocks_forbidden;
  }

  /** Are there forbidden spin blocks set up? */
  bool has_spin_blocks_forbidden() const { return !m_spin_blocks_forbidden.empty(); }

  /** Clear the forbidden spin blocks, i.e. allow all blocks */
  void clear_spin_blocks_forbidden() { m_spin_blocks_forbidden.clear(); }

 private:
  void assert_valid_spinblock(const std::string& block) const;

  std::shared_ptr<const MoSpaces> m_mospaces_ptr;

  /** Parsed lot of subspaces (i.e. spaces string split into subspaces) */
  std::vector<std::string> m_subspaces;

  /** Stores the allowed irreducible representations */
  std::vector<std::string> m_irreps_allowed;

  /** Stores the reference permutations (permutations of 01234...) of the
   *  equivalent permutations */
  std::vector<std::vector<size_t>> m_permutations;

  /** Stores the factor, which relates the equivalent permuation in
   *  m_permutations to the reference permutation (01234...) */
  std::vector<scalar_type> m_permutations_factor;

  /** Stores the equivalent spin blocks */
  std::vector<std::tuple<std::string, std::string, double>> m_spin_block_maps;

  /** Stores the forbidden spin blocks */
  std::vector<std::string> m_spin_blocks_forbidden;

  /** Information about each of the tensor axes */
  std::vector<AxisInfo> m_axes;

  /** The alphabet for various nice printing functions and so on */
  std::string m_alphabet;
};

///@}
}  // namespace libadcc
