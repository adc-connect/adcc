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
#include "AdcMemory.hh"
#include "HartreeFockSolution_i.hh"
#include <algorithm>
#include <map>
#include <memory>
#include <vector>

namespace adcc {
/**
 *  \addtogroup ReferenceObjects
 */
///@{

/** Information about the molecular orbitals spaces and subspaces,
 *  i.e. the occupied versus virtual orbitals, active versus
 *  frozen orbitals, core-occupied versus valence-occupied
 *  orbitals.
 */
struct MoSpaces {
  /** The number of orbitals inside a particular orbital space */
  size_t n_orbs(const std::string& space = "f") const {
    return n_orbs_alpha(space) + n_orbs_beta(space);
  }

  /** The number of alpha orbitals inside a particular orbital space */
  size_t n_orbs_alpha(const std::string& space = "f") const {
    return m_n_orbs_alpha.at(space);
  }

  /** The number of beta orbitals inside a particular orbital space */
  size_t n_orbs_beta(const std::string& space = "f") const {
    return m_n_orbs_beta.at(space);
  }

  /** Obtain a pretty name for the provided subspace */
  std::string subspace_name(const std::string& space = "f") const {
    return m_map_subspace_name.at(space);
  }

  //
  // Global stuff
  //
  /** The name of the point group for which the data in this class has been set up. */
  std::string point_group;

  /** The list of all irreducible representations in this point group.
   *  The first entry will always be the totally symmetric irreducible representation. */
  std::vector<std::string> irreps;

  /** Are the orbitals resulting from a restricted calculation,
   *  such that alpha and beta electron share the same spatial part */
  bool restricted;

  /** The list of occupied subspaces known to this object */
  std::vector<std::string> subspaces_occupied;

  /** List of virtual subspaces known to this object */
  std::vector<std::string> subspaces_virtual;

  /** List of all subspaces known to this object */
  std::vector<std::string> subspaces;

  /** Does this object have a core-occupied space
   *  (i.e. is ready for core-valence separation) */
  bool has_core_occupied_space() const { return has_subspace("o2"); }

  /** Does this object have a particular subspace */
  bool has_subspace(const std::string& space) const {
    return std::find(subspaces_occupied.begin(), subspaces_occupied.end(), space) !=
           subspaces_occupied.end();
  }

  /** Return the totally symmetric irreducible representation. */
  std::string irrep_totsym() const { return irreps[0]; }

  //
  // Mappings per subspace
  //
  /** Contains for each orbital space (e.g. f, o1) a mapping from the indices used inside
   *  the respective space to the molecular orbital index convention used in the host
   *  program, i.e. to the ordering in which the molecular orbitals have been exposed
   *  in the HartreeFockSolution_i or HfProvider object.
   */
  std::map<std::string, std::vector<size_t>> map_index_hf_provider;

  /** Contains for each orbital space the indices at which a new block starts. Thus this
   *  list contains at least on index, namely 0.
   */
  std::map<std::string, std::vector<size_t>> map_block_start;

  /** Contains for each orbital space the mapping from each *block* used inside the space
   *  to the irreducible representation it correspond to.
   */
  std::map<std::string, std::vector<std::string>> map_block_irrep;

  /** Contains for each orbital space the mapping from each *block* used inside the space
   *  to the spin it correspond to ('a' is alpha and 'b' is beta)
   */
  std::map<std::string, std::vector<char>> map_block_spin;

  //
  // Construction / export
  //
  /** Construct an MoSpaces object from a HartreeFockSolution_i, a pointer to
   *  an AdcMemory object.
   *
   *  \param hf                HartreeFockSolution_i object to use
   *  \param adcmem_ptr        ADC memory keep-alive object to be used in all Tensors
   *                           constructed using this MoSpaces object.
   *  \param core_orbitals     List of orbitals indices (in the full fock space, original
   *                           ordering of the hf object), which defines the orbitals to
   *                           be put into the core space, if any. The same number
   *                           of alpha and beta orbitals should be selected. These will
   *                           be forcibly occupied.
   *  \param frozen_core_orbitals
   *                           List of orbital indices, which define the frozen core,
   *                           i.e. those occupied orbitals, which do not take part in
   *                           the ADC calculation. The same number of alpha and beta
   *                           orbitals has to be selected.
   *  \param frozen_virtuals   List of orbital indices, which the frozen virtuals,
   *                           i.e. those virtual orbitals, which do not take part
   *                           in the ADC calculation. The same number of alpha and beta
   *                           orbitals has to be selected.
   */
  MoSpaces(const HartreeFockSolution_i& hf, std::shared_ptr<const AdcMemory> adcmem_ptr,
           std::vector<size_t> core_orbitals, std::vector<size_t> frozen_core_orbitals,
           std::vector<size_t> frozen_virtuals);

  //
  // TODO Helper functions to get ranges for each space that satisfy certain
  //      criteria, e.g. alpha spin, this and that irrep
  //

  /** Return a pointer to the memory keep-alive object */
  std::shared_ptr<const AdcMemory> adcmem_ptr() const { return m_adcmem_ptr; }

  /** Return the libtensor index used for the irrep string
   *
   * \note this function is for internal usage only.
   */
  size_t libtensor_irrep_index(std::string irrep) const;

 private:
  // Pointer to the adc memory object to keep memory alive.
  std::shared_ptr<const AdcMemory> m_adcmem_ptr;

  /** The mapping between an irrep index and the irrep as a string as used by
     libtensor */
  std::map<size_t, std::string> m_libtensor_irrep_labels;

  /** Cache for the number alpha orbitals in a particular orbital subspace. */
  std::map<std::string, size_t> m_n_orbs_alpha;

  /** Cache for the number beta orbitals in a particular orbital subspace. */
  std::map<std::string, size_t> m_n_orbs_beta;

  /** Nice names for subspaces */
  std::map<std::string, std::string> m_map_subspace_name;
};

///@}
}  // namespace adcc

