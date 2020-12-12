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

#include "MoSpaces.hh"
#include "MoSpaces/construct_blocks.hh"
#include "MoSpaces/setup_point_group_table.hh"
#include "exceptions.hh"
#include <algorithm>
#include <cmath>

namespace libadcc {
namespace lt = libtensor;

namespace {

/**
 * Build a list of indices from 0 to n_orbs and "sort" it according
 * to the occupation and energy. An other words by using the indices
 * in the sorted vector for access the higher-occupied orbitals sort
 * first and the lower energy ones. Alpha and beta blocks are sorted
 * separately.
 */
std::vector<size_t> sorted_moidcs(const HartreeFockSolution_i& hf) {
  // Get occupation and orbital energies
  std::vector<double> occupation(hf.n_orbs());
  std::vector<scalar_type> orben_f(hf.n_orbs());
  hf.occupation_f(occupation.data(), occupation.size());
  hf.orben_f(orben_f.data(), orben_f.size());

  // Build sort indices
  std::vector<size_t> sortidx(hf.n_orbs());
  for (size_t i = 0; i < hf.n_orbs(); ++i) sortidx[i] = i;

  auto comparator = [&occupation, &orben_f](size_t l, size_t r) {
    if (occupation[l] != occupation[r]) {
      return occupation[l] > occupation[r];
    } else {
      return orben_f[l] < orben_f[r];
    }
  };

  const ssize_t n_orbs = static_cast<ssize_t>(hf.n_orbs_alpha());
  std::sort(sortidx.begin(), sortidx.begin() + n_orbs, comparator);
  std::sort(sortidx.begin() + n_orbs, sortidx.end(), comparator);
  return sortidx;
}

size_t index_in(const std::string& elem, const std::vector<std::string>& vector) {
  for (size_t i = 0; i < vector.size(); ++i) {
    if (vector[i] == elem) return i;
  }
  throw std::out_of_range("Element " + elem + " not found.");
}

}  // namespace

MoSpaces::MoSpaces(const HartreeFockSolution_i& hf,
                   std::shared_ptr<const AdcMemory> adcmem_ptr,
                   std::vector<size_t> core_orbitals,
                   std::vector<size_t> frozen_core_orbitals,
                   std::vector<size_t> frozen_virtuals)
      : point_group{},
        irreps{},
        restricted{hf.restricted()},
        subspaces_occupied{},
        subspaces_virtual{},
        subspaces{},
        map_index_hf_provider{},
        map_block_start{},
        map_block_irrep{},
        map_block_spin{},
        m_adcmem_ptr{adcmem_ptr},
        m_libtensor_irrep_labels{},
        m_n_orbs_alpha{},
        m_n_orbs_beta{},
        m_map_subspace_name{}

{
  //
  // Checks on HF data
  //

  // Count the number of alpha and beta electrons
  std::vector<double> occupation(hf.n_orbs());
  hf.occupation_f(occupation.data(), occupation.size());
  size_t n_alpha = 0;
  for (size_t i = 0; i < hf.n_orbs_alpha(); ++i) {
    if (std::fabs(occupation[i] - 1.0) < 1e-12) {
      n_alpha += 1;
    } else if (std::fabs(occupation[i]) > 1e-12) {
      throw invalid_argument("Occupation value " + std::to_string(occupation[i]) +
                             " for orbital " + std::to_string(i) +
                             " is invalid, since neither zero nor one.");
    } else {
      occupation[i] = 0;  // Normalise value to be exactly 0
    }
  }
  size_t n_beta = 0;
  for (size_t i = hf.n_orbs_alpha(); i < hf.n_orbs(); ++i) {
    if (std::fabs(occupation[i] - 1.0) < 1e-12) {
      n_beta += 1;
    } else if (std::fabs(occupation[i]) > 1e-12) {
      throw invalid_argument("Occupation value " + std::to_string(occupation[i]) +
                             " for orbital " + std::to_string(i) +
                             " is invalid, since neither zero nor one.");
    } else {
      occupation[i] = 0;  // Normalise value to be exactly 0
    }
  }

  if (n_alpha == 0 || n_beta == 0) {
    throw invalid_argument(
          "The HF data passed to adcc is not valid: Need at "
          "least 1 alpha and 1 beta electron");
  }

  //
  // Point group setup
  //
  point_group = "C1";  // TODO Later get this from the hf object or so.

  // Setup the required point group symmetry table inside libtensor
  m_libtensor_irrep_labels =
        setup_point_group_table(lt::product_table_container::get_instance(), point_group);
  irreps.resize(m_libtensor_irrep_labels.size());
  for (size_t i = 0; i < m_libtensor_irrep_labels.size(); ++i) {
    irreps[i] = m_libtensor_irrep_labels[i];
  }

  //
  // Setup sizes and spin / irrep / subspace data of the full Fock orbital space
  //
  m_n_orbs_alpha["f"] = hf.n_orbs_alpha();
  m_n_orbs_beta["f"]  = hf.n_orbs_alpha();
  const size_t n_f    = hf.n_orbs();

  if (point_group != "C1") {
    throw not_implemented_error("Only point_group == C1 is implemented");
  }
  // Setup mappings for information about each index (in the original ordering)
  std::vector<char> orig_spin(n_f);             // Index -> spin
  std::vector<std::string> orig_irrep(n_f);     // Index -> irrep
  std::vector<size_t> orig_index(n_f);          // Index -> original index (== identity)
  std::vector<std::string> orig_subspace(n_f);  // Index -> desired subspace
  for (size_t i = 0; i < n_f; ++i) {
    orig_irrep[i]    = irreps[0];
    orig_index[i]    = i;    // Will be changed later
    orig_subspace[i] = "f";  // Will be changed later
    if (i < hf.n_orbs_alpha()) {
      orig_spin[i] = 'a';
    } else {
      orig_spin[i] = 'b';
    }
  }

  // Distribute orbitals into subspaces.
  // For this note, that the subspaces are counted from the HOMO-LUMO gap toward the
  // core orbitals (for occupied) and outer virtual orbitals:
  m_map_subspace_name = {
        {"v2", "frozen virtual orbitals"},    //
        {"v1", "active virtual orbitals"},    //
        {"o1", "valence-occupied orbitals"},  //
        {"o2", "core-occupied orbitals"},     //
        {"o3", "frozen occupied orbitals"},   //
        // General names:
        {"f", "molecular orbitals"},
  };

  // Lambda to help distribute the orbitals. Does checks that an orbitals is not
  // distributed twice and that an equal number of alphas and betas sits in v2, o2 and o3.
  //
  // Alters m_n_orbs_alpha, m_n_orbs_beta and orig_subspace
  auto distribute_orbitals = [this, &orig_subspace, &orig_spin, &n_f](
                                   std::string subspace,
                                   const std::vector<size_t>& indices) {
    m_n_orbs_alpha[subspace] = 0;
    m_n_orbs_beta[subspace]  = 0;
    if (indices.empty()) return;

    for (size_t imo : indices) {
      if (imo >= n_f) {
        throw invalid_argument("Orbital index " + std::to_string(imo) +
                               " overshoots range of valid indices. Check the index list "
                               "used to specify the " +
                               subspace_name(subspace) + ".");
      }

      if (orig_subspace[imo] != "f") {
        throw invalid_argument(
              "In orbital to subspace distribution, the MO with index " +
              std::to_string(imo) + " is already distributed to subspace " +
              orig_subspace[imo] + " (" + subspace_name(orig_subspace[imo]) +
              "), but should be distributed again to " + subspace + " (" +
              subspace_name(subspace) +
              "). Please check that the orbital index lists core_orbitals, "
              "frozen_core_orbitals, frozen_virtuals contain no repeated indices and are "
              "disjoint from each other.");
      }

      orig_subspace[imo] = subspace;
      if (orig_spin[imo] == 'a') {
        m_n_orbs_alpha[subspace] += 1;
      } else {
        m_n_orbs_beta[subspace] += 1;
      }
    }

    if (m_n_orbs_alpha[subspace] != m_n_orbs_beta[subspace]) {
      throw invalid_argument(
            "An unequal number of alpha and beta orbitals has been selected for the "
            "orbital subspace " +
            subspace + " (" + subspace_name(subspace) +
            "). Please check the passed lists of orbital indices (core_orbitals, "
            "frozen_core_orbitals and frozen_virtuals) to make sure that in each of them "
            "the same number of alpha and beta orbitals is selected.");
    }
  };
  distribute_orbitals("o2", core_orbitals);
  distribute_orbitals("o3", frozen_core_orbitals);
  distribute_orbitals("v2", frozen_virtuals);

  // Check we have not already distributed too many electrons
  const size_t n_alpha_done = m_n_orbs_alpha["o2"] + m_n_orbs_alpha["o3"];
  const size_t n_beta_done  = m_n_orbs_alpha["o2"] + m_n_orbs_alpha["o3"];
  if (n_alpha <= n_alpha_done) {
    throw invalid_argument(
          "Too many orbitals selected to be " + subspace_name("o2") + " (space o2) or " +
          subspace_name("o3") + " (space o3). Their total number is " +
          std::to_string(n_alpha_done) + ", but the number of alpha electrons is only " +
          std::to_string(n_alpha) +
          ", leaving no active alpha electrons. Reduce the number of orbitals you have "
          "selected for frozen core or to be core-occupied (for CVS).");
  }
  if (n_beta <= n_beta_done) {
    throw invalid_argument(
          "Too many orbitals selected to be " + subspace_name("o2") + " (space o2) or " +
          subspace_name("o3") + " (space o3). Their total number is " +
          std::to_string(n_beta_done) + ", but the number of beta electrons is only " +
          std::to_string(n_beta) +
          ", leaving no active beta electrons. Reduce the number of orbitals you have "
          "selected for frozen core or to be core-occupied (for CVS).");
  }

  // Sort original MO indices by occupation and then by energy to make sure that
  // electrons are first put orbitals marked as occupied and ascending in energy.
  const std::vector<size_t> sortidcs = sorted_moidcs(hf);

  // Use the sorted indices in dist_imo to fill o1 and v1
  m_n_orbs_alpha["o1"] = 0;
  m_n_orbs_beta["o1"]  = 0;
  m_n_orbs_alpha["v1"] = 0;
  m_n_orbs_beta["v1"]  = 0;
  size_t n_alphas_todo = n_alpha - n_alpha_done;
  size_t n_betas_todo  = n_beta - n_beta_done;
  for (size_t imo : sortidcs) {
    if (orig_subspace[imo] != "f") continue;  // Already distributed

    if (orig_spin[imo] == 'a' && n_alphas_todo > 0) {
      orig_subspace[imo] = "o1";
      m_n_orbs_alpha["o1"] += 1;
      n_alphas_todo -= 1;
    } else if (orig_spin[imo] == 'b' && n_betas_todo > 0) {
      orig_subspace[imo] = "o1";
      m_n_orbs_beta["o1"] += 1;
      n_betas_todo -= 1;
    } else if (orig_spin[imo] == 'a' && n_alphas_todo == 0) {
      orig_subspace[imo] = "v1";
      m_n_orbs_alpha["v1"] += 1;
    } else if (orig_spin[imo] == 'b' && n_alphas_todo == 0) {
      orig_subspace[imo] = "v1";
      m_n_orbs_beta["v1"] += 1;
    }
  }

  if (n_alphas_todo > 0 || m_n_orbs_alpha["v1"] == 0) {
    throw invalid_argument(
          "Too many orbitals selected to be " + subspace_name("v2") +
          "(space v2), such that (a) not enough orbitals remain to accompany all "
          "electrons or (b) the active virtual space v1 is empty. Currently " +
          std::to_string(n_alphas_todo) +
          " electrons remain to be distributed. Reduce the number "
          "of orbitals you have selected for frozen virtual space.");
  }
  if (n_betas_todo > 0 || m_n_orbs_beta["v1"] == 0) {
    throw invalid_argument(
          "Too many orbitals selected to be " + subspace_name("v2") +
          "(space v2), such that (a) not enough orbitals remain to accompany all "
          "electrons or (b) the active virtual space v1 is empty. Currently " +
          std::to_string(n_betas_todo) +
          " electrons remain to be distributed. Reduce the number "
          "of orbitals you have selected for frozen virtual space.");
  }

  // Build subspaces_occupied, subspaces_virtual and subspaces
  subspaces_occupied.push_back("o1");
  if (m_n_orbs_alpha["o2"] > 0) subspaces_occupied.push_back("o2");
  if (m_n_orbs_alpha["o3"] > 0) subspaces_occupied.push_back("o3");
  subspaces_virtual.push_back("v1");
  if (m_n_orbs_alpha["v2"] > 0) subspaces_virtual.push_back("v2");
  for (auto& s : subspaces_occupied) subspaces.push_back(s);
  for (auto& s : subspaces_virtual) subspaces.push_back(s);

  //
  // Checks of the above orbital distribution for consistency
  // (we do not want to loose orbitals or electrons)
  //
  for (size_t imo = 0; imo < n_f; ++imo) {
    if (orig_subspace[imo] == "f") {
      throw runtime_error("Internal error: MO index " + std::to_string(imo) +
                          " not distributed.");
    }
  }
  if (m_n_orbs_alpha["o2"] + m_n_orbs_beta["o2"] != core_orbitals.size()) {
    throw runtime_error(
          "Internal error: Number of orbitals in o2 and number of desired core-occupied "
          "orbitals does not agree.");
  }
  if (m_n_orbs_alpha["o3"] + m_n_orbs_beta["o3"] != frozen_core_orbitals.size()) {
    throw runtime_error(
          "Internal error: Number of orbitals in o3 and number of desired frozen core "
          "orbitals does not agree.");
  }
  if (m_n_orbs_alpha["v2"] + m_n_orbs_beta["v2"] != frozen_virtuals.size()) {
    throw runtime_error(
          "Internal error: Number of orbitals in v2 and number of desired frozen "
          "virtuals does not agree.");
  }
  if (m_n_orbs_alpha["o1"] + m_n_orbs_alpha["o2"] + m_n_orbs_alpha["o3"] != n_alpha) {
    throw runtime_error(
          "Internal error: Number of alpha orbitals in o1, o2, o3 does not agree with "
          "number of alpha electrons.");
  }
  if (m_n_orbs_beta["o1"] + m_n_orbs_beta["o2"] + m_n_orbs_beta["o3"] != n_beta) {
    throw runtime_error(
          "Internal error: Number of beta orbitals in o1, o2, o3 does not agree with "
          "number of beta electrons.");
  }
  if (m_n_orbs_alpha["o1"] + m_n_orbs_alpha["o2"] + m_n_orbs_alpha["o3"] +
            m_n_orbs_alpha["v1"] + m_n_orbs_alpha["v2"] !=
      hf.n_orbs_alpha()) {
    throw runtime_error(
          "Internal error: Total number of alpha orbitals does not agree with number of "
          "alpha orbitals in hf object.");
  }
  if (m_n_orbs_beta["o1"] + m_n_orbs_beta["o2"] + m_n_orbs_beta["o3"] +
            m_n_orbs_beta["v1"] + m_n_orbs_beta["v2"] !=
      hf.n_orbs_alpha()) {
    throw runtime_error(
          "Internal error: Total number of beta orbitals in o1, o2, o3 does not agree "
          "with number of beta orbitals in hf object..");
  }

  //
  // Orbital reordering (construct orbital order used in adcc)
  //
  // Get the orbital energies:
  std::vector<scalar_type> orben_f(n_f);
  hf.orben_f(orben_f.data(), orben_f.size());

  // Sorting order for the subspaces:
  const std::vector<std::string> subspace_sort{"o3", "o2", "o1", "v1", "v2"};

  auto orbital_comparator = [&orig_irrep, &orig_spin, &orig_subspace, &orben_f,
                             &subspace_sort, this](size_t i, size_t j) {
    // The comparator to use for sorting the orbitals from the original ordering we
    // get from the host program via the hf object to the ordering we use internally
    // for adcc. The ordering sorts the orbitals according to the following hierachy:
    //    1. Spin: In the order 'a' < 'b'
    //    2. Orbital subspace: In the order o3 < o2 < o1 < v1 < v2
    //                         (i.e. position in subspace_sort)
    //    3. Irrep: By position it has in the irreps map
    //    4. Energy: By MO energy

    if (orig_spin[i] != orig_spin[j]) {
      return orig_spin[i] < orig_spin[j];  // 'a' sorts before 'b'
    }
    if (orig_subspace[i] != orig_subspace[j]) {
      return index_in(orig_subspace[i], subspace_sort) <
             index_in(orig_subspace[j], subspace_sort);
    }
    if (orig_irrep[i] != orig_irrep[j]) {
      return index_in(orig_irrep[i], irreps) < index_in(orig_irrep[j], irreps);
    }
    return orben_f[i] < orben_f[j];
  };

  // Do the actual sorting on the orig_index object. This sets up a mapping new order (in
  // adcc) -> old order (in hf), i.e. what is needed in the map_index_hf_provider object.
  std::sort(orig_index.begin(), orig_index.end(), orbital_comparator);
  map_index_hf_provider["f"] = orig_index;
  // TODO Unsure whether inverses of map_index_hf_provider should be constructed
  //      and stored as well

  // Bring forward the orig_spin, orig_subspace, orig_irrep objects to the new ordering
  std::vector<char> spin_of(n_f);
  std::vector<std::string> subspace_of(n_f);
  std::vector<std::string> irrep_of(n_f);
  for (size_t imo = 0; imo < n_f; ++imo) {
    spin_of[imo]     = orig_spin[orig_index[imo]];
    subspace_of[imo] = orig_subspace[orig_index[imo]];
    irrep_of[imo]    = orig_irrep[orig_index[imo]];
  }

  // TODO Unsure whether this should be stored or not
  // Setup a mapping subspace index -> index in the full space (both in adcc order)
  std::map<std::string, std::vector<size_t>> map_index_sub_to_full;
  for (const std::string& space : subspaces) {
    for (size_t imo = 0; imo < n_f; ++imo) {
      if (subspace_of[imo] == space) {
        map_index_sub_to_full[space].push_back(imo);
      }
    }

    if (map_index_sub_to_full[space].size() != n_orbs(space)) {
      throw runtime_error(
            "Internal error: map_index_sub_to_full size and n_orbs disagrees.");
    }
  }

  // Construct map_index_hf_provider for the subspaces
  for (const std::string& space : subspaces) {
    const std::vector<size_t>& fidx_of = map_index_sub_to_full[space];
    for (size_t iss = 0; iss < n_orbs(space); ++iss) {
      map_index_hf_provider[space].push_back(orig_index[fidx_of[iss]]);
    }
  }

  //
  // Space splittings (split spaces f, oN and vN into blocks)
  //
  // Lambda to construct the blocks for a particular space.
  // Modifies map_block_start, map_block_spin, map_block_irrep
  const size_t max_block_size = adcmem_ptr->max_block_size();
  auto construct_blocks_for   = [this, &map_index_sub_to_full, &irrep_of, &spin_of,
                               &subspace_of, &max_block_size](const std::string& space) {
    std::vector<size_t> fidx_of;
    if (space != "f") {
      fidx_of = map_index_sub_to_full[space];
    } else {
      fidx_of.resize(n_orbs("f"));
      for (size_t i = 0; i < n_orbs("f"); ++i) {
        fidx_of[i] = i;
      }
    }

    // The indices where a block definitely has to start.
    // This needs to be done whenever a new irrep, subspace or spin starts.
    std::vector<size_t> block_starts;

    std::string old_irrep    = "";
    char old_spin            = '?';
    std::string old_subspace = "";
    for (size_t iss = 0; iss < n_orbs(space); ++iss) {
      const size_t iss_full = fidx_of[iss];
      if (spin_of[iss_full] != old_spin || irrep_of[iss_full] != old_irrep ||
          subspace_of[iss_full] != old_subspace) {
        block_starts.push_back(iss);
        old_irrep    = irrep_of[iss_full];
        old_spin     = spin_of[iss_full];
        old_subspace = subspace_of[iss_full];
      }
    }

    // Refine the blocks making sure that they are not larger than max_block_size
    block_starts = construct_blocks(block_starts, n_orbs(space), max_block_size);
    map_block_start[space] = block_starts;

    for (size_t iblock = 0; iblock < block_starts.size(); ++iblock) {
      // The starting index of the block in the full Fock MO index space
      const size_t block_start_index = fidx_of[block_starts[iblock]];

      map_block_spin[space].push_back(spin_of[block_start_index]);
      map_block_irrep[space].push_back(irrep_of[block_start_index]);
    }
  };

  // Call the lambda to get the setup done.
  construct_blocks_for("f");
  for (const std::string& space : subspaces) {
    construct_blocks_for(space);
  }
}

size_t MoSpaces::libtensor_irrep_index(std::string irrep) const {
  for (auto& kv : m_libtensor_irrep_labels) {
    if (kv.second == irrep) return kv.first;
  }
  return static_cast<size_t>(-1);
}

}  // namespace libadcc
