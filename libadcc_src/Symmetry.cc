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

#include "Symmetry.hh"
#include "MoSpaces/construct_blocks.hh"
#include "exceptions.hh"
#include <sstream>

// TODO Temporary until there is a better describe() function
#include "TensorImpl/as_lt_symmetry.hh"
// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/symmetry/print_symmetry.h>
#pragma GCC visibility pop

namespace libadcc {

Symmetry::Symmetry(std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space,
                   std::map<std::string, std::pair<size_t, size_t>> extra_axes_orbs)
      : m_mospaces_ptr(mospaces_ptr),
        m_subspaces{},
        m_irreps_allowed{},
        m_permutations{},
        m_permutations_factor{},
        m_spin_block_maps{},
        m_spin_blocks_forbidden{},
        m_axes{},
        m_alphabet{"ijklmnabcdef"} {

  // Build list of valid subspaces
  std::vector<std::string> valid_subspaces = mospaces_ptr->subspaces;
  valid_subspaces.push_back("f");
  for (auto kv : extra_axes_orbs) {
    const std::string& sp = kv.first;
    if ((sp.size() != 2 && sp.size() != 1) || sp == "o" || sp == "v" || sp == "f") {
      throw invalid_argument("The subspace identifier " + kv.first +
                             " supplied as an extra axis is invalid. It needs to be "
                             "either one or two characters wide and may not one "
                             "of 'o', 'v' or 'f'");
    }
    if (std::find(valid_subspaces.begin(), valid_subspaces.end(), kv.first) !=
        valid_subspaces.end()) {
      throw invalid_argument("The subspace identifier " + kv.first +
                             " is already a subspace inside the MoSpaces object. Choose "
                             "a different name for the extra axis.");
    }
    valid_subspaces.push_back(kv.first);
  }

  //
  // Parse and check space
  //
  std::string buffer;
  for (size_t i = 0; i < space.size(); ++i) {
    // If the buffer + space[i] is not a valid string, push back the space[i] character
    // to the buffer and try the next character. If the buffer is not emtpy, however,
    // then there is an issue (since spaces are at most 2 characters wide).
    if (std::find(valid_subspaces.begin(), valid_subspaces.end(), buffer + space[i]) ==
        valid_subspaces.end()) {
      if (buffer.empty()) {
        buffer = space[i];
      } else {
        throw invalid_argument(
              "Encountered invalid subspace identifier " + buffer + space[i] +
              " while parsing the space identifier " + space +
              ". Check that the space identifier is sound, meaning that all subspaces or "
              "spaces are contained inside the MoSpaces object.");
      }
    } else {
      m_subspaces.push_back(buffer + space[i]);
      buffer = "";
    }
  }
  if (buffer != "") {
    throw invalid_argument(
          "Encountered invalid subspace identifier " + buffer +
          " while parsing the space identifier " + space +
          ". Check that the space identifier is sound, meaning that all subspaces or "
          "spaces are contained inside the MoSpaces object.");
  }

  // Check parsed subspaces again
  for (auto& ss : m_subspaces) {
    if (std::find(valid_subspaces.begin(), valid_subspaces.end(), ss) ==
        valid_subspaces.end()) {
      throw runtime_error("Internal error: Subspace identifier " + ss + " is not valid.");
    }
  }
  if (m_subspaces.size() > m_alphabet.size()) {
    throw not_implemented_error("Builtin alphabet of Symmetry is too short.");
  }

  //
  // Build AxisInfo
  //
  for (const auto& ss : m_subspaces) {
    const auto itextra = extra_axes_orbs.find(ss);
    if (itextra == std::end(extra_axes_orbs)) {
      // ss is in MoSpaces
      m_axes.push_back(AxisInfo{ss, mospaces_ptr->n_orbs_alpha(ss),
                                mospaces_ptr->n_orbs_beta(ss),
                                mospaces_ptr->map_block_start.at(ss)});
    } else {
      // ss is an extra axes
      const size_t n_orbs_alpha = itextra->second.first;
      const size_t n_orbs_beta  = itextra->second.second;
      const size_t n_orbs       = n_orbs_alpha + n_orbs_beta;

      if (n_orbs_alpha == 0) {
        throw invalid_argument(
              "First entry in number of orbital pair corresponding to extra axis '" + ss +
              "' may not be zero.");
      }

      std::vector<size_t> splits_crude{0};
      if (n_orbs_beta > 0) {
        // There are beta orbitals
        splits_crude.push_back(n_orbs_alpha);
      }

      const std::vector<size_t> blks = construct_blocks(
            splits_crude, n_orbs, mospaces_ptr->adcmem_ptr()->max_block_size());

      m_axes.push_back(AxisInfo{ss, n_orbs_alpha, n_orbs_beta, blks});
    }
  }
}

std::string Symmetry::space() const {
  std::string ret;
  ret.reserve(2 * m_subspaces.size());
  for (auto& ss : m_subspaces) ret.append(ss);
  return ret;
}

std::vector<size_t> Symmetry::shape() const {
  std::vector<size_t> ret(m_subspaces.size());
  for (size_t i = 0; i < m_subspaces.size(); ++i) {
    ret[i] = m_axes[i].size();
  }
  return ret;
}

std::string Symmetry::describe() const {
  // TODO This can be replaced by a custom implementation
  //      in order to remove the dependency of the adcc::Symmetry class
  //      to libtensor
  std::stringstream ss;
  if (ndim() == 1) {
    auto sym_ptr = as_lt_symmetry<1>(*this);
    ss << *sym_ptr;
  } else if (ndim() == 2) {
    auto sym_ptr = as_lt_symmetry<2>(*this);
    ss << *sym_ptr;
  } else if (ndim() == 3) {
    auto sym_ptr = as_lt_symmetry<3>(*this);
    ss << *sym_ptr;
  } else if (ndim() == 4) {
    auto sym_ptr = as_lt_symmetry<4>(*this);
    ss << *sym_ptr;
  } else {
    throw not_implemented_error("Dim > 4");
  }
  return ss.str();
}

void Symmetry::clear() {
  clear_irreps_allowed();
  clear_permutations();
  clear_spin_block_maps();
  clear_spin_blocks_forbidden();
}

bool Symmetry::empty() const {
  return !has_irreps_allowed() && !has_permutations() && !has_spin_block_maps() &&
         !has_spin_blocks_forbidden();
}

void Symmetry::set_irreps_allowed(std::vector<std::string> irreps) {
  if (irreps.empty()) {
    m_irreps_allowed.clear();
    return;
  }

  const std::vector<std::string>& valid = m_mospaces_ptr->irreps;
  for (auto& ir : irreps) {
    if (std::find(valid.begin(), valid.end(), ir) == valid.end()) {
      throw invalid_argument("Invalid irreducible representation " + ir +
                             ": Could not be found in the selected point group " +
                             m_mospaces_ptr->point_group + ".");
    }
  }
  m_irreps_allowed = irreps;
}

void Symmetry::set_permutations(std::vector<std::string> permutations) {
  if (permutations.empty()) {
    m_permutations.clear();
    m_permutations_factor.clear();
    return;
  }

  if (permutations.size() < 2) {
    throw invalid_argument(
          "Number of permutations passed to set_permutations needs to be at "
          "least 2, namely the reference permutation string and a permutation to compare "
          "it with.");
  }

  // TODO Check no duplicate or conflicting entries in permutations

  // Extract reference and check it
  const std::string reference = permutations[0];
  if (reference.size() > 0 && reference[0] == '-') {
    throw invalid_argument("The reference (first) permutation (== " + reference +
                           ") may not be prefixed with a '-' character");
  }
  if (reference.size() != ndim()) {
    throw invalid_argument(
          "The number of characters in the reference permutation (== " + reference +
          ") does not agree with the dimensionality (== " + std::to_string(ndim()) +
          " to which the symmetry object is initialised.");
  }

  m_permutations.clear();
  m_permutations_factor.clear();
  for (size_t i = 1; i < permutations.size(); ++i) {
    // Extract actual permutation and prefactor (symmetric or antisymmetric)
    scalar_type factor = 1.0;
    std::string perm;
    if (permutations[i].size() > 0 && permutations[i][0] == '-') {
      perm   = permutations[i].substr(1);
      factor = -1.0;
    } else {
      perm = permutations[i];
    }

    // Check whether we are dealing with a permutation
    if (!std::is_permutation(reference.begin(), reference.end(), perm.begin())) {
      throw invalid_argument(
            "The " + std::to_string(i) +
            "-th permutation in the passed permutation list (== " + perm +
            ") is not a valid permutation of the reference (first) permutation (== " +
            reference + ").");
    }

    // Now parse and add it
    std::vector<size_t> perm_parsed;
    for (char c : perm) {
      perm_parsed.push_back(reference.find_first_of(c));
      if (perm_parsed.back() == std::string::npos) {
        throw runtime_error(
              "Internal error: Permutation character not found in reference.");
      }
    }
    m_permutations.push_back(perm_parsed);
    m_permutations_factor.push_back(factor);
  }

  if (m_permutations_factor.size() != m_permutations.size()) {

    throw runtime_error(
          "Internal error: Permutations and permutation factor list do not agree in "
          "length.");
  }
}

std::vector<std::string> Symmetry::permutations() const {
  if (m_permutations.empty()) {
    return {};
  }

  // Setup returned list and push back the reference permutation:
  std::vector<std::string> ret;
  ret.reserve(m_permutations.size() + 1);
  ret.push_back(m_alphabet.substr(0, ndim()));

  for (size_t i = 0; i < m_permutations.size(); ++i) {
    std::string elem = "";
    if (m_permutations_factor[i] == -1.0) {
      elem.push_back('-');
    } else if (m_permutations_factor[i] != 1.0) {
      throw runtime_error("Internal error: m_permutations_factor not -1.0 or 1.0.");
    }

    for (size_t idx : m_permutations[i]) {
      elem.push_back(m_alphabet[idx]);
    }
    ret.push_back(elem);
  }
  return ret;
}

std::vector<std::pair<std::vector<size_t>, scalar_type>> Symmetry::permutations_parsed()
      const {
  std::vector<std::pair<std::vector<size_t>, scalar_type>> ret;
  for (size_t i = 0; i < m_permutations.size(); ++i) {
    ret.emplace_back(m_permutations[i], m_permutations_factor[i]);
  }
  return ret;
}

void Symmetry::assert_valid_spinblock(const std::string& block) const {
  if (block.size() != ndim()) {
    throw invalid_argument(
          "Number of letters in spin block specifier " + block +
          " does not agree with dimensionality (== " + std::to_string(ndim()) + ")");
  }

  for (size_t i = 0; i < ndim(); ++i) {
    if (m_axes[i].has_spin()) {
      // Normal axis
      if (block[i] != 'a' && block[i] != 'b') {
        throw invalid_argument("The " + std::to_string(i) +
                               "-th letter of the spin block specifier " + block +
                               " is not one of 'a' (alpha) or 'b' (beta)");
      }
    } else {
      // Extra axis without spin
      if (block[i] != 'x') {
        throw invalid_argument("The " + std::to_string(i) +
                               "-th letter of the spin block specifier " + block +
                               " can only be an 'x', because it refers to an extra axis "
                               "without spin symmetry.");
      }
    }
  }
}

void Symmetry::set_spin_block_maps(
      std::vector<std::tuple<std::string, std::string, double>> spin_maps) {
  if (spin_maps.empty()) {
    m_spin_block_maps.clear();
    return;
  }

  // TODO Check no duplicate or conflicting entries in spin_maps

  for (auto& from_to_fac : spin_maps) {
    const std::string block1 = std::get<0>(from_to_fac);
    const std::string block2 = std::get<1>(from_to_fac);
    const double factor      = std::get<2>(from_to_fac);

    assert_valid_spinblock(block1);
    assert_valid_spinblock(block2);

    if (std::find(m_spin_blocks_forbidden.begin(), m_spin_blocks_forbidden.end(),
                  block1) != m_spin_blocks_forbidden.end()) {
      throw invalid_argument("The spin block " + block1 +
                             " cannot be used in an equivalent spin block mapping, "
                             "because it is already marked as a forbidden spin block.");
    }
    if (std::find(m_spin_blocks_forbidden.begin(), m_spin_blocks_forbidden.end(),
                  block2) != m_spin_blocks_forbidden.end()) {
      throw invalid_argument("The spin block " + block2 +
                             " cannot be used in an equivalent spin block mapping, "
                             "because it is already marked as a forbidden spin block.");
    }

    if (factor == 0) {
      throw invalid_argument("Spin block mapping factor may not be zero");
    }
  }

  m_spin_block_maps = spin_maps;
}

void Symmetry::set_spin_blocks_forbidden(std::vector<std::string> forbidden) {
  if (forbidden.empty()) {
    m_spin_blocks_forbidden.clear();
    return;
  }

  // TODO Check no duplicate entries in forbidden

  for (auto& block : forbidden) {
    assert_valid_spinblock(block);

    for (auto& from_to_fac : m_spin_block_maps) {
      const std::string map_block1 = std::get<0>(from_to_fac);
      const std::string map_block2 = std::get<1>(from_to_fac);
      const double factor          = std::get<2>(from_to_fac);

      if (map_block1 == block || map_block2 == block) {
        throw invalid_argument(
              "The spin block " + block +
              " cannot be marked as a forbidden spin block, "
              "because it is already used as part of an equivalent spin block mapping, "
              "namely the mapping " +
              map_block1 + "->" + map_block2 + " (" + std::to_string(factor) + ").");
      }
    }
  }

  m_spin_blocks_forbidden = forbidden;
}

}  // namespace libadcc
