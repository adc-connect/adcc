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

#include "OneParticleOperator.hh"
#include "Tensor.hh"
#include "exceptions.hh"
#include "make_symmetry.hh"
#include <iostream>

namespace libadcc {

OneParticleOperator::OneParticleOperator(std::shared_ptr<const MoSpaces> mospaces_ptr,
                                         bool is_symmetric,
                                         std::string cartesian_transform)
      : m_map{},
        m_is_symmetric(is_symmetric),
        m_orbital_subspaces{},
        m_mospaces_ptr(mospaces_ptr),
        m_cartesian_transform(cartesian_transform) {
  m_orbital_subspaces = build_orbital_subspaces(mospaces_ptr);
  for (auto spx = m_orbital_subspaces.begin(); spx != m_orbital_subspaces.end(); ++spx) {
    for (auto spy = (m_is_symmetric ? spx : m_orbital_subspaces.begin());
         spy != m_orbital_subspaces.end(); ++spy) {
      m_map[*spx + *spy] = nullptr;
    }
  }
}

std::vector<size_t> OneParticleOperator::shape() const {
  size_t size = 0;
  for (auto& ss : m_orbital_subspaces) {
    size += m_mospaces_ptr->n_orbs(ss);
  }
  return {size, size};
}

size_t OneParticleOperator::size() const {
  const auto sh = shape();
  return sh[0] * sh[1];
}

std::shared_ptr<OneParticleOperator> OneParticleOperator::copy() const {
  std::shared_ptr<OneParticleOperator> ret = std::make_shared<OneParticleOperator>(*this);

  // The m_map is copied, here we explicitly copy the contained tensors.
  for (auto& kv : m_map) {
    if (kv.second != nullptr) {
      ret->m_map[kv.first] = kv.second->copy();
    }
  }
  return ret;
}

std::vector<std::string> OneParticleOperator::blocks() const {
  std::vector<std::string> ret;
  for (auto& kv : m_map) {
    ret.push_back(kv.first);
  }
  return ret;
}

std::vector<std::string> OneParticleOperator::blocks_nonzero() const {
  std::vector<std::string> ret;
  for (auto& kv : m_map) {
    if (kv.second != nullptr) {
      ret.push_back(kv.first);
    }
  }
  return ret;
}

std::vector<std::string> OneParticleOperator::parse_split_block(
      const std::string& block) const {
  if (block.size() != 4) {
    throw invalid_argument(
          "Block specifier '" + block +
          "' is not valid: It should consist of exactly four characters.");
  }
  const std::string space_1 = block.substr(0, 2);
  const std::string space_2 = block.substr(2, 4);

  if ((space_1.front() != 'v' && space_1.front() != 'o') ||
      (space_2.front() != 'v' && space_2.front() != 'o')) {
    throw invalid_argument(
          "Block specifier '" + block +
          "' is not valid: The first two characters should mark the first orbital index "
          "space, the second two characters the second orbital index space. Each orbital "
          "index space should start with 'o' or 'v', followed by a number.");
  }

  return {space_1, space_2};
}

std::shared_ptr<Tensor> OneParticleOperator::operator[](std::string block) {
  parse_split_block(block);
  const auto it = m_map.find(block);
  if (it == m_map.end()) {
    throw invalid_argument("Block specified by '" + block +
                           "' could not be found in this density.");
  }

  if (it->second == nullptr) {  // Zero block
    auto sym_ptr = make_symmetry_operator(m_mospaces_ptr, block, m_is_symmetric,
                                          m_cartesian_transform);
    std::shared_ptr<Tensor> newtensor = make_tensor_zero(sym_ptr);
    set_block(block, newtensor);
    return newtensor;
  }

  return it->second;
}

std::shared_ptr<Tensor> OneParticleOperator::block(std::string block) const {
  parse_split_block(block);
  const auto it = m_map.find(block);
  if (it == m_map.end()) {
    throw invalid_argument("Block specified by '" + block +
                           "' could not be found in this density.");
  }

  if (it->second == nullptr) {  // Zero block
    throw invalid_argument("Block specified by '" + block +
                           "' is a zero block, which cannot be returned by block(). Use "
                           "the operator[] for this purpose. Notice, however, that will "
                           "allocate the zero block in the process.");
  }

  return it->second;
}

void OneParticleOperator::set_zero_block(std::string block) {
  parse_split_block(block);
  const auto it = m_map.find(block);
  if (it == m_map.end()) {
    throw invalid_argument("Block specified by '" + block +
                           "' could not be found in this density.");
  }
  it->second = nullptr;  // Remove the block, if it is stored
}

void OneParticleOperator::set_block(std::string block, std::shared_ptr<Tensor> value) {
  const auto spaces = parse_split_block(block);
  const auto it     = m_map.find(block);
  if (it == m_map.end()) {
    throw invalid_argument("Block specified by '" + block +
                           "' could not be found in this density.");
  }

  for (size_t i = 0; i < spaces.size(); ++i) {
    const size_t mo_size = m_mospaces_ptr->n_orbs(spaces[i]);
    if (value->shape()[i] != mo_size) {
      throw invalid_argument(
            "Density block size at dimension " + std::to_string(i) +
            " (== " + std::to_string(value->shape()[i]) +
            ")  does not agree with MoSpaces, which would suggest a size of " +
            std::to_string(mo_size) + ".");
    }
  }

  it->second = value;
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
OneParticleOperator::to_ao_basis(
      std::shared_ptr<const ReferenceState> reference_state_ptr) const {

  // Build the map of all required coefficients and perform transformation
  std::map<std::string, std::shared_ptr<Tensor>> coeff_map;
  for (auto& space : m_orbital_subspaces) {
    coeff_map[space + "_a"] =
          reference_state_ptr->orbital_coefficients_alpha(space + "b");
    coeff_map[space + "_b"] = reference_state_ptr->orbital_coefficients_beta(space + "b");
  }
  return to_ao_basis(std::move(coeff_map));
}

std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
OneParticleOperator::to_ao_basis(
      std::map<std::string, std::shared_ptr<Tensor>> coefficient_ptrs) const {
  if (coefficient_ptrs.size() == 0) {
    throw invalid_argument("Not a single coefficient found in the coefficient_ptrs map.");
  }
  for (auto& kv : coefficient_ptrs) {
    auto& c_xb_ptr = coefficient_ptrs.begin()->second;
    if (c_xb_ptr->ndim() != 2) {
      throw invalid_argument("The dimensionality of the coefficient pointer '" +
                             kv.first +
                             "' is not 2 as it should be for MO coefficients.");
    }
  }

  //
  // Transform MO subspace by MO subspace to AOs and add
  //
  std::shared_ptr<Tensor> dm_bb_a = nullptr;
  std::shared_ptr<Tensor> dm_bb_b = nullptr;

  for (auto spx = m_orbital_subspaces.begin(); spx != m_orbital_subspaces.end(); ++spx) {
    for (auto spy = (m_is_symmetric ? spx : m_orbital_subspaces.begin());
         spy != m_orbital_subspaces.end(); ++spy) {
      if (is_zero_block(*spx + *spy)) {
        continue;
      }

      const std::shared_ptr<Tensor> dm_xy_ptr = m_map.at(*spx + *spy);
      if (dm_xy_ptr == nullptr) {
        throw runtime_error(
              "OneParticleOperator cannot be transformed to AO, since not all MO "
              "blocks have been initialised. First block still a nullptr: " +
              *spx + *spy);
      }

      std::shared_ptr<Tensor> term_bb_a = nullptr;
      std::shared_ptr<Tensor> term_bb_b = nullptr;
      // Transform this block MO -> AO
      auto mo_to_ao = [&dm_xy_ptr](std::shared_ptr<Tensor> c_xb,
                                   std::shared_ptr<Tensor> c_yb) {
        // term_bb = c_xb^T * dm_xy * c_yb
        auto dmc_xb = dm_xy_ptr->tensordot(c_yb, {{1}, {0}}).tensor_ptr;
        return c_xb->tensordot(dmc_xb, {{0}, {0}}).tensor_ptr;
      };
      try {
        term_bb_a =
              mo_to_ao(coefficient_ptrs[*spx + "_a"], coefficient_ptrs[*spy + "_a"]);
        term_bb_b =
              mo_to_ao(coefficient_ptrs[*spx + "_b"], coefficient_ptrs[*spy + "_b"]);
      } catch (std::out_of_range& e) {
        throw invalid_argument(
              "Did not find coefficient for an MO subspace identifier: " +
              std::string(e.what()) + ".");
      }

      if (m_is_symmetric) {  // Symmetrise tensors
        term_bb_a = term_bb_a->symmetrise({{0, 1}});
        term_bb_b = term_bb_b->symmetrise({{0, 1}});
      }

      // if this operator is symmetric and spx != spy, then
      // both spaces are identical, hence this pair contributes
      // twice to the resulting density, since for each pair (o1,v1)
      // we only keep a single combination, even though both blocks
      // (v1,o1) and (o1,v1) exist and contribute in this case.
      const double fac = (m_is_symmetric && spx != spy) ? 2. : 1.;

      if (dm_bb_a == nullptr) {
        dm_bb_a = term_bb_a->scale(fac);
        dm_bb_b = term_bb_b->scale(fac);
      } else {
        dm_bb_a = dm_bb_a->add(term_bb_a->scale(fac));
        dm_bb_b = dm_bb_b->add(term_bb_b->scale(fac));
      }
    }  // spy
  }    // spx

  if (dm_bb_a == nullptr || dm_bb_b == nullptr) {
    throw runtime_error(
          "At least one block of the OneParticleOperator needs to be non-zero.");
  }

  return std::make_pair(evaluate(dm_bb_a), evaluate(dm_bb_b));
}

std::vector<std::string> OneParticleOperator::build_orbital_subspaces(
      std::shared_ptr<const MoSpaces> mospaces_ptr) const {
  // Note: Subspaces are counted 1-based!
  //       Unlike many other objects the subspaces with *larger* indices
  //       are treated first in adcman. So we keep this ordering here
  //       as well for compatiblity.
  //
  std::vector<std::string> occs(mospaces_ptr->subspaces_occupied);
  std::vector<std::string> virts(mospaces_ptr->subspaces_virtual);
  if (occs.size() > 9 || virts.size() > 9) {
    throw not_implemented_error(
          "OneParticleOperator only implemented up to 9 occupied or virtual orbital "
          "subspaces.");
  }
  if (occs.size() < 1 || virts.size() < 1) {
    throw runtime_error(
          "Internal error: Need at least one virtual and one occupied subspace.");
  }

  // Sort, reverse and merge
  std::sort(occs.begin(), occs.end());
  std::reverse(occs.begin(), occs.end());
  std::sort(virts.begin(), virts.end());
  std::reverse(virts.begin(), virts.end());
  for (auto& elem : virts) occs.push_back(elem);
  return occs;
}

void OneParticleOperator::export_to(scalar_type* memptr, size_t size) const {
  if (this->size() != size) {
    throw invalid_argument("The memory provided (== " + std::to_string(size) +
                           ") does not agree with the number of operator elements (== " +
                           std::to_string(this->size()) + ")");
  }
  const std::vector<size_t> res_strides{shape()[1], 1};
  const MoSpaces& mospaces = *m_mospaces_ptr;

  // Row offset inside memptr to which the current block is to be copied
  size_t row_offset = 0;
  for (auto spx = m_orbital_subspaces.begin(); spx != m_orbital_subspaces.end(); ++spx) {
    // Column offset inside memptr to which the current block is to be copied
    size_t col_offset = 0;
    for (auto spy = m_orbital_subspaces.begin(); spy != m_orbital_subspaces.end();
         ++spy) {
      // One has to be a little careful here, since for symmetric
      // density matrices spy < spx is not stored.

      std::shared_ptr<Tensor> dm_xy_ptr = nullptr;
      std::vector<size_t> buffer_strides;

      std::vector<scalar_type> buffer(mospaces.n_orbs(*spx) * mospaces.n_orbs(*spy));
      if (is_symmetric() && spy < spx) {
        // Here we take *spy + *spx (which exists)
        // but alter the strides such that data is read
        // in transposed form.
        buffer_strides = {1, mospaces.n_orbs(*spx)};
        if (!is_zero_block(*spy + *spx)) {
          dm_xy_ptr = block(*spy + *spx);
          dm_xy_ptr->export_to(buffer);
          if (dm_xy_ptr->shape() !=
              std::vector<size_t>{mospaces.n_orbs(*spy), mospaces.n_orbs(*spx)}) {
            throw runtime_error("Internal error: Tensor does not have expected shape.");
          }
        }
      } else {
        buffer_strides = {mospaces.n_orbs(*spy), 1};
        if (!is_zero_block(*spx + *spy)) {
          dm_xy_ptr = block(*spx + *spy);
          dm_xy_ptr->export_to(buffer);
          if (dm_xy_ptr->shape() !=
              std::vector<size_t>{mospaces.n_orbs(*spx), mospaces.n_orbs(*spy)}) {
            throw runtime_error("Internal error: Tensor does not have expected shape.");
          }
        }
      }

      // Copy into res
      for (size_t i = 0; i < mospaces.n_orbs(*spx); ++i) {
        for (size_t j = 0; j < mospaces.n_orbs(*spy); ++j) {
          const size_t i_off   = i + row_offset;
          const size_t j_off   = j + col_offset;
          const size_t toidx   = res_strides[0] * i_off + res_strides[1] * j_off;
          const size_t fromidx = buffer_strides[0] * i + buffer_strides[1] * j;
#ifndef NDEBUG
          if (toidx >= size) {
            throw runtime_error("Internal error: Buffer overflow toidx");
          }
          if (fromidx >= buffer.size()) {
            throw runtime_error("Internal error: Buffer overflow fromidx");
          }
#endif  // NDEBUG
          memptr[toidx] = buffer[fromidx];
        }
      }

      // Advance column offset
      col_offset += mospaces.n_orbs(*spy);
    }
    // Advance row offset
    row_offset += mospaces.n_orbs(*spx);
  }
}

}  // namespace libadcc
