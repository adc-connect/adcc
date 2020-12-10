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

#include "ReferenceState.hh"
#include "MoIndexTranslation.hh"
#include "TensorImpl.hh"
#include "exceptions.hh"
#include "import_eri.hh"
#include "make_symmetry.hh"

namespace libadcc {
ReferenceState::ReferenceState(std::shared_ptr<const HartreeFockSolution_i> hfsoln_ptr,
                               std::shared_ptr<const MoSpaces> mo_ptr,
                               bool symmetry_check_on_import)
      : m_n_alpha{0},
        m_n_beta{0},
        m_orben{},
        m_orbcoeff{},
        m_orbcoeff_alpha{},
        m_orbcoeff_beta{},
        m_fock{},
        m_eri{},
        m_hfsoln_ptr{hfsoln_ptr},
        m_mo_ptr{mo_ptr},
        m_symmetry_check_on_import{symmetry_check_on_import},
        m_timer{} {
  for (auto& ss : mo_ptr->subspaces_occupied) {
    m_n_alpha += mo_ptr->n_orbs_alpha(ss);
    m_n_beta += mo_ptr->n_orbs_beta(ss);
  }

  // Import orbital energies and coefficients
  import_orbital_energies(*hfsoln_ptr, mo_ptr, symmetry_check_on_import);
  import_orbital_coefficients(*hfsoln_ptr, mo_ptr, symmetry_check_on_import);
}

//
// Importers
//
void ReferenceState::import_orbital_energies(
      const HartreeFockSolution_i& hf, const std::shared_ptr<const MoSpaces>& mo_ptr,
      bool symmetry_check_on_import) {
  m_orben.clear();
  RecordTime rec(m_timer, "import/orbital_energies");

  std::vector<scalar_type> tmp_orben(hf.n_orbs());
  hf.orben_f(tmp_orben.data(), tmp_orben.size());
  for (auto& ss : mo_ptr->subspaces) {
    std::shared_ptr<Symmetry> sym_ptr = make_symmetry_orbital_energies(mo_ptr, ss);
    std::shared_ptr<Tensor> orben     = make_tensor_zero(sym_ptr);
    MoIndexTranslation idxtrans(mo_ptr, ss);

    auto orben_generator = [&tmp_orben, &idxtrans](
                                 const std::vector<std::pair<size_t, size_t>>& block,
                                 scalar_type* buffer) {
      const std::pair<size_t, size_t>& block_ss = block[0];
      const size_t length_ss_full               = block_ss.second - block_ss.first;

      std::vector<RangeMapping> mappings = idxtrans.map_range_to_hf_provider(block);
      for (const RangeMapping& map : mappings) {
        // orben is 1D -> idim == 0 always
        const size_t idim = 0;

        // Starting index of the mapping in the adcc index convention
        const size_t start = map.from().axis(idim).start() - block_ss.first;

        // Start and end in the HfProvider indexing convention
        const size_t start_hf = map.to().axis(idim).start();
        const size_t end_hf   = map.to().axis(idim).end();

        if (start + (end_hf - start_hf) > length_ss_full) {
          throw runtime_error("Internal error: Out-of-bounds write in orben_generator");
        }
        // Do the "import" by copying the appropriate part of the buf_orben object
        std::copy(tmp_orben.begin() + static_cast<ptrdiff_t>(start_hf),
                  tmp_orben.begin() + static_cast<ptrdiff_t>(end_hf), buffer + start);
      }
    };
    orben->import_from(orben_generator, hf.conv_tol(), symmetry_check_on_import);
    orben->set_immutable();
    m_orben[ss] = orben;
  }
}

void ReferenceState::import_orbital_coefficients(
      const HartreeFockSolution_i& hf, const std::shared_ptr<const MoSpaces>& mo_ptr,
      bool symmetry_check_on_import) {
  RecordTime rec(m_timer, "import/orbital_coefficients");
  m_orbcoeff.clear();
  m_orbcoeff_alpha.clear();
  m_orbcoeff_beta.clear();

  std::vector<scalar_type> tmp_orbcoeff(hf.n_orbs() * hf.n_bas());
  tmp_orbcoeff.resize(hf.n_orbs() * hf.n_bas());
  hf.orbcoeff_fb(tmp_orbcoeff.data(), tmp_orbcoeff.size());
  for (auto& ss : mo_ptr->subspaces) {
    // The index translation for the first subspace (which is not "b")
    MoIndexTranslation idxtrans(mo_ptr, ss);

    auto orbcoeff_generator_ab = [&tmp_orbcoeff, &idxtrans, &hf](
                                       const std::vector<std::pair<size_t, size_t>>&
                                             block,
                                       scalar_type* buffer, bool alpha, bool beta) {
      const std::pair<size_t, size_t>& block_ss = block[0];
      const std::pair<size_t, size_t>& block_b  = block[1];

      // Length of the axis:
      const size_t length_b       = block_b.second - block_b.first;
      const size_t length_ss_full = block_ss.second - block_ss.first;

      std::vector<RangeMapping> mappings_ss =
            idxtrans.map_range_to_hf_provider({{block_ss}});
      for (const RangeMapping& map_ss : mappings_ss) {
        // Get start and end of index ranges in adcc and HfProvider convention
        const size_t idim      = 0;  // The ss subspace of orbcoeff is 1D => idim == 0
        const size_t start_ss  = map_ss.from().axis(idim).start();
        const size_t end_ss    = map_ss.from().axis(idim).end();
        const size_t length_ss = end_ss - start_ss;

        const size_t start_hf = map_ss.to().axis(idim).start();
        const size_t end_hf   = map_ss.to().axis(idim).end();

        // Since the orbecoeff are 2D-arrays and moreover the tmp_orbcoeff vector is a 2D
        // array in row-major C-like index ordering, the first element of this buffer to
        // copy is
        auto ittmp = tmp_orbcoeff.begin() +
                     static_cast<ptrdiff_t>(start_hf * hf.n_bas() + block_b.first);

        // The first element into which data will be copied, on the other hand,
        // is shifted, like the following, since in the output buffer there will
        // length_b columns and the first (start_ss - block_ss.first) rows are skipped
        // in this mapping loop.
        scalar_type* ptrbuf =
              buffer + static_cast<ptrdiff_t>((start_ss - block_ss.first) * length_b);

        if (alpha && beta) {
          throw runtime_error("Internal error: Requested both alpha and beta block.");
        }

        // Fill with zeros if we are in the alpha or beta block and this
        // should be skipped
        bool fill_zeros = false;
        if (idxtrans.spin_of({start_ss}) == "b") {
          if (idxtrans.spin_of({end_ss - 1}) != "b") {
            throw runtime_error(
                  "Internal error: Assertion about blocks being split whenever spin "
                  "changes is violated.");
          }
          if (!beta) {
            // beta block should be zero
            fill_zeros = true;
          }
        }
        if (idxtrans.spin_of({end_ss - 1}) == "a") {
          if (idxtrans.spin_of({start_ss}) != "a") {
            throw runtime_error(
                  "Internal error: Assertion about blocks being split whenever spin "
                  "changes is violated.");
          }
          if (!alpha) {
            // beta block should be zero
            fill_zeros = true;
          }
        }

        const ptrdiff_t slength_b = static_cast<ptrdiff_t>(length_b);
        const ptrdiff_t sn_bas    = static_cast<ptrdiff_t>(hf.n_bas());

        // Copy length_b elements at a time (one row) and then advance the pointer into
        // the buffer (ptrbuf) and the iterator into the tmp_orbcoeff vector.
        // For the ittmp iterator, n_bas() elements need to be skipped, for ptrbuf only
        // length_b, because it only is a matrix block with length_b columns.
        for (size_t irow = 0; irow < length_ss;
             ++irow, ptrbuf += length_b, ittmp += sn_bas) {
          if (ptrbuf + slength_b > buffer + length_ss_full * length_b) {
            throw runtime_error(
                  "Internal error: Out-of-bounds write in orbcoeff_generator");
          }
          if (fill_zeros) {
            std::fill(ptrbuf, ptrbuf + length_b, 0);
          } else {
            std::copy(ittmp, ittmp + slength_b, ptrbuf);
          }
        }

        if (ittmp != tmp_orbcoeff.begin() +
                           static_cast<ptrdiff_t>(end_hf * hf.n_bas() + block_b.first)) {
          throw runtime_error("Internal error: ittmp index issue in orbcoeff_generator");
        }
        if (ptrbuf != buffer + (end_ss - block_ss.first) * length_b) {
          throw runtime_error("Internal error: ptrbuf index issue in orbcoeff_generator");
        }
      }
    };

    //
    // Import alpha coefficients, i.e. the vector (a)
    //                                            (0)
    //
    std::shared_ptr<Symmetry> sym_a_ptr =
          make_symmetry_orbital_coefficients(mo_ptr, ss + "b", hf.n_bas(), "a");
    std::shared_ptr<Tensor> orbcoeff_a = make_tensor_zero(sym_a_ptr);
    orbcoeff_a->import_from(
          [&orbcoeff_generator_ab](const std::vector<std::pair<size_t, size_t>>& block,
                                   scalar_type* buffer) {
            orbcoeff_generator_ab(block, buffer, /*alpha*/ true, /*beta*/ false);
          },
          hf.conv_tol(), symmetry_check_on_import);
    orbcoeff_a->set_immutable();
    m_orbcoeff_alpha[ss + "b"] = orbcoeff_a;

    //
    // Import beta coefficients, i.e. the vector (0)
    //                                           (b)
    //
    std::shared_ptr<Symmetry> sym_b_ptr =
          make_symmetry_orbital_coefficients(mo_ptr, ss + "b", hf.n_bas(), "b");
    std::shared_ptr<Tensor> orbcoeff_b = make_tensor_zero(sym_b_ptr);
    orbcoeff_b->import_from(
          [&orbcoeff_generator_ab](const std::vector<std::pair<size_t, size_t>>& block,
                                   scalar_type* buffer) {
            orbcoeff_generator_ab(block, buffer, /*alpha*/ false, /*beta*/ true);
          },
          hf.conv_tol(), symmetry_check_on_import);
    orbcoeff_b->set_immutable();
    m_orbcoeff_beta[ss + "b"] = orbcoeff_b;

    //
    // Build full coefficients (a 0)
    //                         (0 b)
    //
    // Export orbcoeff_a and orbcoeff_b for the full coefficient import
    // (The rationale is that these already have the right MO ordering along the MO axis,
    //  which makes the code for the full coefficient import a lot more readable)
    std::vector<scalar_type> exp_orbcoeff_a;
    orbcoeff_a->export_to(exp_orbcoeff_a);
    std::vector<scalar_type> exp_orbcoeff_b;
    orbcoeff_b->export_to(exp_orbcoeff_b);
    if (exp_orbcoeff_a.size() != exp_orbcoeff_b.size()) {
      throw runtime_error(
            "Internal error: exp_orbcoeff_a.size() and exp_orbcoeff_b.size() do not "
            "agree.");
    }

    auto orbcoeff_generator = [&exp_orbcoeff_a, &exp_orbcoeff_b, &hf](
                                    const std::vector<std::pair<size_t, size_t>>& block,
                                    scalar_type* buffer) {
      const std::pair<size_t, size_t>& block_ss = block[0];
      const std::pair<size_t, size_t>& block_b  = block[1];

      // Length of the axis:
      const size_t length_b       = block_b.second - block_b.first;
      const size_t length_ss_full = block_ss.second - block_ss.first;

      // Spin data for axis b
      const std::vector<std::pair<size_t, size_t>> b_spin_ranges{
            {0, hf.n_bas()}, {hf.n_bas(), 2 * hf.n_bas()}};
      const std::vector<scalar_type*> b_spin_data{exp_orbcoeff_a.data(),
                                                  exp_orbcoeff_b.data()};

      // Loop over spins (alpha == 0 and beta == 1)
      for (size_t ispin = 0; ispin < 2; ++ispin) {
        size_t from_b_first  = block_b.first;
        size_t from_b_length = length_b;

        if (block_b.first >= b_spin_ranges[ispin].second) {
          // Beyond the range covered by this spin
          continue;
        }
        if (block_b.second - 1 < b_spin_ranges[ispin].first) {
          // Before the range covered by this spin
          continue;
        }

        // Get the first index in the b axis to be used and the length
        // for the spin chunk covered in this loop iteration
        from_b_first  = std::max(block_b.first, b_spin_ranges[ispin].first);
        from_b_length = std::min(hf.n_bas(), block_b.second - from_b_first);

        // from_b_first has still spin information (i.e. indexed in the range
        // [0,2*n_bas], but exp_orbcoeff_a and exp_orbcoeff_b containeres
        // only covers the spatial range [0, n_bas], so we reduce it to
        // the spatial index only
        const size_t from_b_first_spatial = from_b_first % hf.n_bas();

        // Prepare input and output pointers and copy the data.
        scalar_type* from   = b_spin_data[ispin];
        scalar_type* ptrbuf = buffer + (from_b_first - block_b.first) * length_ss_full;
        size_t offset       = hf.n_bas() * block_ss.first + from_b_first_spatial;
        for (size_t iss = 0; iss < length_ss_full;
             ++iss, ptrbuf += length_b, offset += hf.n_bas()) {
          if (offset + from_b_length > exp_orbcoeff_a.size()) {
            throw runtime_error("Read buffer overflow.");
          }
          if (ptrbuf + from_b_length > buffer + length_b * length_ss_full) {
            throw runtime_error("Write buffer overflow.");
          }
          if (ptrbuf < buffer) {
            throw runtime_error("Write buffer underflow.");
          }
          std::copy(from + offset, from + offset + from_b_length, ptrbuf);
        }
      }
    };
    std::shared_ptr<Symmetry> sym_ab_ptr =
          make_symmetry_orbital_coefficients(mo_ptr, ss + "b", hf.n_bas(), "ab");
    std::shared_ptr<Tensor> orbcoeff_ab = make_tensor_zero(sym_ab_ptr);
    orbcoeff_ab->import_from(orbcoeff_generator, hf.conv_tol(), symmetry_check_on_import);
    orbcoeff_ab->set_immutable();
    m_orbcoeff[ss + "b"] = orbcoeff_ab;
  }
}

//
// Info about the reference state
//
size_t ReferenceState::spin_multiplicity() const {
  if (!restricted()) {
    return 0;
  } else {
    return m_hfsoln_ptr->spin_multiplicity();
  }
}

std::string ReferenceState::irreducible_representation() const {
  if (m_mo_ptr->point_group == "C1") return "A";

  // Notice: This is not true, since the occupation might be uneven
  // between spatial parts
  // if (n_alpha() == n_beta()) return m_mo_ptr->irrep_totsym();

  // For non-closed-shell we would need to see whatever orbitals are occupied
  // and determine from this the resulting overall irrep
  throw not_implemented_error("Only C1 is implemented.");
}

std::vector<scalar_type> ReferenceState::nuclear_multipole(size_t order) const {
  std::vector<scalar_type> ret((order + 2) * (order + 1) / 2);
  m_hfsoln_ptr->nuclear_multipole(order, ret.data(), ret.size());
  return ret;
}

//
// Tensor data
//
std::shared_ptr<Tensor> ReferenceState::orbital_energies(const std::string& space) const {
  const auto itfound = m_orben.find(space);
  if (itfound == m_orben.end()) {
    throw invalid_argument(
          "Invalid space string " + space + ": An object orbital_energies(" + space +
          ") is not known. Check that the string has exactly 2 characters "
          "and describes a valid orbital subspace.");
  }
  return itfound->second;
}

std::shared_ptr<Tensor> ReferenceState::orbital_coefficients(
      const std::string& space) const {
  const auto itfound = m_orbcoeff.find(space);
  if (itfound == m_orbcoeff.end()) {
    throw invalid_argument(
          "Invalid space string " + space + ": An object orbital_coefficients(" + space +
          ") is not known. Check that the string has exactly 3 characters "
          "and describes two valid orbital subspace.");
  }
  return itfound->second;
}

std::shared_ptr<Tensor> ReferenceState::orbital_coefficients_alpha(
      const std::string& space) const {
  const auto itfound = m_orbcoeff_alpha.find(space);
  if (itfound == m_orbcoeff_alpha.end()) {
    throw invalid_argument(
          "Invalid space string " + space + ": An object orbital_coefficients_alpha(" +
          space +
          ") is not known. Check that the string has exactly 3 characters "
          "and describes two valid orbital subspace.");
  }
  return itfound->second;
}

std::shared_ptr<Tensor> ReferenceState::orbital_coefficients_beta(
      const std::string& space) const {
  const auto itfound = m_orbcoeff_beta.find(space);
  if (itfound == m_orbcoeff_beta.end()) {
    throw invalid_argument(
          "Invalid space string " + space + ": An object orbital_coefficients_beta(" +
          space +
          ") is not known. Check that the string has exactly 3 characters "
          "and describes two valid orbital subspace.");
  }
  return itfound->second;
}

std::shared_ptr<Tensor> ReferenceState::fock(const std::string& space) const {
  const auto itfound = m_fock.find(space);
  if (itfound != m_fock.end()) return itfound->second;

  std::unique_ptr<MoIndexTranslation> trans_ptr;
  try {
    trans_ptr.reset(new MoIndexTranslation(m_mo_ptr, space));
  } catch (const invalid_argument& inv) {
    throw invalid_argument("Invalid fock space string: " + std::string(inv.what()));
  }
  MoIndexTranslation& idxtrans = *trans_ptr;
  if (idxtrans.ndim() != 2) {
    throw invalid_argument("Invalid fock space string" + space +
                           ": Needs to consist of exactly 2 spaces.");
  }

  {
    RecordTime rec(m_timer, "import/fock/" + space);
    // Fock operator is symmetric wrt. index permutation and transforms
    // as the totally symmetric irrep
    const bool symmetric = true;
    std::shared_ptr<Symmetry> sym_ptr =
          make_symmetry_operator(m_mo_ptr, space, symmetric, "1");
    std::shared_ptr<Tensor> fock    = make_tensor_zero(sym_ptr);
    const HartreeFockSolution_i& hf = *m_hfsoln_ptr;

    auto fock_generator = [&idxtrans, &hf](
                                const std::vector<std::pair<size_t, size_t>>& block,
                                scalar_type* buffer) {
      // Strides, which should be used to write data to buffer
      // and size of the buffer.
      const std::vector<size_t> blength{block[0].second - block[0].first,
                                        block[1].second - block[1].first};
      const std::vector<size_t> strides{blength[1], 1};
      const size_t buffer_size = strides[0] * blength[0];

      std::vector<RangeMapping> mappings = idxtrans.map_range_to_hf_provider(block);
      for (RangeMapping& map : mappings) {
        const SimpleRange& bidx  = map.from();  // Indices of the MO subspace block
        const SimpleRange& hfidx = map.to();    // Indices translated to HF provider

        // Compute offset and remaining buffer size
        size_t offset = 0;
        for (size_t i = 0; i < 2; ++i) {
          offset += (bidx.axis(i).start() - block[i].first) * strides[i];
        }
        const size_t size = buffer_size - offset;
        hf.fock_ff(hfidx.axis(0).start(), hfidx.axis(0).end(),  //
                   hfidx.axis(1).start(), hfidx.axis(1).end(),  //
                   strides[0], strides[1], buffer + offset, size);
      }
    };
    fock->import_from(fock_generator, hf.conv_tol(), m_symmetry_check_on_import);
    fock->set_immutable();

    m_fock[space] = fock;
    return fock;
  }
}

std::shared_ptr<Tensor> ReferenceState::eri(const std::string& space) const {
  const auto itfound = m_eri.find(space);
  if (itfound != m_eri.end()) return itfound->second;

  std::unique_ptr<MoIndexTranslation> trans_ptr;
  try {
    trans_ptr.reset(new MoIndexTranslation(m_mo_ptr, space));
  } catch (const invalid_argument& inv) {
    throw invalid_argument("Invalid ERI space string: " + std::string(inv.what()));
  }
  if (trans_ptr->ndim() != 4) {
    throw invalid_argument("Invalid ERI space string" + space +
                           ": Needs to consist of exactly 4 spaces.");
  }

  // Branch off to three different ways to import the tensor, depending on the
  // capabilities of the HartreFockSolution_i and the block to be imported.
  RecordTime rec(m_timer, "import/eri/" + space);
  const std::vector<std::string>& ss = trans_ptr->subspaces();
  if (m_hfsoln_ptr->has_eri_phys_asym_ffff()) {
    m_eri[space] =
          import_eri_asym_direct(*m_hfsoln_ptr, *trans_ptr, m_symmetry_check_on_import);
  } else if (ss[2] == ss[3] || ss[0] == ss[1]) {
    m_eri[space] = import_eri_chem_then_asym_fast(*m_hfsoln_ptr, *trans_ptr,
                                                  m_symmetry_check_on_import);
  } else {
    m_eri[space] = import_eri_chem_then_asym(*m_hfsoln_ptr, *trans_ptr,
                                             m_symmetry_check_on_import);
  }
  m_eri[space]->set_immutable();
  return m_eri[space];
}

void ReferenceState::import_all() const {
  auto& ss = m_mo_ptr->subspaces;
  std::vector<std::string> ss_pairs;
  for (auto it1 = ss.begin(); it1 != ss.end(); ++it1) {
    for (auto it2 = it1; it2 != ss.end(); ++it2) {
      ss_pairs.push_back(*it1 + *it2);
      fock(*it1 + *it2);
    }
  }

  for (auto it1 = ss_pairs.begin(); it1 != ss_pairs.end(); ++it1) {
    for (auto it2 = it1; it2 != ss_pairs.end(); ++it2) {
      eri(*it1 + *it2);
    }
  }

  flush_hf_cache();
}

std::vector<std::string> ReferenceState::cached_fock_blocks() const {
  std::vector<std::string> ret;
  for (auto& kv : m_fock) ret.push_back(kv.first);
  return ret;
}

std::vector<std::string> ReferenceState::cached_eri_blocks() const {
  std::vector<std::string> ret;
  for (auto& kv : m_eri) ret.push_back(kv.first);
  return ret;
}

void ReferenceState::set_cached_fock_blocks(std::vector<std::string> newlist) {
  for (auto& item : newlist) fock(item);
  flush_hf_cache();

  std::sort(newlist.begin(), newlist.end());
  auto it = m_fock.begin();
  while (it != m_fock.end()) {
    if (std::binary_search(newlist.begin(), newlist.end(), it->first)) {
      ++it;
    } else {
      it = m_fock.erase(it);
    }
  }
}

void ReferenceState::set_cached_eri_blocks(std::vector<std::string> newlist) {
  for (auto& item : newlist) eri(item);
  flush_hf_cache();

  std::sort(newlist.begin(), newlist.end());
  auto it = m_eri.begin();
  while (it != m_eri.end()) {
    if (std::binary_search(newlist.begin(), newlist.end(), it->first)) {
      ++it;
    } else {
      it = m_eri.erase(it);
    }
  }
}

}  // namespace libadcc
