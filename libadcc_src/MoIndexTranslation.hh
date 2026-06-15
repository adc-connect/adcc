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
#include "MoSpaces.hh"

namespace libadcc {
/**
 *  \addtogroup ReferenceObjects
 */
///@{

/** Very simple structure holding the range of indices for a single axis
 *
 * \note This structure is not intended for wide use throughout
 *       the adcc code, much rather it should help to make the
 *       MoIndexTranslation code more clear. Outside this class it can
 *       be well thought of as a std::pair<size_t, size_t>.
 */
struct AxisRange : std::pair<size_t, size_t> {
  using std::pair<size_t, size_t>::pair;

  //@{
  /** Start of the range, i.e. the first index */
  size_t start() const { return first; }
  size_t& start() { return first; }
  //@}

  //@{
  /** End of the range, i.e. the index past the last index */
  size_t end() const { return second; }
  size_t& end() { return second; }
  //@}

  /** Length of the range, i.e. number of indices contained */
  size_t length() const { return second - first; }
};

/** Very simple structure holding an index range.
 *
 * The ranges are given as lists of ranges in each axis,
 * where for each axis the first number is the first index
 * of the range to be taken and the second number is the
 * first index past the range, i.e. the index range in each
 * axis are given as half-open intervals [start, end).
 *
 * \note This structure is not intended for wide use throughout
 * the adcc code, much rather it should help to make the
 * MoIndexTranslation code more clear. Outside this class it can
 * be well thought of as a std::vector<std::pair<size_t, size_t>>.
 */
struct SimpleRange : std::vector<std::pair<size_t, size_t>> {
  using std::vector<std::pair<size_t, size_t>>::vector;

  /** Build a Range object by specifying all starts and all ends */
  SimpleRange(std::vector<size_t> starts, std::vector<size_t> ends);

  /** Build a Range object by providing a vector of [start, end) pairs */
  SimpleRange(std::vector<std::pair<size_t, size_t>> in)
        : std::vector<std::pair<size_t, size_t>>(in){};

  /** Build an empty SimpleRange object */
  SimpleRange() : std::vector<std::pair<size_t, size_t>>{} {}

  /** Return the dimensionality, i.e. the size of the vector */
  size_t ndim() const { return size(); }

  /** Return the range along an axis */
  AxisRange axis(size_t i) const {
    return AxisRange{(*this)[i].first, (*this)[i].second};
  }

  /** Add an axis by specifying its range */
  void push_axis(std::pair<size_t, size_t> new_axis) { push_back(std::move(new_axis)); }

  /** Get the vector of all range starts */
  std::vector<size_t> starts() const;

  /** Get the vectors of the last indices in all axes */
  std::vector<size_t> lasts() const;

  /** Get the vector of all range ends */
  std::vector<size_t> ends() const;
};

/** Datastructure to describe a mapping of one range of indices
 *  to another range of indices.
 *
 * \note This structure is not intended for wide use throughout
 * the adcc code, much rather it should help to make the
 * MoIndexTranslation code more clear. Outside this class it can
 * be well thought of as a pair of above range objects, which
 * are itself effectively std::vector<std::pair<size_t, size_t>>.
 */
struct RangeMapping : std::pair<SimpleRange, SimpleRange> {
  using std::pair<SimpleRange, SimpleRange>::pair;

  //@{
  /** The range from which we map */
  SimpleRange& from() { return first; }
  const SimpleRange& from() const { return first; }
  //@}

  //@{
  /** The range to which we map */
  SimpleRange& to() { return second; }
  const SimpleRange& to() const { return second; }
  //@}
};

/** Translation object, which helps to translate various representations
 *  of a tensorial index over orbitals subspaces. E.g. it helps to
 *  identify the spin block, reduce indices to spatial indices,
 *  translate indices or index ranges to the original ordering used
 *  in the SCF program and similar tasks.
 */
class MoIndexTranslation {
 public:
  /** Create an MoIndexTranslation object for the provided space string. */
  MoIndexTranslation(std::shared_ptr<const MoSpaces> mospaces_ptr,
                     const std::string& space);

  /** Create an MoIndexTranslation object for a list of subspace identifiers */
  MoIndexTranslation(std::shared_ptr<const MoSpaces> mospaces_ptr,
                     const std::vector<std::string>& subspaces);

  //
  // Access to shape and sizes
  //
  /** Return the MoSpaces to which this MoIndexTranslation is set up */
  std::shared_ptr<const MoSpaces> mospaces_ptr() const { return m_mospaces_ptr; }

  /** The space used to initialise the object */
  std::string space() const;

  /** The space splitup into subspaces along each dimension */
  const std::vector<std::string>& subspaces() const { return m_subspaces; }

  /** Number of dimensions */
  size_t ndim() const { return m_subspaces.size(); }

  /** Shape of each dimension */
  const std::vector<size_t>& shape() const { return m_shape; }

  //
  // Mappings between full MO space (e.g ffff) and subspace (e.g. o1v1o1o1)
  //
  /** Map an index given in the space passed upon construction to the corresponding
   *  index in the range of all MO indices (the ffff space) */
  std::vector<size_t> full_index_of(const std::vector<size_t>& index) const;

  //
  // Splitting and combination of subspace indices
  //
  /** Get the block index of an index */
  std::vector<size_t> block_index_of(const std::vector<size_t>& index) const;

  /** Get the spatial block index of an index
   *
   * The spatial block index is the result of block_index_of modulo the spin blocks,
   * i.e. it maps an index onto the index of the *spatial* blocks only, such that
   * the resulting value is idential for two index where the MOs only differ
   * by spin. For example the 1st core alpha and the 1st core beta orbital will
   * map to the same value upon a call of this function. */
  std::vector<size_t> block_index_spatial_of(const std::vector<size_t>& index) const;

  /** Get the in-block index, i.e. the index within the tensor block */
  std::vector<size_t> inblock_index_of(const std::vector<size_t>& index) const {
    return split(index).second;
  }

  /** Get the spin block of each of the index components */
  std::string spin_of(const std::vector<size_t>& index) const;

  /** Split an index into block index and in-block index and return the result */
  std::pair<std::vector<size_t>, std::vector<size_t>> split(
        const std::vector<size_t>& index) const;

  /** Split an index into a spin block descriptor, a spatial block index and an in-block
   * index. */
  std::tuple<std::string, std::vector<size_t>, std::vector<size_t>> split_spin(
        const std::vector<size_t>& index) const;

  /** Combine a block index and an in-block index into a subspace index */
  std::vector<size_t> combine(const std::vector<size_t>& block_index,
                              const std::vector<size_t>& inblock_index) const;

  /** Combine a descriptor for the spin block, a spatial-only block index
   *  and an in-block index into a subspace index. */
  std::vector<size_t> combine(const std::string& spin_block,
                              const std::vector<size_t>& block_index_spatial,
                              const std::vector<size_t>& inblock_index) const;

  //
  // Index mapping between subspace index and host program index
  //
  // TODO For these operations it would be nice to have the reverse as well.
  //      This is in general not so easy (since the host index might live in a
  //      different subspace) and probably requires some extra data structures
  //      and bookkeeping data inside MoSpaces as well. Maybe it is, however,
  //      best to implement the inverse operation as a free function. where
  //      the correct subspace is part of the data such a function would return.
  //
  /** Map an index given in the space passed upon construction to the corresponding index
   * in the HF provider, which was used to start off the adcc calculation */
  std::vector<size_t> hf_provider_index_of(const std::vector<size_t>& index) const;

  /** Map a range of indices to host program indices, i.e. the indexing convention
   * used in the HfProvider / HartreeFockSolution_i classes.
   *
   * Since the mapping between subspace and host program indices might not be contiguous,
   * a list of range pairs is returned. In each pair the first entry represents a
   * range of indices (indexed in the MO subspace) and the second entry represents
   * the equivalent range of indices in the Hartree-Fock provider these are mapped to.
   */
  std::vector<RangeMapping> map_range_to_hf_provider(const SimpleRange& range) const;

  //
  // TODO A generalisation of the map_range_to_hf_provider function could be helpful
  //      if a CVS approximation should be added to or removed from a ReferenceState,
  //      where some blocks of the ERI tensor are already imported (to avoid
  //      recomputing the AO->MO wherever possible). This, however, requires
  //      for sure some kind of arbitrary index mapping with potentially
  //      non-contiguous ranges of indices for those blocks of a tensor,
  //      which are split due to a new MO subspace being formed or a subspace
  //      boundary being removed.
  //
 private:
  std::shared_ptr<const MoSpaces> m_mospaces_ptr;

  /** Parsed lot of subspaces (i.e. spaces string split into subspaces) */
  std::vector<std::string> m_subspaces;

  /** Shape of the indices */
  std::vector<size_t> m_shape;

  /** Number of alpha blocks in each axis */
  std::vector<size_t> m_n_blocks_alpha;
};

///@}
}  // namespace libadcc
