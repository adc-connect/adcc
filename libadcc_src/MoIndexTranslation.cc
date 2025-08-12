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

#include "MoIndexTranslation.hh"
#include "exceptions.hh"
#include "shape_to_string.hh"

namespace libadcc {

namespace {
std::vector<std::string> split_subspaces(std::shared_ptr<const MoSpaces> mospaces_ptr,
                                         const std::string& space) {

  std::vector<std::string> subspaces;
  const std::vector<std::string>& valid_subspaces = mospaces_ptr->subspaces;
  for (size_t i = 0; i < space.size(); ++i) {
    if (space[i] == 'f') {
      subspaces.push_back("f");
    } else {
      std::string cur = space.substr(i, 2);
      if (std::find(valid_subspaces.begin(), valid_subspaces.end(), cur) ==
          valid_subspaces.end()) {
        throw invalid_argument(
              "Encountered invalid subspace identifier " + cur +
              " while parsing the space identifier " + space +
              ". Check that the space identifier is sound, meaning that all subspaces or "
              "spaces are contained inside the MoSpaces object.");
      }
      subspaces.push_back(cur);

      // Advance by two characters in the for loop instead of one:
      i += 1;
    }
  }  // for
  return subspaces;
}

/** Increase the dimensionality of the mapping,
 *  by providing the way the ext axis should be mapped.
 *
 *  The function will automatically take care of building products of the
 *  mappings, i.e. the present object (this) is a 2D mapping, where the
 *  axis is split into two ranges and the axis to be added is also
 *  added into two ranges, the resulting RangeMapping will be two-dimensional
 *  and contain *four* elements covering the 4 respective blocks of consecutive
 *  indices in the HfProvider indexing */
std::vector<RangeMapping> mapping_add_axis(std::vector<RangeMapping> original,
                                           const std::vector<AxisRange> axis_sources,
                                           const std::vector<AxisRange> axis_targets) {
  std::vector<RangeMapping> ret;

  // If the original mapping is a completely empty object,
  // then we need at least a dummy RangeMapping for the loop
  // below to do the right thing.
  if (original.empty()) {
    original.push_back(RangeMapping{});
  }

  // Amend each existing mapping by adding the
  // source -> target mapping of this iteration
  // in an extra axis and push back the result.
  for (const RangeMapping& elem : original) {
    for (size_t i = 0; i < axis_sources.size(); ++i) {
      const AxisRange& source = axis_sources[i];
      const AxisRange& target = axis_targets[i];

      RangeMapping mapping(elem);
      mapping.from().push_axis(source);
      mapping.to().push_axis(target);
      ret.push_back(mapping);
    }
  }

  return ret;
}
}  // namespace

std::vector<size_t> SimpleRange::starts() const {
  std::vector<size_t> ret;
  ret.reserve(size());
  for (auto& elem : *this) ret.push_back(elem.first);
  return ret;
}

std::vector<size_t> SimpleRange::lasts() const {
  std::vector<size_t> ret;
  ret.reserve(size());
  for (auto& elem : *this) {
    if (elem.second > 0) {
      ret.push_back(elem.second - 1);
    } else {
      ret.push_back(0);
    }
  }
  return ret;
}

std::vector<size_t> SimpleRange::ends() const {
  std::vector<size_t> ret;
  ret.reserve(size());
  for (auto& elem : *this) ret.push_back(elem.second);
  return ret;
}

MoIndexTranslation::MoIndexTranslation(std::shared_ptr<const MoSpaces> mospaces_ptr,
                                       const std::vector<std::string>& subspaces)
      : m_mospaces_ptr(mospaces_ptr),
        m_subspaces{subspaces},
        m_shape{},
        m_n_blocks_alpha{} {
  const std::vector<std::string>& valid_subspaces = mospaces_ptr->subspaces;
  for (const std::string& ss : subspaces) {
    if (ss == "f") continue;
    if (std::find(valid_subspaces.begin(), valid_subspaces.end(), ss) ==
        valid_subspaces.end()) {
      throw invalid_argument("Encountered invalid subspace identifier " + ss + ".");
    }
  }

  m_shape.resize(m_subspaces.size());
  for (size_t i = 0; i < m_subspaces.size(); ++i) {
    m_shape[i] = m_mospaces_ptr->n_orbs(m_subspaces[i]);
  }

  m_n_blocks_alpha.resize(m_subspaces.size());
  for (size_t idim = 0; idim < m_subspaces.size(); ++idim) {
    const std::string& ss          = m_subspaces[idim];
    const std::vector<char>& spins = m_mospaces_ptr->map_block_spin.at(ss);
    m_n_blocks_alpha[idim] =
          static_cast<size_t>(std::count(spins.begin(), spins.end(), 'a'));
  }
}

MoIndexTranslation::MoIndexTranslation(std::shared_ptr<const MoSpaces> mospaces_ptr,
                                       const std::string& space)
      : MoIndexTranslation(mospaces_ptr, split_subspaces(mospaces_ptr, space)) {}

std::string MoIndexTranslation::space() const {
  std::string ret;
  ret.reserve(2 * m_subspaces.size());
  for (auto& ss : m_subspaces) ret.append(ss);
  return ret;
}

std::vector<size_t> MoIndexTranslation::full_index_of(
      const std::vector<size_t>& /*index*/) const {
  throw not_implemented_error("full_index_of not yet implemented.");
}

std::vector<size_t> MoIndexTranslation::block_index_of(
      const std::vector<size_t>& index) const {
  if (index.size() != ndim()) {
    throw dimension_mismatch("MoIndexTranslation is for subspace (" + space() +
                             "), which is of dimension " + std::to_string(ndim()) +
                             ", but passed index has a dimension of " +
                             std::to_string(index.size()) + ".");
  }
  for (size_t i = 0; i < ndim(); ++i) {
    if (index[i] >= m_shape[i]) {
      throw invalid_argument("Passed index " + shape_to_string(index) +
                             " overshoots shape " + shape_to_string(m_shape) +
                             " at dimension " + std::to_string(i) + ".");
    }
  }

  std::vector<size_t> ret(index.size());
  for (size_t idim = 0; idim < index.size(); ++idim) {
    const std::string& ss             = m_subspaces[idim];
    const std::vector<size_t>& starts = m_mospaces_ptr->map_block_start.at(ss);

    for (size_t isp = 0; isp < starts.size(); ++isp) {
      if (starts[isp] > index[idim]) break;
      ret[idim] = isp;
    }
  }
  return ret;
}

std::vector<size_t> MoIndexTranslation::block_index_spatial_of(
      const std::vector<size_t>& index) const {
  std::vector<size_t> ret(ndim());
  std::vector<size_t> block_index = block_index_of(index);
  for (size_t idim = 0; idim < index.size(); ++idim) {
    const std::string& ss = m_subspaces[idim];
    const size_t bidx     = block_index[idim];
    const char spin       = m_mospaces_ptr->map_block_spin.at(ss)[bidx];
    if (spin == 'a') {
      ret[idim] = bidx;
    } else {
      ret[idim] = bidx - m_n_blocks_alpha[idim];
    }
  }
  return ret;
}

std::string MoIndexTranslation::spin_of(const std::vector<size_t>& index) const {
  std::string ret;
  std::vector<size_t> block_index = block_index_of(index);
  for (size_t idim = 0; idim < index.size(); ++idim) {
    const std::string& ss = m_subspaces[idim];
    const size_t bidx     = block_index[idim];
    ret.push_back(m_mospaces_ptr->map_block_spin.at(ss)[bidx]);
  }
  return ret;
}

std::pair<std::vector<size_t>, std::vector<size_t>> MoIndexTranslation::split(
      const std::vector<size_t>& index) const {
  std::vector<size_t> block_index = block_index_of(index);
  std::vector<size_t> inblock_index(block_index.size());
  for (size_t idim = 0; idim < index.size(); ++idim) {
    const std::string& ss             = m_subspaces[idim];
    const std::vector<size_t>& starts = m_mospaces_ptr->map_block_start.at(ss);
    inblock_index[idim]               = index[idim] - starts[block_index[idim]];
  }
  return {block_index, inblock_index};
}

std::tuple<std::string, std::vector<size_t>, std::vector<size_t>>
MoIndexTranslation::split_spin(const std::vector<size_t>& index) const {
  std::vector<size_t> block_spatial(ndim());
  std::string spinstring(ndim(), ' ');
  std::vector<size_t> block_index = block_index_of(index);
  for (size_t idim = 0; idim < index.size(); ++idim) {
    const std::string& ss = m_subspaces[idim];
    const size_t bidx     = block_index[idim];
    spinstring[idim]      = m_mospaces_ptr->map_block_spin.at(ss)[bidx];
    if (spinstring[idim] == 'a') {
      block_spatial[idim] = bidx;
    } else {
      block_spatial[idim] = bidx - m_n_blocks_alpha[idim];
    }
  }
  return {spinstring, block_spatial, inblock_index_of(index)};
}

std::vector<size_t> MoIndexTranslation::combine(
      const std::vector<size_t>& block_index,
      const std::vector<size_t>& inblock_index) const {
  if (block_index.size() != ndim()) {
    throw dimension_mismatch("MoIndexTranslation is for subspace (" + space() +
                             "), which is of dimension " + std::to_string(ndim()) +
                             ", but passed block_index " + shape_to_string(block_index) +
                             " has a dimension of " + std::to_string(block_index.size()) +
                             ".");
  }
  if (inblock_index.size() != ndim()) {
    throw dimension_mismatch("MoIndexTranslation is for subspace (" + space() +
                             "), which is of dimension " + std::to_string(ndim()) +
                             ", but passed inblock_index " +
                             shape_to_string(inblock_index) + " has a dimension of " +
                             std::to_string(inblock_index.size()) + ".");
  }

  std::vector<size_t> ret(ndim());
  for (size_t idim = 0; idim < ndim(); ++idim) {
    const std::string& ss             = m_subspaces[idim];
    const std::vector<size_t>& bstart = m_mospaces_ptr->map_block_start.at(ss);
    const size_t iblk                 = block_index[idim];

    if (iblk >= bstart.size()) {
      throw invalid_argument("Passed block index " + shape_to_string(block_index) +
                             " overshoots number of blocks at dimension " +
                             std::to_string(idim) +
                             " (== " + std::to_string(bstart.size()) + ").");
    }

    const size_t block_size = [this, &iblk, &bstart, &idim] {
      if (iblk == bstart.size() - 1) {
        // Last block has this size:
        return m_shape[idim] - bstart.back();
      } else {
        return bstart[iblk + 1] - bstart[iblk];
      }
    }();

    if (inblock_index[idim] >= block_size) {
      throw invalid_argument("Passed in-block index " + shape_to_string(inblock_index) +
                             " overshoots number of elements in block for axis " +
                             std::to_string(idim) + " (== " + std::to_string(block_size) +
                             ").");
    }

    ret[idim] = bstart[iblk] + inblock_index[idim];
  }
  return ret;
}

std::vector<size_t> MoIndexTranslation::combine(
      const std::string& spin_block, const std::vector<size_t>& block_index_spatial,
      const std::vector<size_t>& inblock_index) const {
  if (block_index_spatial.size() != ndim()) {
    throw dimension_mismatch(
          "MoIndexTranslation is for subspace (" + space() + "), which is of dimension " +
          std::to_string(ndim()) + ", but passed block_index_spatial " +
          shape_to_string(block_index_spatial) + " has a dimension of " +
          std::to_string(block_index_spatial.size()) + ".");
  }
  if (spin_block.size() != ndim()) {
    throw dimension_mismatch("MoIndexTranslation is for subspace (" + space() +
                             "), which is of dimension " + std::to_string(ndim()) +
                             ", but passed spin-block identifier was '" + spin_block +
                             "'.");
  }

  std::vector<size_t> block_index(block_index_spatial.size());
  for (size_t idim = 0; idim < block_index_spatial.size(); ++idim) {
    if (spin_block[idim] == 'a') {
      block_index[idim] = block_index_spatial[idim];
    } else if (spin_block[idim] == 'b') {
      block_index[idim] = block_index_spatial[idim] + m_n_blocks_alpha[idim];
    } else {
      throw invalid_argument(
            "spin-block identifier '" + spin_block +
            "' contains invalid character. Only 'a' and 'b' are allowed.");
    }
  }
  return combine(block_index, inblock_index);
}

std::vector<size_t> MoIndexTranslation::hf_provider_index_of(
      const std::vector<size_t>& index) const {
  if (index.size() != ndim()) {
    throw dimension_mismatch("MoIndexTranslation is for subspace (" + space() +
                             "), which is of dimension " + std::to_string(ndim()) +
                             ", but passed index has a dimension of " +
                             std::to_string(index.size()) + ".");
  }
  for (size_t i = 0; i < ndim(); ++i) {
    if (index[i] >= m_shape[i]) {
      throw invalid_argument("Passed index " + shape_to_string(index) +
                             " overshoots shape " + shape_to_string(m_shape) +
                             " at dimension " + std::to_string(i) + ".");
    }
  }

  std::vector<size_t> ret(ndim());
  for (size_t idim = 0; idim < index.size(); ++idim) {
    const std::string& ss = m_subspaces[idim];
    ret[idim]             = m_mospaces_ptr->map_index_hf_provider.at(ss)[index[idim]];
  }
  return ret;
}

std::vector<RangeMapping> MoIndexTranslation::map_range_to_hf_provider(
      const SimpleRange& range) const {
  if (range.ndim() != ndim()) {
    throw dimension_mismatch("MoIndexTranslation is for subspace (" + space() +
                             "), which is of dimension " + std::to_string(ndim()) +
                             ", but passed index range has a dimension of " +
                             std::to_string(range.ndim()) + ".");
  }

  //
  // Determine the way each axis has to be split into
  // subranges in order for each range to be subject to a continuous mapping,
  // i.e. a mapping where a block of indices in the adcc ordering is mapped
  // to a block of indices in the HfProvider ordering.
  //

  // List of [begin, end) source ranges into which the indices are
  // split for each axis
  std::vector<std::vector<AxisRange>> sources_per_axis;

  // List of [begin, end) target ranges to which the splits of the list above
  // are mapped to in the HfProvider
  std::vector<std::vector<AxisRange>> targets_per_axis;

  for (size_t idim = 0; idim != ndim(); ++idim) {
    const std::string& ss           = m_subspaces[idim];
    const std::vector<size_t>& hmap = m_mospaces_ptr->map_index_hf_provider.at(ss);
    const AxisRange axrange         = range.axis(idim);

    if (axrange.start() > axrange.end()) {
      throw invalid_argument("Passed index range at dimension " + std::to_string(idim) +
                             "(== [" + std::to_string(axrange.start()) + "," +
                             std::to_string(axrange.end()) +
                             ")) is invalid: start larger than end.");
    }
    if (axrange.end() > m_shape[idim]) {
      throw invalid_argument("Passed index range at dimension " + std::to_string(idim) +
                             "(== [" + std::to_string(axrange.start()) + "," +
                             std::to_string(axrange.end()) + ")) overshoots shape " +
                             shape_to_string(m_shape) + ".");
    }

    std::vector<AxisRange> sources;  // Splits for the current axis

    // Buffer to store the HfProvider index to which the
    // previous index from the axrange was mapped to.
    size_t prev_target = 0;
    for (size_t i = axrange.start(); i < axrange.end(); ++i) {
      if (i == axrange.start() || hmap[i] != prev_target + 1) {
        // I.e. either we are dealing with the first index
        // or the mapping is no longer contiguous, so we
        // need to start a new range for this axis.
        sources.push_back({i, i + 1});
      } else {
        // Mapping is still contiguous => just increase the existing range further
        sources.back().end() = i + 1;
      }
      prev_target = hmap[i];
    }
    sources_per_axis.push_back(sources);

    // Build the range of HfProvider target indices for current axis
    std::vector<AxisRange> targets;
    for (const AxisRange& sp : sources) {
      targets.emplace_back(hmap[sp.start()], sp.length() + hmap[sp.start()]);
    }
    targets_per_axis.push_back(targets);
  }

  std::vector<RangeMapping> ret;
  for (size_t idim = 0; idim < ndim(); ++idim) {
    ret = mapping_add_axis(ret, sources_per_axis[idim], targets_per_axis[idim]);
  }
  return ret;
}

}  // namespace libadcc
