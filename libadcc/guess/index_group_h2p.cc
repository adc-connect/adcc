#include "index_group_h2p.hh"
#include "../exceptions.hh"
#include <cmath>

namespace libadcc {

using namespace libtensor;
using libtensor::index;

libtensor::mask<3> index_group_h2p::get_spin_mask(size_t sp) const {
  if (m_s.count(sp) == 0) {
    throw runtime_error("Could not find spin state sp ==" + std::to_string(sp) + ".");
  }
  return compute_spin_mask(sp);
}

size_t index_group_h2p::compute_spin(const mask<3>& spm) {

  size_t s = 0;
  for (size_t i = 0; i < 3; i++) s = s * 2 + (spm[i] ? 1 : 0);

  return s;
}

mask<3> index_group_h2p::compute_spin_mask(size_t sp) {

  mask<3> m;
  size_t i = 0, curbit = 1 << 2;
  while (sp != 0 && i < 3) {
    m[i++] = (sp & curbit);
    curbit >>= 1;
  }
  return m;
}

void index_group_map_h2p::add_index(double val, mask<3> spm, index<3> spidx, index<3> idx) {

  find_canonical_index(spm, spidx, idx);

  // Loop over group map and look for similar value
  std::multimap<double, index_group_h2p>::iterator it = m_idxmap.begin();
  for (; it != m_idxmap.end(); it++) {
    if (fabs(val - it->first) < m_thresh) break;
  }

  // Try to add element to index groups which belong to similar
  // values
  bool added = false;
  while (it != m_idxmap.end() && fabs(val - it->first) < m_thresh && !added) {

    index_group_h2p& grp = it->second;
    if (spidx == grp.get_spatial_bidx() && idx == grp.get_idx()) {
      grp.add(spm);
      added = true;
    }
    it++;
  }

  // If no index group found start a new one.
  if (!added) {
    std::multimap<double, index_group_h2p>::iterator ic = m_idxmap.insert(
          std::pair<double, index_group_h2p>(val, index_group_h2p(spidx, idx)));
    ic->second.add(spm);
  }
}

void index_group_map_h2p::find_canonical_index(mask<3>& m, index<3>& spidx,
                                             index<3>& idx) const {

  if (m_sym_v) {
    if (spidx[1] == spidx[2]) {
      if (idx[1] > idx[2]) {
        std::swap(idx[1], idx[2]);
        std::swap(m[1], m[2]);
      }
    } else if (spidx[1] > spidx[2]) {
      std::swap(spidx[1], spidx[2]);
      std::swap(idx[1], idx[2]);
      std::swap(m[1], m[2]);
    }
  }
}

}  // namespace libadcc