#include "index_group_d.hh"
#include "../exceptions.hh"
#include <cmath>

namespace libadcc {

using namespace libtensor;
using libtensor::index;

libtensor::mask<4> index_group_d::get_spin_mask(size_t sp) const {
  if (m_s.count(sp) == 0) {
    throw runtime_error("Could not find spin state sp ==" + std::to_string(sp) + ".");
  }
  return compute_spin_mask(sp);
}

size_t index_group_d::compute_spin(const mask<4>& spm) {

  size_t s = 0;
  for (size_t i = 0; i < 4; i++) s = s * 2 + (spm[i] ? 1 : 0);

  return s;
}

mask<4> index_group_d::compute_spin_mask(size_t sp) {

  mask<4> m;
  size_t i = 0, curbit = 1 << 3;
  while (sp != 0 && i < 4) {
    m[i++] = (sp & curbit);
    curbit >>= 1;
  }
  return m;
}

void index_group_map_d::add_index(double val, mask<4> spm, index<4> spidx, index<4> idx) {

  find_canonical_index(spm, spidx, idx);

  // Loop over group map and look for similar value
  std::multimap<double, index_group_d>::iterator it = m_idxmap.begin();
  for (; it != m_idxmap.end(); it++) {
    if (fabs(val - it->first) < m_thresh) break;
  }

  // Try to add element to index groups which belong to similar
  // values
  bool added = false;
  while (it != m_idxmap.end() && fabs(val - it->first) < m_thresh && !added) {

    index_group_d& grp = it->second;
    if (spidx == grp.get_spatial_bidx() && idx == grp.get_idx()) {
      grp.add(spm);
      added = true;
    }
    it++;
  }

  // If no index group found start a new one.
  if (!added) {
    std::multimap<double, index_group_d>::iterator ic = m_idxmap.insert(
          std::pair<double, index_group_d>(val, index_group_d(spidx, idx)));
    ic->second.add(spm);
  }
}

void index_group_map_d::find_canonical_index(mask<4>& m, index<4>& spidx,
                                             index<4>& idx) const {

  if (m_sym_o) {
    if (spidx[0] == spidx[1]) {
      if (idx[0] > idx[1]) {
        std::swap(idx[0], idx[1]);
        std::swap(m[0], m[1]);
      }
    } else if (spidx[0] > spidx[1]) {
      std::swap(spidx[0], spidx[1]);
      std::swap(idx[0], idx[1]);
      std::swap(m[0], m[1]);
    }
  }

  if (m_sym_v) {
    if (spidx[2] == spidx[3]) {
      if (idx[2] > idx[3]) {
        std::swap(idx[2], idx[3]);
        std::swap(m[2], m[3]);
      }
    } else if (spidx[2] > spidx[3]) {
      std::swap(spidx[2], spidx[3]);
      std::swap(idx[2], idx[3]);
      std::swap(m[2], m[3]);
    }
  }
}

}  // namespace libadcc
