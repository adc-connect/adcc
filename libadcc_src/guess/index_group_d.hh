#pragma once
// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/libtensor.h>
#pragma GCC visibility pop
#include <map>
#include <set>

namespace libadcc {

/** \brief Group of 4D block tensor elements with common spatial index

    An index group is constructed by passing the spatial block index and
    in-block index of a 4D block tensor element. Those two indexes define
    the index group.
    The spin states belonging to the index group can be added using the
    functions
    \code
    void add(const libtensor::mask<4> &);
    \endcode
    \code
    void add(size_t);
    \endcode
 **/
class index_group_d {
 public:
  typedef std::set<size_t>::const_iterator iterator;

  enum {
    aaaa = 0,
    aaab = 1,
    aaba = 2,
    aabb = 3,
    abaa = 4,
    abab = 5,
    abba = 6,
    abbb = 7,
    baaa = 8,
    baab = 9,
    baba = 10,
    babb = 11,
    bbaa = 12,
    bbab = 13,
    bbba = 14,
    bbbb = 15
  };

 private:
  libtensor::index<4> m_spidx;
  libtensor::index<4> m_idx;
  std::set<size_t> m_s;

 public:
  /** \brief Constructor
      \param spidx Spatial block index.
      \param idx In-block index
   **/
  index_group_d(const libtensor::index<4>& spidx, const libtensor::index<4>& idx)
        : m_spidx(spidx), m_idx(idx) {}

  /** \brief Add spin state to index group
      \param s Spin states index (see enum)
   **/
  void add(size_t s) { m_s.insert(s); }

  /** \brief Add spin state to index group
      \param spm Mask representing the spin states (beta == true)
   **/
  void add(const libtensor::mask<4>& spm) { add(compute_spin(spm)); }

  /** \brief Return in-block index of index group
   **/
  const libtensor::index<4>& get_idx() const { return m_idx; }

  /** \brief Return spatial block index of index group
   **/
  const libtensor::index<4>& get_spatial_bidx() const { return m_spidx; }

  /** \brief Check if the spin state exists for index group
   **/
  bool has_spin_state(size_t sp) const { return m_s.find(sp) != m_s.end(); }

  /** \brief Check if the spin state exists for index group
   **/
  bool has_spin_state(const libtensor::mask<4>& spm) const {
    return has_spin_state(compute_spin(spm));
  }

  /** \brief Return the number of spin states
   **/
  size_t size() const { return m_s.size(); }

  /** \brief STL-style iterator to the start of the list of spin states
   **/
  iterator begin() const { return m_s.begin(); }

  /** \brief STL-style iterator to the end of the list of spin states
   **/
  iterator end() const { return m_s.end(); }

  /** \brief Get current spin state
   **/
  size_t get_spin_state(iterator it) const { return *it; }

  /** \brief Get spin state as mask
   **/
  libtensor::mask<4> get_spin_mask(size_t sp) const;

  /** \brief Get current spin state as mask
   **/
  libtensor::mask<4> get_spin_mask(iterator it) const { return get_spin_mask(*it); }

 private:
  static size_t compute_spin(const libtensor::mask<4>& spm);
  static libtensor::mask<4> compute_spin_mask(size_t sp);
};

/** \brief Map of (value, index group) pairs

     \sa adc_guess_d, adc_guess_d
 **/
class index_group_map_d {
 public:
  typedef std::multimap<double, index_group_d>::const_iterator iterator;

 private:
  bool m_sym_o, m_sym_v;  //!< Permutational anti-symmetry of occ / vir indexes
  double m_thresh;        //!< Threshold for identical values

  std::multimap<double, index_group_d> m_idxmap;

 public:
  /** \brief Constructor

      \param thresh Threshold for identical values
      \param sym_o Occ. indexes have perm. anti-symmetry
      \param sym_v Vir. indexes have perm. anti-symmetry
   */
  index_group_map_d(double thresh, bool sym_o = true, bool sym_v = true)
        : m_sym_o(sym_o), m_sym_v(sym_v), m_thresh(thresh) {}

  /** \brief Remove all elements from list
   **/
  void clear() { m_idxmap.clear(); }

  /** \brief Add an index to the map
      \param val Value assigned to the index
      \param spm Spin state mask
      \param spidx Spatial block index
      \param idx In-block index
   **/
  void add_index(double val, libtensor::mask<4> spm, libtensor::index<4> spidx,
                 libtensor::index<4> idx);

  /** \brief STL-style iterator to first element
   **/
  iterator begin() const { return m_idxmap.begin(); }

  /** \brief STL-style iterator to end
   **/
  iterator end() const { return m_idxmap.end(); }

  /** \brief Return the value at the current position
   **/
  double get_value(iterator it) const { return it->first; }

  /** \brief Return the index group at the current position
   **/
  const index_group_d& get_group(iterator it) const { return it->second; }

 private:
  void find_canonical_index(libtensor::mask<4>& m, libtensor::index<4>& spidx,
                            libtensor::index<4>& idx) const;
};

}  // namespace libadcc
