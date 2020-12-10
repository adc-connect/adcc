#include "adc_guess_d.hh"
#include "../exceptions.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/block_tensor/btod_dotprod.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/block_tensor/btod_select.h>
#include <libtensor/block_tensor/btod_set_elem.h>
#include <libtensor/libtensor.h>
#include <libtensor/symmetry/symmetry_element_set_adapter.h>
#pragma GCC visibility pop

namespace libadcc {

// TODO This file definitely needs a cleanup.

using namespace libtensor;
using libtensor::index;

/** \brief Element type for guess vectors
 **/
template <size_t N>
struct guess_element {
  libtensor::index<N> bidx;  //!< Block index
  libtensor::index<N> idx;   //!< In block index
  double coeff;              //!< Coefficient;

  guess_element(const libtensor::index<N>& bidx_, const libtensor::index<N>& idx_,
                const double& coeff_)
        : bidx(bidx_), idx(idx_), coeff(coeff_) {}
};

/** \brief Base class for guess formation **/
template <size_t N>
class index_handler {
 public:
 protected:
  libtensor::sequence<N, std::vector<bool>*> m_ab;  //!< Alpha-beta block markers

 private:
  libtensor::sequence<N, size_t> m_na;  //!< Number ofalpha spin blocks

 public:
  /** \brief Constructor
      \param ab Alpha-beta block markers (for N orbital spaces)
      \param sym Symmetry of guess vectors
      \param ms Spin multiplicity
   **/
  index_handler(const libtensor::sequence<N, std::vector<bool>*>& ab)
        : m_ab(ab), m_na(0) {
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < m_ab[i]->size(); j++) {
        if (!m_ab[i]->at(j)) m_na[i]++;
      }
    }
  }

  /** \brief Calculates the spin projection \f$ m_s \f$ of the block.
      \param bidx Block index
      \param orb_type  Orbital type per dim (true = occupied)
      \return 0 for spin conserving excitations,
          -2 or +2 for spin-flip excitations with even N


   */
  int get_spin_proj(const libtensor::mask<N>& orb_type,
                    const libtensor::index<N>& bidx) const {
    for (size_t i = 0; i < N; i++) {
      if (bidx[i] > m_ab[i]->size()) {
        throw runtime_error("Block index exceeds dim");
      }
    }

    int ms = 0;
    for (size_t i = 0; i < N; i++) {
      if (orb_type[i] == m_ab[i]->at(bidx[i]))
        ms += 1;
      else
        ms -= 1;
    }
    return ms;
  }

  /** \brief Split block index into spatial part and spin part
      \param bidx Input block index
      \param sp Spin index (alpha = false, beta = true)
      \param sbidx Spatial block index
   **/
  void split_block_index(const libtensor::index<N>& bidx, libtensor::mask<N>& sp,
                         libtensor::index<N>& sbidx) const {
    for (size_t i = 0; i < N; i++) {
      sp[i]    = m_ab[i]->at(bidx[i]);
      sbidx[i] = (sp[i] ? bidx[i] - m_na[i] : bidx[i]);
    }
  }

  /** \brief Merge spatial part and spin part of block index
      \param sp Spin index (alpha = false, beta = true)
      \param sbidx Spatial block index
      \param bidx Input block index
   **/
  void merge_block_index(const libtensor::mask<N>& sp, const libtensor::index<N>& sbidx,
                         libtensor::index<N>& bidx) const {
    for (size_t i = 0; i < N; i++) {
      bidx[i] = (sp[i] ? sbidx[i] + m_na[i] : sbidx[i]);
    }
  }
};

namespace {
typedef libtensor::compare4min compare_t;
typedef libtensor::btod_select<2, compare_t>::list_type list2d_t;
typedef libtensor::btod_select<4, compare_t>::list_type list4d_t;
typedef std::list<std::pair<libtensor::btensor<4, double>*, double>> list_t;

/** Determine if occupied and virtual indexes should be symmetrized */
void determine_sym(const symmetry<4, double>& sym, bool& sym_o, bool& sym_v) {
  sym_o = false;
  sym_v = false;
  for (symmetry<4, double>::iterator it1 = sym.begin(); it1 != sym.end(); it1++) {

    const symmetry_element_set<4, double>& set = sym.get_subset(it1);
    const std::string& id                      = set.get_id();

    if (id.compare(se_perm<4, double>::k_sym_type) != 0) continue;
    if (set.is_empty()) return;

    typedef symmetry_element_set_adapter<4, double, se_perm<4, double>> adapter_t;

    adapter_t ad(set);
    for (adapter_t::iterator it2 = ad.begin(); it2 != ad.end(); it2++) {

      const se_perm<4, double>& el = ad.get_elem(it2);

      const permutation<4>& p = el.get_perm();
      sym_o |= (p[0] == 1 && p[1] == 0);
      sym_v |= (p[2] == 3 && p[3] == 2);
    }
  }
}

/** Determine the spin of the guess vectors from symmetry */
unsigned determine_spin(const symmetry<4, double>& sym) {
  for (symmetry<4, double>::iterator it1 = sym.begin(); it1 != sym.end(); it1++) {

    const symmetry_element_set<4, double>& set = sym.get_subset(it1);
    const std::string& id                      = set.get_id();

    if (id.compare(se_part<4, double>::k_sym_type) != 0) continue;
    if (set.is_empty()) return 0;

    typedef symmetry_element_set_adapter<4, double, se_part<4, double>> adapter_t;

    adapter_t ad(set);
    for (adapter_t::iterator it2 = ad.begin(); it2 != ad.end(); it2++) {

      const se_part<4, double>& el = ad.get_elem(it2);

      const dimensions<4>& pdims = el.get_pdims();
      if (pdims[0] != 2 || pdims[1] != 2 || pdims[2] != 2 || pdims[3] != 2) continue;

      index<4> i1, i2;
      i2[0] = 1;
      i2[1] = 1;
      i2[2] = 1;
      i2[3] = 1;
      if (!el.map_exists(i1, i2)) continue;

      if (el.get_transf(i1, i2).get_coeff() == 1.0)
        return 1;
      else
        return 3;
    }
  }
  return 0;
}

void transfer_elements(const list2d_t& ov1, const list2d_t& ov2, index_group_map_d& to,
                       const libtensor::symmetry<4, double>& sym,
                       const index_handler<4>& base, int dm_s) {

  // Determine symmetry
  bool sym_o;  // Are the two occupied indexes identical
  bool sym_v;  // Are the two virtual indexes identical
  determine_sym(sym, sym_o, sym_v);

  to.clear();

  dimensions<4> bidims = sym.get_bis().get_block_index_dims();

  for (list2d_t::const_iterator ita = ov1.begin(); ita != ov1.end(); ita++) {

    for (list2d_t::const_iterator itb = ov2.begin(); itb != ov2.end(); itb++) {

      // Discard element combinations which are not allowed due to the
      // permutational symmetry!!!
      const index<2>& bidxa = ita->get_block_index();
      const index<2>& idxa  = ita->get_in_block_index();
      const index<2>& bidxb = itb->get_block_index();
      const index<2>& idxb  = itb->get_in_block_index();

      if (sym_o && bidxa[0] == bidxb[0] && idxa[0] == idxb[0]) continue;
      if (sym_v && bidxa[1] == bidxb[1] && idxa[1] == idxb[1]) continue;

      double value = ita->get_value() + itb->get_value();

      libtensor::index<4> bidx, idx;
      bidx[0] = bidxa[0];
      bidx[1] = bidxb[0];
      bidx[2] = bidxa[1];
      bidx[3] = bidxb[1];
      idx[0]  = idxa[0];
      idx[1]  = idxb[0];
      idx[2]  = idxa[1];
      idx[3]  = idxb[1];

      if (sym_o && bidx[0] > bidx[1]) {
        std::swap(bidx[0], bidx[1]);
        std::swap(idx[0], idx[1]);
      } else if (sym_o && bidx[0] == bidx[1] && idx[0] > idx[1]) {
        std::swap(idx[0], idx[1]);
      }
      if (sym_v && bidx[2] > bidx[3]) {
        std::swap(bidx[2], bidx[3]);
        std::swap(idx[2], idx[3]);
      } else if (sym_v && bidx[2] == bidx[3] && idx[2] > idx[3]) {
        std::swap(idx[2], idx[3]);
      }

      // Ignore blocks where the targeted spin_change is not achieved
      mask<4> orb_type;
      orb_type[0] = true;
      orb_type[1] = true;
      if (base.get_spin_proj(orb_type, bidx) != dm_s) continue;

      // Check if the block is allowed in the symmetry of the guess
      orbit<4, double> orb(sym, bidx);
      if (!orb.is_allowed()) continue;

      // Find canonical index
      abs_index<4> abi(orb.get_acindex(), bidims);
      const tensor_transf<4, double>& tr = orb.get_transf(bidx);
      bidx                               = abi.get_index();
      permutation<4> pinv(tr.get_perm(), true);
      idx.permute(pinv);

      // Split block index into spin part and spatial part
      mask<4> spm;
      index<4> spi;
      base.split_block_index(bidx, spm, spi);

      to.add_index(value, spm, spi, idx);
    }  // for itb
  }    // for ita
}

size_t build_guesses(list_t::iterator& cur_guess, list_t::iterator end,
                     const index_group_d& ig, double value,
                     const symmetry<4, double>& sym, index_handler<4>& base) {
  bool sym_o;  // Are the two occupied indexes identical
  bool sym_v;  // Are the two virtual indexes identical
  determine_sym(sym, sym_o, sym_v);
  const unsigned spin = determine_spin(sym);  // Spin of the symmetry

  if (cur_guess == end) return 0;

  const index<4>& spidx = ig.get_spatial_bidx();
  const index<4>& idx   = ig.get_idx();

  std::vector<std::list<guess_element<4>>> lv;

  // No specific spin create as many guesses as there are available in the
  // index group
  if (spin == 0) {
    lv.resize(ig.size());

    // Reform full block indexes
    size_t i = 0;
    std::vector<index<4>> bidx(ig.size());
    for (index_group_d::iterator it = ig.begin(); it != ig.end(); it++, i++) {
      base.merge_block_index(ig.get_spin_mask(it), spidx, bidx[i]);
    }

    if (ig.size() == 2) {

      static double coeff[2] = {1.0, 1.0};
      for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++)
          lv[j].push_back(guess_element<4>(bidx[i], idx, coeff[j]));
        coeff[1] *= -1.0;
      }
    } else if (ig.size() == 6) {
      static const double coeff[6][6] = {{2.0, 1.0, 1.0, 1.0, 1.0, 2.0},    // singlet 1
                                         {0.0, 1.0, -1.0, -1.0, 1.0, 0.0},  // singlet 2
                                         {1.0, 0.0, 0.0, 0.0, 0.0, -1.0},   // triplet 1
                                         {0.0, 1.0, 1.0, -1.0, -1.0, 0.0},  // triplet 2
                                         {0.0, 1.0, -1.0, 1.0, -1.0, 0.0},  // triplet 3
                                         {1.0, -1.0, -1.0, -1.0, -1.0, 1.0}};  // quintet

      for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++)
          lv[i].push_back(guess_element<4>(bidx[j], idx, coeff[i][j]));
      }
    } else {
      // Form spin elements
      for (size_t i = 0; i < ig.size(); i++)
        lv[i].push_back(guess_element<4>(bidx[i], idx, 1.0));
    }
  }
  // else decide how many to create based on the spatial indexes
  else if (spin == 1) {
    if ((sym_o && spidx[0] == spidx[1] && idx[0] == idx[1]) ||
        (sym_v && spidx[2] == spidx[3] && idx[2] == idx[3])) {

      lv.resize(1);

      index<4> bidx;
      mask<4> sp;
      sp[0] = sp[2] = false;
      sp[1] = sp[3] = true;
      base.merge_block_index(sp, spidx, bidx);  // abab
      lv[0].push_back(guess_element<4>(bidx, idx, 1.0));

      if (spidx[0] != spidx[1] && idx[0] != idx[1]) {

        sp[1] = sp[2] = false;
        sp[0] = sp[3] = true;
        base.merge_block_index(sp, spidx, bidx);  // baab
        lv[0].push_back(guess_element<4>(bidx, idx, -1.0));
      } else if (spidx[2] != spidx[3] && idx[2] != idx[3]) {

        sp[0] = sp[3] = false;
        sp[1] = sp[2] = true;
        base.merge_block_index(sp, spidx, bidx);  // abba
        lv[0].push_back(guess_element<4>(bidx, idx, -1.0));
      }
    } else {
      lv.resize(2);
      mask<4> sp;
      index<4> bidx;
      sp[0] = sp[1] = sp[2] = sp[3] = false;
      base.merge_block_index(sp, spidx, bidx);  // aaaa
      lv[0].push_back(guess_element<4>(bidx, idx, 2.0));

      sp[0] = sp[2] = false;
      sp[1] = sp[3] = true;
      base.merge_block_index(sp, spidx, bidx);  // abab
      lv[0].push_back(guess_element<4>(bidx, idx, 1.0));
      lv[1].push_back(guess_element<4>(bidx, idx, 1.0));

      sp[0] = sp[3] = false;
      sp[1] = sp[2] = true;
      base.merge_block_index(sp, spidx, bidx);  // abba
      lv[0].push_back(guess_element<4>(bidx, idx, 1.0));
      lv[1].push_back(guess_element<4>(bidx, idx, -1.0));

      sp[1] = sp[3] = false;
      sp[0] = sp[2] = true;
      base.merge_block_index(sp, spidx, bidx);  // baba
      lv[0].push_back(guess_element<4>(bidx, idx, 1.0));
      lv[1].push_back(guess_element<4>(bidx, idx, 1.0));

      sp[1] = sp[2] = false;
      sp[0] = sp[3] = true;
      base.merge_block_index(sp, spidx, bidx);  // baab
      lv[0].push_back(guess_element<4>(bidx, idx, 1.0));
      lv[1].push_back(guess_element<4>(bidx, idx, -1.0));
    }
  } else if (spin == 3) {
    if ((sym_o && spidx[0] == spidx[1] && idx[0] == idx[1]) &&
        (sym_v && spidx[2] == spidx[3] && idx[2] == idx[3]))
      return 0;

    if ((sym_o && spidx[0] == spidx[1] && idx[0] == idx[1]) ||
        (sym_v && spidx[2] == spidx[3] && idx[2] == idx[3])) {

      lv.resize(1);
      mask<4> sp;
      index<4> bidx;
      sp[0] = sp[2] = false;
      sp[1] = sp[3] = true;
      base.merge_block_index(sp, spidx, bidx);  // abab
      lv[0].push_back(guess_element<4>(bidx, idx, 1.0));

      if (spidx[0] != spidx[1] && idx[0] != idx[1]) {

        sp[1] = sp[2] = false;
        sp[0] = sp[3] = true;
        base.merge_block_index(sp, spidx, bidx);  // baab
        lv[0].push_back(guess_element<4>(bidx, idx, 1.0));
      } else if (spidx[2] != spidx[3] && idx[2] != idx[3]) {
        sp[0] = sp[3] = false;
        sp[1] = sp[2] = true;
        base.merge_block_index(sp, spidx, bidx);  // abba
        lv[0].push_back(guess_element<4>(bidx, idx, 1.0));
      }
    } else {
      lv.resize(3);
      mask<4> sp;
      index<4> bidx;
      sp[0] = sp[1] = sp[2] = sp[3] = false;
      base.merge_block_index(sp, spidx, bidx);  // aaaa
      lv[0].push_back(guess_element<4>(bidx, idx, 1.0));

      sp[0] = sp[2] = false;
      sp[1] = sp[3] = true;
      base.merge_block_index(sp, spidx, bidx);  // abab
      lv[1].push_back(guess_element<4>(bidx, idx, 1.0));
      lv[2].push_back(guess_element<4>(bidx, idx, 1.0));

      sp[0] = sp[3] = false;
      sp[1] = sp[2] = true;
      base.merge_block_index(sp, spidx, bidx);  // abba
      lv[1].push_back(guess_element<4>(bidx, idx, -1.0));
      lv[2].push_back(guess_element<4>(bidx, idx, 1.0));

      sp[1] = sp[3] = false;
      sp[0] = sp[2] = true;
      base.merge_block_index(sp, spidx, bidx);  // baba
      lv[1].push_back(guess_element<4>(bidx, idx, -1.0));
      lv[2].push_back(guess_element<4>(bidx, idx, -1.0));

      sp[1] = sp[2] = false;
      sp[0] = sp[3] = true;
      base.merge_block_index(sp, spidx, bidx);  // baab
      lv[1].push_back(guess_element<4>(bidx, idx, 1.0));
      lv[2].push_back(guess_element<4>(bidx, idx, -1.0));
    }
  }

  size_t i = 0;
  for (; i < lv.size() && cur_guess != end; i++, cur_guess++) {
    libtensor::btensor<4, double>& bt = *(cur_guess->first);
    {  // Setup up the symmetry
      libtensor::block_tensor_wr_ctrl<4, double> ctrl(bt);
      ctrl.req_zero_all_blocks();
      libtensor::symmetry<4, double>& sym_to = ctrl.req_symmetry();
      libtensor::so_copy<4, double>(sym).perform(sym_to);
    }

    // Set the elements
    libtensor::btod_set_elem<4> set_op;
    for (auto it = lv[i].begin(); it != lv[i].end(); it++) {
      set_op.perform(bt, it->bidx, it->idx, it->coeff);
    }

    // Normalise
    double norm = libtensor::btod_dotprod<4>(bt, bt).calculate();
    if (norm != 1.0) {
      libtensor::btod_scale<4>(bt, 1.0 / sqrt(norm)).perform();
    }
    cur_guess->second = value;
  }
  return i;
}

}  // namespace

size_t adc_guess_d(std::list<std::pair<libtensor::btensor<4, double>*, double>>& va,
                   libtensor::btensor_i<2, double>& d1,
                   libtensor::btensor_i<2, double>& d2,
                   const libtensor::symmetry<4, double>& sym,
                   const libtensor::sequence<4, std::vector<bool>*>& ab, int dm_s,
                   double degeneracy_tolerance) {

  size_t nguesses = va.size();
  if (nguesses == 0) return 0;

  // TODO sym_o and sym_v should be stored in an adc_guess_base-like object
  // Determine symmetry and spin
  bool sym_o;  // Are the two occupied indexes identical
  bool sym_v;  // Are the two virtual indexes identical
  determine_sym(sym, sym_o, sym_v);
  const unsigned spin = determine_spin(sym);  // Spin of the symmetry

  size_t ns = nguesses;
  index_group_map_d igm(degeneracy_tolerance, sym_o, sym_v);

  bool max_reached = false;
  // Create empty 2d symmetry to use with btod_select
  symmetry<2, double> sym1(d1.get_bis()), sym2(d2.get_bis());

  // Search for smallest elements until we have found enough.
  size_t size = 0;
  while (size < nguesses && !max_reached) {

    igm.clear();
    size = 0;

    ns *= 2;
    list2d_t ilx, ily;
    btod_select<2, compare_t>(d1, sym1).perform(ilx, ns);
    btod_select<2, compare_t>(d2, sym2).perform(ily, ns);

    max_reached = ilx.size() < ns;

    index_handler<4> base(ab);
    transfer_elements(ilx, ily, igm, sym, base, dm_s);
    ilx.clear();

    // Count the number of elements
    if (spin == 0) {
      for (index_group_map_d::iterator it = igm.begin(); it != igm.end(); it++)
        size += igm.get_group(it).size();
    } else if (spin == 1) {
      for (index_group_map_d::iterator it = igm.begin(); it != igm.end(); it++) {
        const index<4>& idx = igm.get_group(it).get_idx();
        if ((sym_o && idx[0] == idx[1]) || (sym_v && idx[2] == idx[3]))
          size++;
        else
          size += 2;
      }
    } else if (spin == 3) {
      for (index_group_map_d::iterator it = igm.begin(); it != igm.end(); it++) {

        const index<4>& idx = igm.get_group(it).get_idx();

        if ((sym_o && idx[0] == idx[1]) && (sym_v && idx[2] == idx[3])) continue;
        if ((sym_o && idx[0] == idx[1]) || (sym_v && idx[2] == idx[3]))
          size++;
        else
          size += 3;
      }
    }
  }  // while

  // Now form the guess vectors
  nguesses = 0;

  auto guess = va.begin();
  // Loop until list is empty or we have constructed all guesses
  index_group_map_d::iterator it = igm.begin();
  while (it != igm.end() && guess != va.end()) {
    index_handler<4> base(ab);
    nguesses +=
          build_guesses(guess, va.end(), igm.get_group(it), igm.get_value(it), sym, base);
    it++;
  }

  return nguesses;
}
}  // namespace libadcc
