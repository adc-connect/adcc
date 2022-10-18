#include "ip_adc_guess_d.hh"
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
  libtensor::sequence<N, size_t> m_na;  //!< Number of alpha spin blocks

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
      \return -1 or +1 for ionization of an alpha or beta electron


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
      // Left side is true if orbital is occ.
      // Right side is true if orbital has beta spin
      // Hence it is true for occ. beta orbitals and virt. alpha orbitals
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
typedef libtensor::btod_select<1, compare_t>::list_type list1d_t;
typedef libtensor::btod_select<3, compare_t>::list_type list3d_t;
typedef std::list<std::pair<libtensor::btensor<3, double>*, double>> list_t;

/** Determine if occupied indices should be symmetrized */
void determine_sym(const symmetry<3, double>& sym, bool& sym_o) {

  sym_o = false;
  for (symmetry<3, double>::iterator it1 = sym.begin(); it1 != sym.end(); it1++) {

    const symmetry_element_set<3, double>& set = sym.get_subset(it1);
    const std::string& id                      = set.get_id();

    if (id.compare(se_perm<3, double>::k_sym_type) != 0) continue;
    if (set.is_empty()) return;

    typedef symmetry_element_set_adapter<3, double, se_perm<3, double>> adapter_t;

    adapter_t ad(set);
    for (adapter_t::iterator it2 = ad.begin(); it2 != ad.end(); it2++) {

      const se_perm<3, double>& el = ad.get_elem(it2);

      const permutation<3>& p = el.get_perm();
      sym_o |= (p[0] == 1 && p[1] == 0);
    }
  }
}

/** Determine the spin of the guess vectors from symmetry */
unsigned determine_spin(bool restricted, bool doublet) {

  if (restricted) {
      if (doublet) return 2;
      else return 4;
  } else {
      return 0;
  }
}

/** Transfers the elements of a 1D list to a 3D list */
void transfer_elements(const list1d_t& o, const list1d_t& v, 
                       index_group_map_p2h& to, const libtensor::symmetry<3, 
                       double>& sym, const index_handler<3>& base, int dm_s) {

  // Determine symmetry (TO MODIFY)
  bool sym_o;  // Are the two occupied indices identical
  determine_sym(sym, sym_o);

  to.clear();

  dimensions<3> bidims = sym.get_bis().get_block_index_dims();

  for (list1d_t::const_iterator ita = o.begin(); ita != o.end(); ita++) {

    for (list1d_t::const_iterator itb = o.begin(); itb != o.end(); itb++) {

      for (list1d_t::const_iterator itc = v.begin(); itc != v.end(); itc++) {

      // Discard element combinations which are not allowed due to the
      // permutational symmetry!!!
      const index<1>& bidxa = ita->get_block_index();
      const index<1>& idxa  = ita->get_in_block_index();
      const index<1>& bidxb = itb->get_block_index();
      const index<1>& idxb  = itb->get_in_block_index();
      const index<1>& bidxc = itc->get_block_index();
      const index<1>& idxc  = itc->get_in_block_index();

      if (sym_o && bidxa[0] == bidxb[0] && idxa[0] == idxb[0]) continue;

      double value = ita->get_value() + itb->get_value() + itc->get_value();

      libtensor::index<3> bidx, idx;
      bidx[0] = bidxa[0];
      bidx[1] = bidxb[0];
      bidx[2] = bidxc[0];
      idx[0]  = idxa[0];
      idx[1]  = idxb[0];
      idx[2]  = idxc[0];

      if (sym_o && bidx[0] > bidx[1]) {
        std::swap(bidx[0], bidx[1]);
        std::swap(idx[0], idx[1]);
      } else if (sym_o && bidx[0] == bidx[1] && idx[0] > idx[1]) {
        std::swap(idx[0], idx[1]);
      }

      // Ignore blocks where the targeted spin_change is not achieved
      mask<3> orb_type;
      orb_type[0] = true;
      orb_type[1] = true;
      if (base.get_spin_proj(orb_type, bidx) != dm_s) continue;

      // Check if the block is allowed in the symmetry of the guess
      orbit<3, double> orb(sym, bidx);
      if (!orb.is_allowed()) continue;

      // Find canonical index
      abs_index<3> abi(orb.get_acindex(), bidims);
      const tensor_transf<3, double>& tr = orb.get_transf(bidx);
      bidx                               = abi.get_index();
      permutation<3> pinv(tr.get_perm(), true);
      idx.permute(pinv);

      // Split block index into spin part and spatial part
      mask<3> spm;
      index<3> spi;
      base.split_block_index(bidx, spm, spi);

      to.add_index(value, spm, spi, idx);
      }  // for itc
    }  // for itb
  }  // for ita
}

size_t build_guesses(list_t::iterator& cur_guess, list_t::iterator end,
                     const index_group_p2h& ig, double value,
                     const symmetry<3, double>& sym, bool a_spin, 
                     bool restricted, bool doublet, index_handler<3>& base) {
  bool sym_o;  // Are the two occupied indices identical
  determine_sym(sym, sym_o);
  const unsigned spin = determine_spin(restricted, doublet);  // Spin of the symmetry

  if (cur_guess == end) return 0;

  int ms = a_spin ? -1 : 1;

  const index<3>& spidx = ig.get_spatial_bidx();
  const index<3>& idx   = ig.get_idx();

  std::vector<std::list<guess_element<3>>> lv;

  // No specific spin create as many guesses as there are available in the
  // index group
  if (spin == 0) {
    lv.resize(ig.size());

    // Reform full block indices
    size_t i = 0;
    std::vector<index<3>> bidx(ig.size());
    for (index_group_p2h::iterator it = ig.begin(); it != ig.end(); it++, i++) {
      base.merge_block_index(ig.get_spin_mask(it), spidx, bidx[i]);
    }

    if (ig.size() == 1) {

      static double coeff[1] = {1.0};
      for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 1; j++)
          lv[j].push_back(guess_element<3>(bidx[i], idx, coeff[j]));
      }
    } else if (ig.size() == 3) {
      static const double coeff[3][3] = {
                // in case of ms == -1 (alpha ionization)
                // aaa   abb   bab
                // and in case of ms == 1 (beta ionization)
                // bbb   baa   aba
                {  1.0, -1.0,  -1.0}, // quartet
                {  0.0, -1.0,  1.0},  // doublet 1
                { -2.0, -1.0, -1.0}}; // doublet 2

        if (ms == -1) { // alpha ionization
          for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) 
              lv[i].push_back(guess_element<3>(bidx[j], idx, coeff[i][j])); 
          }   
        } else { // beta ionization
          for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++)
              lv[i].push_back(guess_element<3>(bidx[2-j], idx, coeff[i][j]));
          }  
        }
    } else {
      // Form spin elements
      for (size_t i = 0; i < ig.size(); i++)
        lv[i].push_back(guess_element<3>(bidx[i], idx, 1.0));
    }
  }
  else if (spin == 2) {
    // Reform full block indices
    size_t i = 0;
    std::vector<index<3>> bidx(ig.size());
    for (index_group_p2h::iterator it = ig.begin(); it != ig.end(); it++, i++) {
      base.merge_block_index(ig.get_spin_mask(it), spidx, bidx[i]);
    }

    if (ig.size() == 1) {
      lv.resize(1);

      static double coeff[1] = {1.0};
      for (size_t i = 0; i < 1; i++) {
        for (size_t j = 0; j < 1; j++)
          lv[j].push_back(guess_element<3>(bidx[i], idx, coeff[j]));
      }
    } else if (ig.size() == 3) {
      lv.resize(2);

      static const double coeff[2][3] = {
                // in case of ms == -1 (alpha ionization)
                // aaa   abb   bab
                // and in case of ms == 1 (beta ionization)
                // bbb   baa   aba
                {  0.0, -1.0,  1.0},  // doublet 1
                { -2.0, -1.0, -1.0}}; // doublet 2

      if (ms == -1) { // alpha ionization
        for (size_t i = 0; i < 2; i++) {
          for (size_t j = 0; j < 3; j++) 
            lv[i].push_back(guess_element<3>(bidx[j], idx, coeff[i][j])); 
        }   
      } else { // beta ionization
        for (size_t i = 0; i < 2; i++) {
          for (size_t j = 0; j < 3; j++)
            lv[i].push_back(guess_element<3>(bidx[2-j], idx, coeff[i][j]));
        }  
      }
    } else {
      // Form spin elements
      lv.resize(ig.size());
      for (size_t i = 0; i < ig.size(); i++) {
        lv[i].push_back(guess_element<3>(bidx[i], idx, 1.0));
      }
    }
  } else if (spin == 4) {
    // Reform full block indices
    size_t i = 0;
    std::vector<index<3>> bidx(ig.size());
    for (index_group_p2h::iterator it = ig.begin(); it != ig.end(); it++, i++) {
      base.merge_block_index(ig.get_spin_mask(it), spidx, bidx[i]);
    }

    if (ig.size() == 3) {
      lv.resize(1);

      static const double coeff[1][3] = {
                  // in case of ms == -1 (alpha ionization)
                  // aaa   abb   bab
                  // and in case of ms == 1 (beta ionization)
                  // bbb   baa   aba
                  {  1.0, -1.0,  -1.0}}; // quartet

      if (ms == -1) { // alpha ionization
        for (size_t i = 0; i < 1; i++) {
          for (size_t j = 0; j < 3; j++) 
            lv[i].push_back(guess_element<3>(bidx[j], idx, coeff[i][j])); 
        }   
      } else { // beta ionization
        for (size_t i = 0; i < 1; i++) {
          for (size_t j = 0; j < 3; j++)
            lv[i].push_back(guess_element<3>(bidx[2-j], idx, coeff[i][j]));
        }  
      }
    } else {
      // Form spin elements
      lv.resize(ig.size());
      for (size_t i = 0; i < ig.size(); i++) {
        lv[i].push_back(guess_element<3>(bidx[i], idx, 1.0));
      }
    }
  }


  size_t i = 0;
  for (; i < lv.size() && cur_guess != end; i++, cur_guess++) {
    libtensor::btensor<3, double>& bt = *(cur_guess->first);
    {  // Setup up the symmetry
      libtensor::block_tensor_wr_ctrl<3, double> ctrl(bt);
      ctrl.req_zero_all_blocks();
      libtensor::symmetry<3, double>& sym_to = ctrl.req_symmetry();
      libtensor::so_copy<3, double>(sym).perform(sym_to);
    }

    // Set the elements
    libtensor::btod_set_elem<3> set_op;
    for (auto it = lv[i].begin(); it != lv[i].end(); it++) {
      set_op.perform(bt, it->bidx, it->idx, it->coeff);
    }

    // Normalise
    double norm = libtensor::btod_dotprod<3>(bt, bt).calculate();
    if (norm != 1.0) {
      libtensor::btod_scale<3>(bt, 1.0 / sqrt(norm)).perform();
    }
    cur_guess->second = value;
  }
  return i;
}

}  // namespace

size_t ip_adc_guess_d(std::list<std::pair<libtensor::btensor<3, double>*, double>>& va,
                      libtensor::btensor_i<1, double>& d_o,
                      libtensor::btensor_i<1, double>& d_v,
                      const libtensor::symmetry<3, double>& sym,
                      bool a_spin, bool restricted, bool doublet,
                      const libtensor::sequence<3, std::vector<bool>*>& ab, 
                      int dm_s, double degeneracy_tolerance) {

  size_t nguesses = va.size();
  if (nguesses == 0) return 0;

  // TODO sym_o should be stored in an adc_guess_base-like object
  // Determine symmetry and spin
  bool sym_o;  // Are the two occupied indices identical
  determine_sym(sym, sym_o);
  const unsigned spin = determine_spin(restricted, doublet);  // Spin of the symmetry

  size_t ns = nguesses;
  index_group_map_p2h igm(degeneracy_tolerance, sym_o);

  bool max_reached = false;
  // Create empty 1d symmetry to use with btod_select
  symmetry<1, double> sym1(d_o.get_bis()), sym2(d_v.get_bis());

  // Search for smallest elements until we have found enough.
  size_t size = 0;
  while (size < nguesses && !max_reached) {

    igm.clear();
    size = 0;

    ns *= 2;
    list1d_t ilx_o, ilx_v;
    btod_select<1, compare_t>(d_o, sym1).perform(ilx_o, ns);
    btod_select<1, compare_t>(d_v, sym2).perform(ilx_v, ns);

    max_reached = ilx_o.size() < ns;

    index_handler<3> base(ab);
    transfer_elements(ilx_o, ilx_v, igm, sym, base, dm_s);
    ilx_o.clear();
    ilx_v.clear();

    //size++; // we want to have the real size of the index group guesses
    // Count the number of elements
    if (spin == 0) {
      for (index_group_map_p2h::iterator it = igm.begin(); it != igm.end(); it++) {
        size += igm.get_group(it).size();
      } 
    } else if (spin == 2) {
      for (index_group_map_p2h::iterator it = igm.begin(); it != igm.end(); it++) {
        if (igm.get_group(it).size() == 3) {
            size += 2; // two doublets, one quartet
        } else if (igm.get_group(it).size() == 1) {
            size += 1; // only a doublet in this case
        }
      }
    } else if (spin == 4) {
      } // quartets not implemented
  }  // while

  // Now form the guess vectors
  nguesses = 0;

  list_t::iterator guess = va.begin();
  // Loop until list is empty or we have constructed all guesses
  index_group_map_p2h::iterator it = igm.begin();
  while (it != igm.end() && guess != va.end()) {
    index_handler<3> base(ab);
    nguesses +=
          build_guesses(guess, va.end(), igm.get_group(it), igm.get_value(it), 
            sym, a_spin, restricted, doublet, base);
    it++;
  }

  return nguesses;
}
}  // namespace libadcc
