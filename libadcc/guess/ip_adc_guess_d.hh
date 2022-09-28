#pragma once

#include "index_group_p2h.hh"

namespace libadcc {

/** \brief Forms a list of doubles guess vectors.

    Selects the smallest elements from the provided OV matrices (for Koopman's
    guess this should be the delta Fock matrix) and combines two of these
    elements to form the doubles guesses.

      \param va List of doubles-value pairs to initialize.
      \param d_o matrix to construct guesses from (occ.)
      \param d_v matrix to construct guesses from. (virt.)
      \param sym Symmetry of guess vectors.
      \param a_spin If alpha ionization (false: beta)
      \param restricted Is this a restricted calculation
      \param ab Alpha/beta spin blocks of occupied orbitals.
      \param dm_s Delta m_s, spin-change
      \return Number of guess vectors created
 **/
size_t ip_adc_guess_d(std::list<std::pair<libtensor::btensor<3, double>*, double>>& va,
                      libtensor::btensor_i<1, double>& d_o,
                      libtensor::btensor_i<1, double>& d_v,
                      const libtensor::symmetry<3, double>& sym,
                      bool a_spin, bool restricted,
                      const libtensor::sequence<3, std::vector<bool>*>& ab, int dm_s,
                      double degeneracy_tolerance);

}  // namespace libadcc