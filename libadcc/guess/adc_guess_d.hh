#pragma once

#include "index_group_d.hh"

namespace libadcc {

/** \brief Forms a list of doubles guess vectors.

    Selects the smallest elements from the provided OV matrices (for Koopman's
    guess this should be the delta Fock matrix) and combines two of these
    elements to form the doubles guesses.

      \param va List of doubles-value pairs to initialize.
      \param d1 \f$ o_1v_1 \f$ matrix to construct guesses from
      \param d2 \f$ o_2v_2 \f$ matrix to construct guesses from.
      \param sym Symmetry of guess vectors.
      \param ab Alpha/beta spin blocks of occupied orbitals.
      \return Number of guess vectors created
 **/
size_t adc_guess_d(std::list<std::pair<libtensor::btensor<4, double>*, double>>& va,
                   libtensor::btensor_i<2, double>& d1,
                   libtensor::btensor_i<2, double>& d2,
                   const libtensor::symmetry<4, double>& sym,
                   const libtensor::sequence<4, std::vector<bool>*>& ab, int dm_s,
                   double degeneracy_tolerance);

}  // namespace libadcc
