//
// Copyright (C) 2017 by the adcc authors
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
#include "config.hh"
#include <cstddef>
#include <string>

namespace libadcc {
/** Access to a Hartree-Fock solution where all quantities are
 *  available in the *MO* orbital basis.
 *
 * This is the interface an SCF code needs to supply in order to be useful for libadcc.
 *
 * To explain the index ordering, we will refer to a calculation with six basis
 * functions, labelled from b1 to b6 and four spatial molecular orbitals.
 * In order of increasing energy these will be labelled 1a to 4a for the orbitals
 * with alpha spin and 1b to 4b the orbitals with beta spin. We take n_alpha == 2
 * and n_beta == 2, such that 1a, 2a, 1b and 2b are occupied and the others are
 * virtual.
 *
 * In general **b** indices run over the basis functions, i.e. [b1, b2, ..., b6]
 * and **f** indices over the mos, blocked by alpha and beta, i.e.
 * [1a, 2a, 3a, 4a, 1b, 2b, 3b, 4b].
 */
class HartreeFockSolution_i {
 public:
  /** \name System information */
  ///@{
  /** Fill a buffer with nuclear multipole data for the nuclear multipole of
   *  given order. */
  virtual void nuclear_multipole(size_t order, scalar_type* buffer,
                                 size_t size) const = 0;
  //@}

  /** \name Sizes of the data */
  ///@{
  /** Return the number of HF *spin* molecular orbitals of alpha spin.
   *  It is assumed the same number of beta spin orbitals are obtained.
   *  In our example: 4
   **/
  virtual size_t n_orbs_alpha() const = 0;

  /** Return the total number of computed HF *spin* molecular orbitals
   *
   * \note equals n_alpha_orbs() + n_beta_orbs()
   */
  size_t n_orbs() const { return 2 * n_orbs_alpha(); }

  /** Return the number of *spatial* one electron basis functions
   *
   * In our example: 6
   * */
  virtual size_t n_bas() const = 0;
  ///@}

  /** \name Access to HF SCF results */
  ///@{
  /** Provide access to a descriptive string, which identifies the implementation /
   *  SCF program used to provide the data */
  virtual std::string backend() const = 0;

  /** SCF convergence threshold */
  virtual real_type conv_tol() const = 0;

  /** Was the HF algorithm a restricted HF
   *
   * If this is true adcc will assume that the alpha and beta values
   * of certain quantities (like the MO coefficients) are exactly
   * identical.
   * */
  virtual bool restricted() const = 0;

  /** What is the spin multiplicity of the reference state
   *  represented by this class.
   *
   *  A special value of 0 indicates that the spin is not known
   *  or an unrestricted calculation was employed
   */
  virtual size_t spin_multiplicity() const = 0;

  /** \brief Fill a buffer with the HF molecular orbital coefficients.
   *
   * After the function call the buffer contains the coefficient
   * matrix as a full n_orbs() times n_bas() block in
   * row-major ordering.
   *
   * The indexing convention for f and b is as explained above,
   * i.e. the first index runs like [1a 2a 3a 4a 1b 2b 3b 4b]
   * and the second like [b1 b2 b3 b4 b5 b6] in the example
   * explained above.
   *
   * @param buffer        The pointer into the memory to fill
   * @param size          The maximal number of elements the buffer
   *                      pointer is valid for.
   */
  virtual void orbcoeff_fb(scalar_type* buffer, size_t size) const = 0;

  /** \brief Fill a buffer with the HF molecular orbital energies
   *  ordered increasing in energy but blocked by alpha and beta,
   *  i.e. in our example we have [1a 2a 3a 4a 1b 2b 3b 4b]
   *
   * After the function call the buffer contains the energies
   * as a full n_orbs() block.
   *
   * @param buffer        The pointer into the memory to fill
   * @param size          The maximal number of elements the buffer
   *                      pointer is valid for.
   */
  virtual void orben_f(scalar_type* buffer, size_t size) const = 0;

  /** Occupation numbers of each of the spin orbitals.
   *  Only the values 1.0 and 0.0 at each of the entries are supported.
   *  After the function call the buffer contains the occupation numbers
   *  as a full n_orbs() block.
   *
   *  In our example: [1.0 1.0 1.0 1.0 0.0 0.0 1.0 1.0 1.0 1.0 0.0 0.0]
   *
   * @param buffer        The pointer into the memory to fill
   * @param size          The maximal number of elements the buffer
   *                      pointer is valid for.
   */
  virtual void occupation_f(scalar_type* buffer, size_t size) const = 0;

  /** Return the final, total SCF energy.
   *
   * This energy should contain the nuclear repulsion energy contribution.
   */
  virtual real_type energy_scf() const = 0;
  ///@}

  /** \name Access to quantities in an MO basis */
  ///@{
  /** Fill a buffer with a *block* of the fock matrix elements
   *  in the **MO** basis.
   *
   * The block to access is specified by the means of half-open ranges
   * in each of the two index dimensions as [``d1_start``, `d1_end``)
   * as well as [``d2_start``, ``d2_end``).
   *
   * The ordering is such that the ``n_orbs_alpha()`` orbitals of
   * alpha spin go first in each of the dimensions followed by the
   * ``n_orbs_alpha()`` beta orbitals. Otherwise the ordering
   * should be identical to the one of ``orbital_energies_f()``
   * and ``coeff_fb()``.
   * I.e. in our example ``f`` runs like [1a 2a 3a 4a 1b 2b 3b 4b]
   * in both dimensions.
   *
   * The buffer is used to return the computed values. The storage
   * format is determined by d1_stride and d2_stride, which define
   * the strides to be used to write data in each dimension.
   *
   * The theoretical full fock matrix which could be returned is hence of
   * size n_orbs() times n_orbs(). The implementation should not
   * assume that the alpha-beta and beta-alpha blocks are not
   * accessed even though they are zero by spin symmetry.
   *
   * \note For a canonical basis this thing is diagonal with the orbital
   * energies, but for non-canonical ADC it might not be the case.
   */
  virtual void fock_ff(size_t d1_start, size_t d1_end, size_t d2_start, size_t d2_end,
                       size_t d1_stride, size_t d2_stride, scalar_type* buffer,
                       size_t size) const = 0;

  /** Fill a buffer with a *block* of the two electron repulsion integrals
   *  in the **MO** basis.
   *
   * The block is specified by the means of half-open ranges in each of
   * the four indices using the parameters ``d1_start``, ``d1_end``,
   * ``d2_start``, ``d2_end``, ``d3_start``, ``d3_end``, ``d4_start``,
   * ``d4_end``, for example
   * ```
   * std::vector<double> buffer;
   * hf.repulsion_integrals_ffff(0,5, 0,5, 0,5, 0,5, buffer.data());
   * ```
   * will return the block of integrals running from indices 0-4 in each of
   * the indices, i.e. a buffer of 5^4 = 625 entries.
   *
   * Within each dimension the ordering is assumed to be the same as
   * ``orbital_energies_f()`` and ``coeff_fb()``, i.e. the
   * ``n_orbs_alpha()`` orbitals of alpha spin go first, followed by the
   * ``n_orbs_alpha()`` beta orbitals.
   * I.e. in our example ``f`` runs like [1a 2a 3a 4a 1b 2b 3b 4b]
   * in both dimensions.
   *
   * The buffer is used to return the computed values. The storage
   * format is determined by d1_stride to d3_stride, which define
   * the strides to be used to write data in each dimension.
   *
   * \note The convention used for the indexing in this function is
   * *chemist's notation* or *shell-pair notation*, i.e. if ``(ij|kl)`` is
   * the element (i,j,k,l) of the buffer, then this describes the integral
   * \f[
   *	\int_\Omega \int_\Omega d r_1 d r_2 \frac{\phi_i(r_1) \phi_j(r_1)
   *	\phi_k(r_2) \phi_l(r_2)}{|r_1 - r_2|}
   * \f]
   * such that orbital indices i/j and k/l are on the same centre.
   *
   * \note If `has_eri_phys_asym_ffff()` is true, this function is never called directly
   *       and may implemented only as a dummy.
   */
  virtual void eri_ffff(size_t d1_start, size_t d1_end, size_t d2_start, size_t d2_end,
                        size_t d3_start, size_t d3_end, size_t d4_start, size_t d4_end,
                        size_t d1_stride, size_t d2_stride, size_t d3_stride,
                        size_t d4_stride, scalar_type* buffer, size_t size) const = 0;

  /** Fill a buffer with a block of the *antisymmetrised* two electron repulsion integrals
   *  in the MO basis using the *physicist's indexing convention*.
   *
   *  This function works similar to ``repulsion_integrals_ffff`` but returns the
   *  anti-symmetrised integrals in *physicist's notation* instead. In other words
   *  if <ij||kl> is the term
   * \f[
   *	\int_\Omega \int_\Omega d r_1 d r_2 \frac{\phi_i(r_1) \phi_j(r_2)
   *	\phi_k(r_1) \phi_l(r_2)}{|r_1 - r_2|}
   *	- \int_\Omega \int_\Omega d r_1 d r_2 \frac{\phi_i(r_1) \phi_j(r_2)
   *	\phi_l(r_1) \phi_k(r_2)}{|r_1 - r_2|}
   * \f]
   *
   * \note If `has_eri_phys_asym_ffff()` is false, this function is never called directly
   *       and may implemented only as a dummy.
   */
  virtual void eri_phys_asym_ffff(size_t d1_start, size_t d1_end, size_t d2_start,
                                  size_t d2_end, size_t d3_start, size_t d3_end,
                                  size_t d4_start, size_t d4_end,  //
                                  size_t d1_stride, size_t d2_stride, size_t d3_stride,
                                  size_t d4_stride, scalar_type* buffer,
                                  size_t size) const = 0;

  /** Has the eri_phys_asym_ffff function been implemented and is available */
  virtual bool has_eri_phys_asym_ffff() const { return false; }

  /** Tell the implementation side, that potentially cached data could now be flushed
   *  to save memory or other resources. This can be used to purge e.g. intermediates
   *  for the computation of eri tensor data.
   */
  virtual void flush_cache() const {}

  /** Virtual desctructor */
  virtual ~HartreeFockSolution_i() {}
};
}  // namespace libadcc
