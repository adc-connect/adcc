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
#include "AdcMemory.hh"
#include "HartreeFockSolution_i.hh"
#include "MoSpaces.hh"
#include "Tensor.hh"
#include "Timer.hh"

namespace adcc {
/**
 *  \defgroup ReferenceObjects Objects importing and setting up the reference state
 *                             for adcc
 */
///@{

/** Struct containing data about the reference state on top of which the ADC calculation
 * is performed */
class ReferenceState {
 public:
  /* Setup a ReferenceState object using a MoSpaces pointer
   *
   * \param hfsoln_ptr    Pointer to the Interface to the host program,
   *                      providing the HartreeFockSolution data, which
   *                      will be provided by this object.
   * \param mo_ptr        MoSpaces object containing info about the MoSpace setup
   *                      and the point group symmetry.
   * \param symmetry_check_on_import   Should symmetry of the imported objects be checked
   *                      explicitly during the import process. This massively slows
   *                      down the import process and has a dramatic impact on memory
   *                      usage and should thus only be used for debugging import routines
   *                      from the host programs. Do not enable this unless you know
   *                      that you really want to.
   *
   *  \note    This constructor is kept protected, because inconsistencies between the
   *           passed hfsoln_ptr and the passed mo_ptr can lead to tons of issues when
   *           using the constructed object.
   */
  ReferenceState(std::shared_ptr<const HartreeFockSolution_i> hfsoln_ptr,
                 std::shared_ptr<const MoSpaces> mo_ptr, bool symmetry_check_on_import);

  //
  // Info about the reference state
  //
  /** Return whether the reference is restricted or not */
  bool restricted() const { return m_mo_ptr->restricted; }

  /** Return the spin multiplicity of the reference state. 0 indicates that the spin
   * cannot be determined or is not integer (e.g. UHF) */
  size_t spin_multiplicity() const;

  /** Flag to indicate whether this is a cvs reference state */
  bool has_core_occupied_space() const { return m_mo_ptr->has_core_occupied_space(); }

  /** Ground state irreducible representation */
  std::string irreducible_representation() const;

  /** Return the MoSpaces to which this ReferenceState is set up */
  std::shared_ptr<const MoSpaces> mospaces_ptr() const { return m_mo_ptr; }

  /** Return the number of orbitals */
  size_t n_orbs() const { return m_mo_ptr->n_orbs(); }

  /** Return the number of alpha orbitals */
  size_t n_orbs_alpha() const { return m_mo_ptr->n_orbs_alpha(); }

  /** Return the number of beta orbitals */
  size_t n_orbs_beta() const { return m_mo_ptr->n_orbs_beta(); }

  /** Return the number of alpha electrons */
  size_t n_alpha() const { return m_n_alpha; }

  /** Return the number of beta electrons */
  size_t n_beta() const { return m_n_beta; }

  /** Return the nuclear contribution to the cartesian multipole moment
   *  (in standard ordering, i.e. xx, xy, xz, yy, yz, zz) of the given order. */
  std::vector<scalar_type> nuclear_multipole(size_t order) const;

  /** Return the SCF convergence tolerance */
  double conv_tol() const { return m_hfsoln_ptr->conv_tol(); }

  /** Final total SCF energy */
  double energy_scf() const { return m_hfsoln_ptr->energy_scf(); }

  /** String identifying the backend used for the computation */
  std::string backend() const { return m_hfsoln_ptr->backend(); }

  //
  // Tensor data
  //
  /** Return the orbital energies corresponding to the provided space */
  std::shared_ptr<Tensor> orbital_energies(const std::string& space) const;

  /** Return the molecular orbital coefficients corresponding to the provided space
   *  (alpha and beta coefficients are returned) */
  std::shared_ptr<Tensor> orbital_coefficients(const std::string& space) const;

  /** Return the alpha molecular orbital coefficients corresponding
   *  to the provided space */
  std::shared_ptr<Tensor> orbital_coefficients_alpha(const std::string& space) const;

  /** Return the beta molecular orbital coefficients corresponding
   *  to the provided space */
  std::shared_ptr<Tensor> orbital_coefficients_beta(const std::string& space) const;

  /** Return the fock object of the context corresponding to the provided space. */
  std::shared_ptr<Tensor> fock(const std::string& space) const;

  /** Return the ERI (electron-repulsion integrals) contained in the context
   *  corresponding to the provided space. */
  std::shared_ptr<Tensor> eri(const std::string& space) const;

  /** Normally the class only imports the Fock matrix blocks and
   *  electron-repulsion integrals of a particular
   *  space combination when this is requested by a call to above fock() or
   *  eri() functions.
   *
   *  This function call, however, instructs the class to immediately import *all*
   *  ERI tensors.
   *
   *  \note This typically does not need to be called explicitly.
   */
  void import_all() const;

  /** Return the list of momentarily cached fock matrix blocks */
  std::vector<std::string> cached_fock_blocks() const;

  /** Set the list of momentarily cached fock matrix blocks
   * \note This function is both capable to drop blocks
   *       (to save memory) and to request extra caching. */
  void set_cached_fock_blocks(std::vector<std::string> newlist);

  /** Return the list of momentarily cached eri tensor blocks */
  std::vector<std::string> cached_eri_blocks() const;

  /** Set the list of momentarily cached ERI tensor blocks
   * \note This function is both capable to drop blocks
   *       (to save memory) and to request extra caching. */
  void set_cached_eri_blocks(std::vector<std::string> newlist);

  /** Tell the contained HartreeFockSolution_i object, that a larger amount of
   *  ERI data has just been computed and that the next request to further ERI
   *  tensor objects will potentially take some time, such that intermediate
   *  caches can now be flushed to save memory or other resources. */
  void flush_hf_cache() const { m_hfsoln_ptr->flush_cache(); }

  /** Obtain the timing info contained in this object */
  const Timer& timer() const { return m_timer; }

 private:
  /** Import orbital energies into this object */
  void import_orbital_energies(const HartreeFockSolution_i& hf,
                               const std::shared_ptr<const MoSpaces>& mo_ptr,
                               bool symmetry_check_on_import);

  /** Import orbital coefficients into this object */
  void import_orbital_coefficients(const HartreeFockSolution_i& hf,
                                   const std::shared_ptr<const MoSpaces>& mo_ptr,
                                   bool symmetry_check_on_import);

  size_t m_n_alpha;  //< Number of alpha electrons
  size_t m_n_beta;   //< Number of beta electrons

  /** Map containing the orben tensors */
  std::map<std::string, std::shared_ptr<Tensor>> m_orben;

  /** Map containing the orbcoeff tensors */
  std::map<std::string, std::shared_ptr<Tensor>> m_orbcoeff;

  /** Map containing the alpha orbcoeff tensors */
  std::map<std::string, std::shared_ptr<Tensor>> m_orbcoeff_alpha;

  /** Map containing the beta orbcoeff tensors */
  std::map<std::string, std::shared_ptr<Tensor>> m_orbcoeff_beta;

  /** Map containing the fock matrices */
  mutable std::map<std::string, std::shared_ptr<Tensor>> m_fock;

  /** Map containing the electron-repulsion integral tensors */
  mutable std::map<std::string, std::shared_ptr<Tensor>> m_eri;

  /** The pointer to the underlying hfsoln object */
  std::shared_ptr<const HartreeFockSolution_i> m_hfsoln_ptr;

  /** The pointer to the MoSpace data this ReferenceState uses */
  std::shared_ptr<const MoSpaces> m_mo_ptr;

  /** Should symmetry be checked when importing tensor data (ERI, Fock) */
  bool m_symmetry_check_on_import;

  /** Timer to time various import events */
  mutable Timer m_timer;
};

///@}
}  // namespace adcc
