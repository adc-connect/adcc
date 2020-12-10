//
// Copyright (C) 2018 by the adcc authors
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
#include "Symmetry.hh"
#include "config.hh"
#include "exceptions.hh"
#include <array>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

namespace libadcc {
/**
 *  \defgroup Tensor Tensor interface
 */
///@{

class Tensor;

/** Poor-mans variant for a Tensor and a scalar */
struct TensorOrScalar {
  std::shared_ptr<Tensor> tensor_ptr;
  scalar_type scalar;
};

/** The tensor interface used by libadcc */
class Tensor {
 public:
  /** Construct a tensor interface class.
   *
   * @param adcmem_ptr  Memory to use for the tensor.
   * @param axes        Information about the tensor axes.
   */
  Tensor(std::shared_ptr<const AdcMemory> adcmem_ptr, std::vector<AxisInfo> axes);

  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;
  virtual ~Tensor()           = default;
  Tensor(const Tensor&)       = default;
  Tensor& operator=(const Tensor&) = default;

  //@{
  /** \name Interface of a tensor, which is exposed to python.
   */

  /** Number of dimensions */
  size_t ndim() const { return m_axes.size(); }

  /** Shape of each dimension */
  const std::vector<size_t>& shape() const { return m_shape; }

  /** Number of elements */
  size_t size() const { return m_size; }

  /** Subspace labels identifying each axis */
  std::vector<std::string> subspaces() const;

  /** Space to which the tensor is initialised */
  std::string space() const;

  /** Set some flags for playing around */
  const std::vector<std::string>& flags() const { return m_flags; }
  void set_flags(const std::vector<std::string>& new_flags) { m_flags = new_flags; }

  // TODO It would be nice to have this function,
  //      but this would require to store the mospaces or the symmetry
  //      object in this base class (instead of adcmem_ptr) or in the
  //      derived TensorImpl class. This is skipped for now, since
  //      in order to do this properly it requires to keep track of
  //      the spaces after all tensor operations (including contractions
  //      and the alike). At the moment this boils down to parsing
  //      the libtensor symmetry object to an adcc:Symmetry object
  //      from time to time. Just storing an MoSpaces pointer could
  //      already be a good start, is however not enough, since
  //      the "b" AoSpace is not covered by it ...
  //      Also in light of the fact, that we might want to replace
  //      libtensor at some point, it feels to me we are locking
  //      ourselves into their ecosystem to much with implementing this
  //      at the moment and before it is clear in which direction we will
  //      go in the future.
  //
  // /** Return a Symmetry object representing the Symmetry of the tensor
  //  *  \note This represents a copy of the symmetry, i.e. the tensor
  //  *        cannot be modified using this function.
  //  */
  // virtual std::shared_ptr<Symmetry> symmetry_ptr() const = 0;

  /** Return a new tensor with the same dimensionality and symmetry
   *  as the passed tensor. The elements have undefined values. */
  virtual std::shared_ptr<Tensor> empty_like() const = 0;

  /** Return a new tensor with same dimensionality and shape
   *  (and bispace) but without any symmetry copied over */
  virtual std::shared_ptr<Tensor> nosym_like() const = 0;

  /** Return a new tensor with the same dimensionality and symmetry
   *  as the passed tensor and all elements set to zero. */
  std::shared_ptr<Tensor> zeros_like() const;

  /** Return a new tensor with the same dimensionality and symmetry
   *  as the passed tensor and all elements set to one. */
  std::shared_ptr<Tensor> ones_like() const;

  //
  // TODO The next two need to be rethought. Look especially at the shift function
  // in
  // /libtensor/libtensor/expr/operators/set.h
  //
  /** Set a mask into a tensor to a specific value.
   *  The mask to set is defined by the mask string. Repetitive
   *  indices define the values to be set. E.g. for a 6D tensor
   *  the mask string "iijkli" would set all elements T_{iijkli} for all
   *  ijkl to the given value.
   */
  virtual void set_mask(std::string mask, scalar_type value) = 0;

  /** Shift the elements matched by a mask into a tensor by a specific value.
   *  The mask to set is defined by the mask string. Repetitive
   *  indices define the values to be shifted. E.g. for a 6D tensor
   *  the mask string "iijkli" would shift all elements T_{iijkli} for all
   *  ijkl by the given value.
   */
  // map lazily to libtensor shift operation
  // virtual void shift_mask(std::string mask, scalar_type value) = 0;

  /** Fill the tensor with a particular value, adhereing to symmetry. */
  void fill(scalar_type value);

  /** Extract a generalised diagonal from this Tensor. `axes` defines
   *  the axes over which to extract. The resulting new axis is appended
   *  to the remaining axis. For example if we have a Tensor fourth order `t`
   *  then diagonal(1, 2) returns `d_{iax} = t_{ixxa}` and diagonal(0, 1, 2)
   *  returns `d_{ax} = t_{xxxa}`.
   */
  virtual std::shared_ptr<Tensor> diagonal(std::vector<size_t> axes) = 0;

  /** Set the tensor to random data preserving symmetry */
  virtual void set_random() = 0;

  /** Scale tensor by a scalar value and return the result */
  virtual std::shared_ptr<Tensor> scale(scalar_type c) const = 0;

  /** Add another tensor and return the result */
  virtual std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other) const = 0;

  /** Add a linear combination to the represented tensor and *store the result*.
   * Note that unlike `add` and `scale` this operation is *not* lazy and
   * is provided for optimising on cases where a lazy linear combination introduces
   * a too large overhead. */
  virtual void add_linear_combination(
        std::vector<scalar_type> scalars,
        std::vector<std::shared_ptr<Tensor>> tensors) const = 0;

  /** Multiply another tensor elementwise and return the result */
  virtual std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> other) const = 0;

  /** Divide another tensor elementwise and return the result */
  virtual std::shared_ptr<Tensor> divide(std::shared_ptr<Tensor> other) const = 0;

  /** Return a deep copy of this tensor. */
  virtual std::shared_ptr<Tensor> copy() const = 0;

  /** Return a transpose form of this tensor
   *
   * \param axes  New permutation of the dimensions of this tensor (e.g.
   *              (1,0,2,3) will permute dimension 0 and 1 and keep the others)
   *              If missing, just reverse the dimension order.
   */
  virtual std::shared_ptr<Tensor> transpose(std::vector<size_t> axes) const = 0;

  /** Return the tensordot of this tensor with another
   *
   * \param axes  Axes to contract over. The first vector refers to the axes of
   *              this tensor, the second vector to the axes of the other tensor.
   */
  virtual TensorOrScalar tensordot(
        std::shared_ptr<Tensor> other,
        std::pair<std::vector<size_t>, std::vector<size_t>> axes) const = 0;

  /** Compute the Frobenius or l2 inner product with a list of other tensors,
   *  returning the results as a scalar array */
  virtual std::vector<scalar_type> dot(
        std::vector<std::shared_ptr<Tensor>> tensors) const = 0;

  // TODO One *could* implement a partial trace by contracting with a delta matrix
  //      which would be pretty expensive, however, ...
  /** Return the full trace of a tensor
   *
   * \param contraction Indices to contract over using Einstein
   *                    summation convention, e.g. "abab" traces
   *                    over the 1st and 3rd and 2nd and 4th axis.
   */
  virtual double trace(std::string contraction) const = 0;

  /** Compute the direct sum of this tensor with another tensor.
   *  All axes are used in the order of appearance (this and then other).
   */
  virtual std::shared_ptr<Tensor> direct_sum(std::shared_ptr<Tensor> other) const = 0;

  /** Symmetrise with respect to the given index permutations
   *  by adding the elements resulting from appropriate index permutations.
   *
   * \param permutations    The list of permutations to be applied
   *                        *simultaneously*
   *
   * Examples for permutations. Take a rank-4 tensor T_{ijkl}
   *    {{0,1}}             Permute the first two indices, i.e. form
   *                        0.5 * (T_{ijkl} + T_{jikl})
   *    {{0,1}, {2,3}}      Form 0.5 * (T_{ijkl} + T_{jilk})
   *
   * \note Unlike the implementation in libtensor, this function includes the
   *       prefactor 0.5
   */
  virtual std::shared_ptr<Tensor> symmetrise(
        const std::vector<std::vector<size_t>>& permutations) const = 0;

  /** Antisymmetrise with respect to the given index permutations and return the
   *  result. For details with respect to the
   *  format of the permutations, see symmetrise. In this case the output tensor
   *  will be antisymmetric with respect to these permutations.
   *
   * \note Unlike the implementation in libtensor, this function includes the
   *       prefactor 0.5
   */
  virtual std::shared_ptr<Tensor> antisymmetrise(
        const std::vector<std::vector<size_t>>& permutations) const = 0;

  /** Make sure to evaluate this tensor in case this has not yet happened */
  virtual void evaluate() const = 0;

  /** Check weather the object represents an evaluated expression or not. */
  virtual bool needs_evaluation() const = 0;

  /** Flag the tensor as immutable, allows some optimisations to be performed */
  virtual void set_immutable() = 0;

  /** Is the tensor mutable */
  virtual bool is_mutable() const = 0;

  /** Set the value of a single tensor element
   *  \note This is a slow function and should be avoided.
   **/
  virtual void set_element(const std::vector<size_t>& idx, scalar_type value) = 0;

  /** Get the value of a single tensor element
   *  \note This is a slow function and should be avoided.
   * */
  virtual scalar_type get_element(const std::vector<size_t>& tidx) const = 0;

  /** Return whether the element referenced by tidx is allowed (non-zero)
   *  by the symmetry of the Tensor or not.
   */
  virtual bool is_element_allowed(const std::vector<size_t>& tidx) const = 0;

  /** Get the n absolute largest elements along with their values
   *  \param n                     Number of elements to select
   *  \param unique_by_symmetry    By default the returned elements
   *                               are made unique by symmetry of the tensor.
   *                               Setting this to false disables this feature.
   **/
  virtual std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_absmax(
        size_t n, bool unique_by_symmetry = true) const = 0;

  /** Get the n absolute smallest elements along with their values
   *  \param n                     Number of elements to select
   *  \param unique_by_symmetry    By default the returned elements
   *                               are made unique by symmetry of the tensor.
   *                               Setting this to false disables this feature.
   **/
  virtual std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_absmin(
        size_t n, bool unique_by_symmetry = true) const = 0;

  /** Get the n largest elements along with their values
   *  \param n                     Number of elements to select
   *  \param unique_by_symmetry    By default the returned elements
   *                               are made unique by symmetry of the tensor.
   *                               Setting this to false disables this feature.
   **/
  virtual std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_max(
        size_t n, bool unique_by_symmetry = true) const = 0;

  /** Get the n smallest elements along with their values
   *  \param n                     Number of elements to select
   *  \param unique_by_symmetry    By default the returned elements
   *                               are made unique by symmetry of the tensor.
   *                               Setting this to false disables this feature.
   **/
  virtual std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_min(
        size_t n, bool unique_by_symmetry = true) const = 0;

  /** Extract the tensor to plain memory provided by the given pointer.
   *
   *  \note This will return a full, *dense* tensor.
   *        At least size elements of space are assumed at the provided memory location.
   *        The data is stored in row-major (C-like) format
   */
  virtual void export_to(scalar_type* memptr, size_t size) const = 0;

  /** Extract the tensor into a std::vector, which will be resized to fit the data.
   * \note All data is stored in row-major (C-like) format
   */
  virtual void export_to(std::vector<scalar_type>& output) const {
    output.resize(size());
    export_to(output.data(), output.size());
  }

  /** Import the tensor from plain memory provided by the given
   *  pointer. The memory will be copied and all existing data overwritten.
   *  If symmetry_check is true, the process will check that the data has the
   *  required symmetry to fit into the tensor. This requires
   *  a slower algorithm to be chosen.
   *
   *  \param memptr      Full, dense memory pointer to the tensor data to be imported.
   *  \param size        Size of the dense memory.
   *  \param tolerance   Threshold to account for numerical inconsistencies
   *                     when checking the symmetry or for determining zero blocks.
   *  \param symmetry_check  Should symmetry be explicitly checked during the import.
   *
   *  \note This function requires a full, *dense* tensor with the data stored in
   *        row-major (C-like) format.
   */
  virtual void import_from(const scalar_type* memptr, size_t size,
                           scalar_type tolerance = 0, bool symmetry_check = true) = 0;

  /** Import the tensor from plain memory provided by the given
   *  vector. The memory will be copied and all existing data overwritten.
   *  If symmetry_check is true, the process will check that the data has the
   *  required symmetry to fit into the tensor. This requires
   *  a slower algorithm to be chosen.
   *
   *  \param input       Input data in a linearised vector
   *  \param tolerance   Threshold to account for numerical inconsistencies
   *                     when checking the symmetry or for determining zero blocks.
   *  \param symmetry_check  Should symmetry be explicitly checked during the import.
   *
   *  \note This function requires a full, *dense* tensor with the data stored in
   *        row-major (C-like) format.
   */
  virtual void import_from(const std::vector<scalar_type>& input,
                           scalar_type tolerance = 0, bool symmetry_check = true) {
    import_from(input.data(), input.size(), tolerance, symmetry_check);
  }

  /** Import the tensor from a generator functor. All existing data will be overwritten.
   *  If symmetry_check is true, the process will check that the data has the
   *  required symmetry to fit into the tensor. This requires
   *  a slower algorithm to be chosen.
   *
   *  \param generator   Generator functor. The functor is called with a list of ranges
   *                     for each dimension. The ranges are half-open, left-inclusive
   *                     and right-exclusive. The corresponding data should be written
   *                     to the passed pointer into raw memory. This is an advanced
   *                     functionality. Use only if you know what you are doing.
   *  \param tolerance   Threshold to account for numerical inconsistencies
   *                     when checking the symmetry or for determining zero blocks.
   *  \param symmetry_check  Should symmetry be explicitly checked during the import.
   *
   *  \note The generator is required to produce data in row-major (C-like) format
   *        at a designated memory location.
   */
  virtual void import_from(
        std::function<void(const std::vector<std::pair<size_t, size_t>>&, scalar_type*)>
              generator,
        scalar_type tolerance = 0, bool symmetry_check = true) = 0;

  // TODO begin_canonical()
  // TODO end_canonical()
  //      ... loop over canonical blocks

  /**
   * Return a std::string providing hopefully helpful information
   * about the symmetries stored inside the tensor object.
   */
  virtual std::string describe_symmetry() const = 0;

  /**
   * Return a std::string providing hopefully helpful information
   * about the expression tree stored inside the tensor object.
   * Valid values for stage are "unoptimised", "optimised", "evaluation"
   * returning the unoptimised expression tree as stored, an optimised form of it
   * or the tree as used actually for evaluating the expression.
   */
  virtual std::string describe_expression(std::string stage = "unoptimised") const = 0;
  //@}

  /** Return a pointer to the memory keep-alive object */
  std::shared_ptr<const AdcMemory> adcmem_ptr() const { return m_adcmem_ptr; }

  /** Return the axes info for this tensor */
  const std::vector<AxisInfo>& axes() const { return m_axes; }
  //@}

 protected:
  size_t m_size;                 // Cache for number of elements
  std::vector<size_t> m_shape;   // Cache for shape
  std::vector<AxisInfo> m_axes;  //  Information about each of the tensor axes
  std::shared_ptr<const AdcMemory> m_adcmem_ptr;  // Pointer to the adc memory object
  std::vector<std::string> m_flags;               // Just some flags to playing around.
};

//
// Some operators
//

inline std::shared_ptr<Tensor> evaluate(std::shared_ptr<Tensor> in) {
  in->evaluate();
  return in;
}

/** Construct an uninitialised tensor using a Symmetry object */
std::shared_ptr<Tensor> make_tensor(std::shared_ptr<Symmetry> symmetry);

/** Construct an uninitialised tensor using no symmetry at all */
std::shared_ptr<Tensor> make_tensor(std::shared_ptr<const AdcMemory> adcmem_ptr,
                                    std::vector<AxisInfo> axes);
// Note: Unlike most other functions in this header, these functions
//       is only implemented in the file TensorImpl.cc

/** Construct a tensor initialised to zero using a Symmetry object */
std::shared_ptr<Tensor> make_tensor_zero(std::shared_ptr<Symmetry> symmetry);

///@}
}  // namespace libadcc
