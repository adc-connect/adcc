//
// Copyright (C) 2020 by the adcc authors
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
#include "Tensor.hh"
#include "TensorImpl/ExpressionTree.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/libtensor.h>
#pragma GCC visibility pop

namespace libadcc {
/**
 *  \defgroup TensorLibtensor libtensor backend
 */
///@{

/** Actual implementation of the Tensor class. For details about the functions see
 * Tensor.hh */
template <size_t N>
class TensorImpl : public Tensor {
 public:
  using Tensor::dot;
  using Tensor::export_to;
  using Tensor::import_from;

  /** Construction of the tensor implementation class
   *
   * \note This constructor is for experts only. Non-experts should use
   *       the make_tensor method instead.
   */
  TensorImpl(std::shared_ptr<const AdcMemory> adcmem_ptr, std::vector<AxisInfo> axes,
             std::shared_ptr<libtensor::btensor<N, scalar_type>> libtensor_ptr = nullptr,
             std::shared_ptr<ExpressionTree> expr_ptr                          = nullptr);
  TensorImpl(std::shared_ptr<const AdcMemory> adcmem_ptr, std::vector<AxisInfo> axes,
             std::shared_ptr<ExpressionTree> expr_ptr)
        : TensorImpl(adcmem_ptr, axes, nullptr, expr_ptr){};

  std::shared_ptr<Tensor> empty_like() const override;
  std::shared_ptr<Tensor> nosym_like() const override;
  void set_mask(std::string mask, scalar_type value) override;
  std::shared_ptr<Tensor> diagonal(std::vector<size_t> axes) override;
  void set_random() override;

  std::shared_ptr<Tensor> scale(scalar_type c) const override;
  std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> other) const override;
  std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> other) const override;
  std::shared_ptr<Tensor> divide(std::shared_ptr<Tensor> other) const override;

  void add_linear_combination(
        std::vector<scalar_type> scalars,
        std::vector<std::shared_ptr<Tensor>> tensors) const override;

  std::shared_ptr<Tensor> copy() const override;
  std::shared_ptr<Tensor> transpose(std::vector<size_t> axes) const override;
  std::shared_ptr<Tensor> direct_sum(std::shared_ptr<Tensor> other) const override;
  TensorOrScalar tensordot(
        std::shared_ptr<Tensor> other,
        std::pair<std::vector<size_t>, std::vector<size_t>> axes) const override;
  std::vector<scalar_type> dot(
        std::vector<std::shared_ptr<Tensor>> tensors) const override;
  std::shared_ptr<Tensor> symmetrise(
        const std::vector<std::vector<size_t>>& permutations) const override;
  std::shared_ptr<Tensor> antisymmetrise(
        const std::vector<std::vector<size_t>>& permutations) const override;
  double trace(std::string contraction) const override;

  void evaluate() const override;
  bool needs_evaluation() const override { return m_expr_ptr != nullptr; }
  void set_immutable() override { libtensor_ptr()->set_immutable(); }
  bool is_mutable() const override { return !libtensor_ptr()->is_immutable(); }

  void set_element(const std::vector<size_t>& tidx, scalar_type value) override;
  scalar_type get_element(const std::vector<size_t>& tidx) const override;
  bool is_element_allowed(const std::vector<size_t>& tidx) const override;
  std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_absmax(
        size_t n, bool unique_by_symmetry) const override;
  std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_absmin(
        size_t n, bool unique_by_symmetry) const override;
  std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_max(
        size_t n, bool unique_by_symmetry) const override;
  std::vector<std::pair<std::vector<size_t>, scalar_type>> select_n_min(
        size_t n, bool unique_by_symmetry) const override;

  void export_to(scalar_type* memptr, size_t size) const override;
  void import_from(const scalar_type* memptr, size_t size, scalar_type tolerance,
                   bool symmetry_check) override;
  virtual void import_from(
        std::function<void(const std::vector<std::pair<size_t, size_t>>&, scalar_type*)>
              generator,
        scalar_type tolerance, bool symmetry_check) override;
  std::string describe_symmetry() const override;
  std::string describe_expression(std::string stage = "unoptimised") const override;

  /** Convert object to btensor for use in libtensor functions. */
  explicit operator libtensor::btensor<N, scalar_type>&() { return *libtensor_ptr(); }

  /** Return inner btensor object
   *
   * \note This is an advanced function. Use with care, since this can corrupt the
   *       internal state of the Tensor object.
   */
  std::shared_ptr<libtensor::btensor<N, scalar_type>> libtensor_ptr() {
    return static_cast<const TensorImpl<N>*>(this)->libtensor_ptr();
  }

  /** Return permutation, expression object and required keep-alives to this tensor
   *
   * \note This is an advanced function. Use with care, since this can corrupt the
   *       internal state of the Tensor object.
   */
  std::shared_ptr<ExpressionTree> expression_ptr() const;

 protected:
  /** Check that the internally cached state agrees with the one from the contained
   * libtensor pointer. */
  void check_state() const;

  std::shared_ptr<libtensor::btensor<N, scalar_type>> libtensor_ptr() const {
    evaluate();
    return m_libtensor_ptr;
  }

  /** Replace the internal state by an evaluated tensor */
  void reset_state(
        std::shared_ptr<libtensor::btensor<N, scalar_type>> libtensor_ptr) const;

  /** Replace the internal state by expression tree */
  void reset_state(std::shared_ptr<ExpressionTree> expr) const;

  // Tensor data and data for lazy evaluation:
  //     Either: m_libtensor_ptr points to an actual in-memory pointer
  //     Or:     m_expr_ptr contains an expression tree to be evaluated
  mutable std::shared_ptr<libtensor::btensor<N, scalar_type>> m_libtensor_ptr;
  mutable std::shared_ptr<ExpressionTree> m_expr_ptr;
};

/** Extractor function to convert the contained tensor from adcc::Tensor to
 *  an expression.
 *  \note This is an internal function. Use only if you know what you are doing.
 */
std::shared_ptr<ExpressionTree> as_expression(const std::shared_ptr<Tensor>& tensor);

/** Extractor functions to convert the contained tensor of an adcc::Tensor object to
 * an appropriate libtensor::btensor, The dimensionality N has to match */
template <size_t N>
libtensor::btensor<N, scalar_type>& as_btensor(const std::shared_ptr<Tensor>& tensor);

/** Extractor functions to convert the contained tensor of an adcc::Tensor object to
 * an appropriate libtensor::btensor, The dimensionality N has to match */

/** Extractor function to convert the contained tensor of an adcc::Tensor object to
 *  an appropriate pointer to libtensor::btensor. The dimensionality N has to match. */
template <size_t N>
std::shared_ptr<libtensor::btensor<N, scalar_type>> as_btensor_ptr(
      const std::shared_ptr<Tensor>& tensor);

//@{
/**
 * \name Shortcuts for as_btensor */
inline libtensor::btensor<1, scalar_type>& asbt1(const std::shared_ptr<Tensor>& tensor) {
  return as_btensor<1>(tensor);
}
inline libtensor::btensor<2, scalar_type>& asbt2(const std::shared_ptr<Tensor>& tensor) {
  return as_btensor<2>(tensor);
}
inline libtensor::btensor<3, scalar_type>& asbt3(const std::shared_ptr<Tensor>& tensor) {
  return as_btensor<3>(tensor);
}
inline libtensor::btensor<4, scalar_type>& asbt4(const std::shared_ptr<Tensor>& tensor) {
  return as_btensor<4>(tensor);
}
//@}

///@}
}  // namespace libadcc
