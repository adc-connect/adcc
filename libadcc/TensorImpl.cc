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

#include "TensorImpl.hh"
#include "TensorImpl/as_bispace.hh"
#include "TensorImpl/as_lt_symmetry.hh"
#include "TensorImpl/get_block_starts.hh"
#include "shape_to_string.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_dotprod.h>
#include <libtensor/block_tensor/btod_export.h>
#include <libtensor/block_tensor/btod_random.h>
#include <libtensor/block_tensor/btod_select.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/block_tensor/btod_set_diag.h>
#include <libtensor/block_tensor/btod_set_elem.h>
#include <libtensor/symmetry/print_symmetry.h>
#pragma GCC visibility pop

namespace libadcc {
namespace lt = libtensor;

#define DIMENSIONALITY_CHECK(OTHER)                                                    \
  {                                                                                    \
    if (ndim() != OTHER->ndim()) {                                                     \
      throw dimension_mismatch(                                                        \
            "Dimensionality of this tensor (" + std::to_string(ndim()) +               \
            ") does not agree with the dimensionality of the other tensor"             \
            " passed, which has dimensionality " +                                     \
            std::to_string(OTHER->ndim()) + ".");                                      \
    }                                                                                  \
    if (shape() != OTHER->shape()) {                                                   \
      throw dimension_mismatch("Shape of this tensor (" + shape_to_string(shape()) +   \
                               ") does not agree with the shape of the other tensor" + \
                               " passed, which has shape " +                           \
                               shape_to_string(OTHER->shape()) + ".");                 \
    }                                                                                  \
    if (axes() != OTHER->axes()) {                                                     \
      throw dimension_mismatch("Axes of this tensor (" + axes_to_string(axes()) +      \
                               ") do not agree with the axes of the other tensor "     \
                               "passed, which has axis labels " +                      \
                               axes_to_string(OTHER->axes()) + ".");                   \
    }                                                                                  \
  }

namespace {
std::string axes_to_string(const std::vector<AxisInfo>& axes) {
  std::string res = "";
  for (auto& ax : axes) res.append(ax.label);
  return res;
}

/** Build an identity permutation of length n */
std::vector<size_t> identity_permutation(size_t n) {
  std::vector<size_t> permutation;
  for (size_t i = 0; i < n; ++i) {
    permutation.push_back(i);
  }
  return permutation;
}

template <size_t N>
lt::expr::label<N> strip_safe(const std::vector<std::shared_ptr<const lt::letter>>& in) {
  std::vector<const lt::letter*> label_unsafe;
  for (size_t i = 0; i < in.size(); ++i) {
    label_unsafe.push_back(in[i].get());
  }
  return lt::expr::label<N>(label_unsafe);
}

template <size_t N>
std::vector<size_t> extract_expr_permutation(
      const lt::expr::expr_rhs<N, scalar_type>& expr,
      const std::vector<std::shared_ptr<const lt::letter>>& label) {
  std::vector<size_t> permutation;
  lt::permutation<N> perm = expr.get_label().permutation_of(strip_safe<N>(label));
  perm.invert();
  for (size_t i = 0; i < N; ++i) {
    permutation.push_back(perm[i]);
  }
  return permutation;
}

/** Merge two keepalive lists */
std::vector<std::shared_ptr<void>> merge(std::vector<std::shared_ptr<void>> lhs,
                                         const std::vector<std::shared_ptr<void>>& rhs) {
  for (auto& ptr : rhs) lhs.push_back(ptr);
  return lhs;
}

std::vector<std::shared_ptr<const lt::letter>> make_label(size_t n) {
  std::vector<std::shared_ptr<const lt::letter>> ret;
  for (size_t i = 0; i < n; ++i) {
    ret.push_back(std::make_shared<lt::letter>());
  }
  return ret;
}

template <size_t M, size_t N>
std::pair<lt::expr::label<M>, lt::expr::label<M>> parse_permutation(
      const std::vector<AxisInfo>& axes, const lt::expr::label<N>& label,
      const std::vector<std::vector<size_t>>& permutations) {
  std::vector<const lt::letter*> set1;
  std::vector<const lt::letter*> set2;

  std::vector<size_t> processed_indices;
  for (const auto& perm : permutations) {
    if (perm.size() < 2) {
      throw invalid_argument("A permutation tuple has to have 2 or more indices.");
    } else if (perm.size() == 2) {
      if (perm[0] == perm[1]) {
        throw invalid_argument(
              "A permutation tuple cannot have duplicate indices. Here " +
              std::to_string(perm[0]) + " is a duplicate.");
      }
      auto find_0 =
            std::find(processed_indices.begin(), processed_indices.end(), perm[0]);
      auto find_1 =
            std::find(processed_indices.begin(), processed_indices.end(), perm[1]);

      if (find_0 != processed_indices.end() or find_1 != processed_indices.end()) {
        throw invalid_argument(
              "Provided index tuples in a permutation list have to be disjoint.");
      }
      if (perm[0] >= N || perm[1] >= N) {
        throw invalid_argument(
              "Index in permutation list cannot be larger than dimension.");
      }
      if (axes[perm[0]] != axes[perm[1]]) {
        throw invalid_argument(
              "(Anti)-Symmetrisation can only be performed over equivalent axes (not '" +
              axes[perm[0]].label + "' and '" + axes[perm[1]].label + "').");
      }

      set1.push_back(&label.letter_at(perm[0]));
      set2.push_back(&label.letter_at(perm[1]));

      processed_indices.push_back(perm[0]);
      processed_indices.push_back(perm[1]);
    } else {
      throw not_implemented_error(
            "Permutations for tuple length larger 2 not implemented.");
    }
  }
  return {lt::expr::label<M>(set1), lt::expr::label<M>(set2)};
}

template <size_t N, typename T>
std::pair<lt::index<N>, lt::index<N>> assert_convert_tensor_index(
      lt::btensor<N, T>& tensor, const std::vector<size_t>& idx) {
  if (idx.size() != N) {
    throw dimension_mismatch("Tensor is of dimension " + std::to_string(N) +
                             ", but passed index has a dimennsion of " +
                             std::to_string(idx.size()) + ".");
  }
  const lt::dimensions<N>& dims = tensor.get_bis().get_dims();
  for (size_t i = 0; i < N; ++i) {
    if (idx[i] >= dims[i]) {
      throw invalid_argument("Passed index " + shape_to_string(idx) +
                             " overshoots Tensor at dimension " + std::to_string(i) +
                             " (with extent: " + std::to_string(dims[i]) + ")");
    }
  }

  lt::index<N> block_idx;
  for (size_t idim = 0; idim < N; ++idim) {
    // Find splits
    const size_t dim_type     = tensor.get_bis().get_type(idim);
    const lt::split_points sp = tensor.get_bis().get_splits(dim_type);
    block_idx[idim]           = 0;
    for (size_t isp = 0; isp < sp.get_num_points(); ++isp) {
      if (sp[isp] > idx[idim]) break;
      block_idx[idim] = isp + 1;
    }
  }

  lt::index<N> inblock_idx;
  lt::index<N> bstart    = tensor.get_bis().get_block_start(block_idx);
  lt::dimensions<N> bdim = tensor.get_bis().get_block_dims(block_idx);
  for (size_t idim = 0; idim < N; ++idim) {
    inblock_idx[idim] = idx[idim] - bstart[idim];
    if (inblock_idx[idim] >= bdim[idim]) {
      throw runtime_error(
            "Internal error: Determined in-block index overshoots block dimensionality");
    }
  }

  return std::make_pair(block_idx, inblock_idx);
}

template <typename Comparator, size_t N>
std::vector<std::pair<std::vector<size_t>, scalar_type>> execute_select_n(
      lt::btensor<N, scalar_type>& tensor, size_t n, bool unique_by_symmetry) {
  using btod_select_t = lt::btod_select<N, Comparator>;
  std::list<lt::block_tensor_element<N, scalar_type>> il;
  if (!unique_by_symmetry) {
    lt::symmetry<N, scalar_type> nosym(tensor.get_bis());
    btod_select_t(tensor, nosym).perform(il, n);
  } else {
    btod_select_t(tensor).perform(il, n);
  }

  std::vector<std::pair<std::vector<size_t>, scalar_type>> ret;
  for (auto it = il.begin(); it != il.end(); ++it) {
    std::vector<size_t> fidx(N);
    const lt::index<N> bstart = tensor.get_bis().get_block_start(it->get_block_index());
    for (size_t i = 0; i < N; ++i) {
      fidx[i] = bstart[i] + it->get_in_block_index()[i];
    }
    ret.emplace_back(fidx, it->get_value());
  }
  return ret;
}
}  // namespace

template <size_t N>
void TensorImpl<N>::check_state() const {
  if (m_expr_ptr == nullptr && m_libtensor_ptr == nullptr) {
    throw runtime_error(
          "Internal error: m_libtensor_ptr and m_expr_ptr cannot both be nullptr.");
  }
  if (m_expr_ptr != nullptr && m_libtensor_ptr != nullptr) {
    throw runtime_error(
          "Internal error: m_libtensor_ptr and m_expr_ptr cannot both be set pointers.");
  }

  if (N != ndim()) {
    throw runtime_error("Internal error: libtensor dimension (== " + std::to_string(N) +
                        ") and tensor dimension (==" + std::to_string(ndim()) +
                        ") differ.");
  }

  if (m_libtensor_ptr) {
    std::vector<size_t> btshape(N);
    const lt::dimensions<N>& dims = m_libtensor_ptr->get_bis().get_dims();
    for (size_t i = 0; i < N; ++i) {
      btshape[i] = dims.get_dim(i);
    }
    if (shape() != btshape) {
      throw runtime_error(
            "Internal error: libtensor shape (== " + shape_to_string(btshape) +
            ") and tensor shape (==" + shape_to_string(shape()) + ") differ.");
    }

    const std::vector<std::vector<size_t>> tensorblocks =
          get_block_starts(*m_libtensor_ptr);
    for (size_t i = 0; i < N; ++i) {
      if (axes()[i].block_starts != tensorblocks[i]) {
        throw runtime_error("Internal error: Block starts of btensor " +
                            shape_to_string(tensorblocks[i]) + " at dimension " +
                            std::to_string(i) +
                            " do not agree with the cached block sarts " +
                            shape_to_string(axes()[i].block_starts) + ".");
      }
    }
  }

  if (m_expr_ptr) {
    if (m_expr_ptr->permutation.size() != N) {
      throw runtime_error(
            "Internal error: Expression dimension (== " + std::to_string(N) +
            ") and tensor dimension (==" + std::to_string(ndim()) + ") differ.");
    }
  }
}

template <size_t N>
void TensorImpl<N>::reset_state(
      std::shared_ptr<lt::btensor<N, scalar_type>> libtensor_ptr) const {
  if (m_expr_ptr != nullptr && m_libtensor_ptr != nullptr) {
    throw runtime_error(
          "Internal error: m_libtensor_ptr and m_expr_ptr cannot both be set pointers.");
  }
  if (libtensor_ptr == nullptr) {
    throw runtime_error(
          "Internal error: libtensor_ptr to be used for reset_state is a nullptr.");
  }
  m_libtensor_ptr = libtensor_ptr;
  m_expr_ptr.reset();
  check_state();
}

template <size_t N>
void TensorImpl<N>::reset_state(std::shared_ptr<ExpressionTree> expr_ptr) const {
  if (m_expr_ptr != nullptr && m_libtensor_ptr != nullptr) {
    throw runtime_error(
          "Internal error: m_libtensor_ptr and m_expr_ptr cannot both be set pointers.");
  }
  if (expr_ptr == nullptr) {
    throw runtime_error(
          "Internal error: expr_ptr to be used for reset_state is a nullptr.");
  }
  m_expr_ptr = expr_ptr;
  m_libtensor_ptr.reset();
  check_state();
}

template <size_t N>
TensorImpl<N>::TensorImpl(std::shared_ptr<const AdcMemory> adcmem_ptr,
                          std::vector<AxisInfo> axes,
                          std::shared_ptr<lt::btensor<N, scalar_type>> libtensor_ptr,
                          std::shared_ptr<ExpressionTree> expr_ptr)
      : Tensor(adcmem_ptr, axes), m_libtensor_ptr(nullptr), m_expr_ptr(nullptr) {
  if (axes.size() != N) {
    throw invalid_argument("axes length (== " + std::to_string(axes.size()) +
                           ") does not agree with tensor dimensionality " +
                           std::to_string(N));
  }

  if (expr_ptr != nullptr && libtensor_ptr != nullptr) {
    throw invalid_argument("libtensor_ptr and expr_ptr cannot both be set pointers.");
  }
  if (expr_ptr == nullptr && libtensor_ptr == nullptr) {
    // Allocate an empty tensor.
    libtensor_ptr = std::make_shared<lt::btensor<N, scalar_type>>(as_bispace<N>(axes));
  }

  if (expr_ptr != nullptr) reset_state(expr_ptr);
  if (libtensor_ptr != nullptr) reset_state(libtensor_ptr);
}

template <size_t N>
void TensorImpl<N>::evaluate() const {
  check_state();
  if (!needs_evaluation()) return;

  // Allocate output tensor and evaluate
  auto newtensor_ptr =
        std::make_shared<lt::btensor<N, scalar_type>>(as_bispace<N>(m_axes));
  m_expr_ptr->evaluate_to(*newtensor_ptr, /* add = */ false);

  // Check and test new tensor, cleanup expression
  reset_state(newtensor_ptr);
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::empty_like() const {
  check_state();

  // TODO This evaluates the expression, which is probably an unexpected effect.

  // Create new btensor using the old bispace
  auto newtensor_ptr =
        std::make_shared<lt::btensor<N, scalar_type>>(libtensor_ptr()->get_bis());

  // Copy the symmetry over
  lt::block_tensor_ctrl<N, scalar_type> ctrl_to(*newtensor_ptr);
  lt::block_tensor_ctrl<N, scalar_type> ctrl_from(*libtensor_ptr());
  lt::so_copy<N, scalar_type>(ctrl_from.req_const_symmetry())
        .perform(ctrl_to.req_symmetry());

  // Enwrap inside TensorImpl and return
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, std::move(newtensor_ptr));
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::nosym_like() const {
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes);
}

template <size_t N>
void TensorImpl<N>::set_mask(std::string mask, scalar_type value) {
  if (N != mask.size()) {
    throw invalid_argument("The number of characters in the index mask (== " + mask +
                           ") does not agree with the Tensor dimensionality (== " +
                           std::to_string(N) + ")");
  }

  // Non-obviously the indices for the mask have to start with 1,
  // 0 gives utterly weird values
  size_t next_idx = 1;
  std::map<char, size_t> char_to_idx;
  lt::sequence<N, size_t> seq(0);
  for (size_t i = 0; i < mask.size(); ++i) {
    const char c  = mask[i];
    const auto it = char_to_idx.find(c);
    if (it == char_to_idx.end()) {
      char_to_idx[c] = next_idx;
      seq[i]         = next_idx;
      next_idx += 1;
    } else {
      seq[i] = it->second;
    }
  }

  if (char_to_idx.size() == N) {
    // Every character in the mask is different ... just use bto_set
    // TODO Optimise: This evaluates, but here there is no point ... Just allocate
    lt::btod_set<N>(value).perform(*libtensor_ptr());
  } else {
    lt::btod_set_diag<N>(seq, value).perform(*libtensor_ptr());
  }
}

template <size_t N>
void TensorImpl<N>::set_random() {
  // TODO optimise: No point in evaluating ... just allocate
  lt::btod_random<N>().perform(*libtensor_ptr());
}

namespace {

template <size_t R, size_t D, size_t N>
std::shared_ptr<Tensor> execute_diagonal(
      std::shared_ptr<const AdcMemory> adcmem_ptr,
      const std::vector<std::shared_ptr<const lt::letter>>& label_result,
      const std::vector<std::shared_ptr<const lt::letter>>& label_diag,
      const std::vector<std::shared_ptr<const lt::letter>>& label_expr,
      std::shared_ptr<ExpressionTree> expr, std::vector<AxisInfo> axes) {
  auto lthis    = expr->attach_letters<N>(label_expr);
  auto res      = lt::expr::diag(*label_diag[0], strip_safe<D>(label_diag), lthis);
  auto expr_ptr = std::make_shared<ExpressionTree>(
        res.get_expr(), extract_expr_permutation(res, label_result), expr->keepalives);
  return std::make_shared<TensorImpl<R>>(adcmem_ptr, axes, expr_ptr);
}

}  // namespace

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::diagonal(std::vector<size_t> axes) {
  if (axes.size() <= 1) {
    throw invalid_argument("Axes needs to have at least two entries.");
  }
  auto label                                = make_label(N);
  std::shared_ptr<ExpressionTree> expr_this = expression_ptr();

  std::vector<std::shared_ptr<const lt::letter>> diag;
  std::unique_ptr<AxisInfo> diagaxis_ptr;
  std::vector<size_t> used_indices;
  for (size_t& i : axes) {
    auto it = std::find(used_indices.begin(), used_indices.end(), i);
    if (it != used_indices.end()) {
      throw invalid_argument("Axes may not have repeated indices.");
    }
    if (i >= N) {
      throw invalid_argument("Axis index (== " + std::to_string(i) +
                             ") goes beyond dimensionality of tensor (" +
                             std::to_string(N) + ")");
    }
    if (diagaxis_ptr == nullptr) {
      diagaxis_ptr.reset(new AxisInfo(m_axes[i]));
    } else {
      if (*diagaxis_ptr != m_axes[i]) {
        throw invalid_argument("Cannot form diagonal over differing axes. " +
                               diagaxis_ptr->label + " versus " + m_axes[i].label + ".");
      }
    }

    diag.push_back(label[i]);
    used_indices.push_back(i);
  }

  // Collect letters, which are to be left unchanged.
  std::vector<AxisInfo> axes_result;
  std::vector<std::shared_ptr<const lt::letter>> label_result;
  for (size_t i = 0; i < N; ++i) {
    auto it = std::find(used_indices.begin(), used_indices.end(), i);
    if (it == used_indices.end()) {
      label_result.push_back(label[i]);
      axes_result.push_back(m_axes[i]);
    }
  }
  label_result.push_back(diag[0]);
  axes_result.push_back(*diagaxis_ptr);

#define IF_MATCHES_EXECUTE(NTHIS, DIAG)                                            \
  if (N == NTHIS && DIAG == diag.size()) {                                         \
    constexpr size_t DIMOUT = NTHIS - DIAG + 1;                                    \
    static_assert((DIMOUT > 0 && DIMOUT < 100),                                    \
                  "Internal error with DIMOUT computation");                       \
    return execute_diagonal<DIMOUT, DIAG, NTHIS>(m_adcmem_ptr, label_result, diag, \
                                                 label, expr_this, axes_result);   \
  }

  IF_MATCHES_EXECUTE(2, 2)  //
  IF_MATCHES_EXECUTE(3, 2)  //
  IF_MATCHES_EXECUTE(3, 3)  //
  IF_MATCHES_EXECUTE(4, 2)  //
  IF_MATCHES_EXECUTE(4, 3)  //
  IF_MATCHES_EXECUTE(4, 3)  //

  throw not_implemented_error("diagonal not implemented for dimensionality " +
                              std::to_string(N) + " and " + std::to_string(diag.size()) +
                              " axes indices.");

#undef IF_MATCHES_EXECUTE
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::scale(scalar_type c) const {
  // Collect labelled expressions
  std::vector<std::shared_ptr<const lt::letter>> label = make_label(N);
  std::shared_ptr<ExpressionTree> expr_this            = expression_ptr();
  auto lthis = expr_this->attach_letters<N>(label);

  // Execute the operation
  auto scaled = c * lthis;

  auto expr_ptr = std::make_shared<ExpressionTree>(
        scaled.get_expr(), extract_expr_permutation(scaled, label),
        expr_this->keepalives);
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, expr_ptr);
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::add(std::shared_ptr<Tensor> other) const {
  DIMENSIONALITY_CHECK(other);

  // Collect labelled expressions
  auto label                                 = make_label(N);
  std::shared_ptr<ExpressionTree> expr_this  = expression_ptr();
  std::shared_ptr<ExpressionTree> expr_other = as_expression(other);
  auto lthis                                 = expr_this->attach_letters<N>(label);
  auto lother                                = expr_other->attach_letters<N>(label);

  // Execute the operation
  auto sum = lthis + lother;

  auto expr_ptr = std::make_shared<ExpressionTree>(
        sum.get_expr(), extract_expr_permutation(sum, label),
        merge(expr_this->keepalives, expr_other->keepalives));
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, expr_ptr);
}

template <size_t N>
void TensorImpl<N>::add_linear_combination(
      std::vector<scalar_type> scalars,
      std::vector<std::shared_ptr<Tensor>> tensors) const {
  if (scalars.size() != tensors.size()) {
    throw dimension_mismatch(
          "std::vector of scalars has size " + std::to_string(scalars.size()) +
          ", but passed vector of tensors has size " + std::to_string(tensors.size()));
  }
  if (scalars.size() == 0) return;

  std::unique_ptr<lt::btod_add<N>> operator_ptr;
  for (size_t i = 0; i < scalars.size(); ++i) {
    DIMENSIONALITY_CHECK(tensors[i]);
    if (!operator_ptr) {
      operator_ptr.reset(new lt::btod_add<N>(as_btensor<N>(tensors[i]), scalars[i]));
    } else {
      operator_ptr->add_op(as_btensor<N>(tensors[i]), scalars[i]);
    }
  }
  operator_ptr->perform(*libtensor_ptr(), 1.0);
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::multiply(std::shared_ptr<Tensor> other) const {
  DIMENSIONALITY_CHECK(other);

  // Collect labelled expressions
  auto label                                 = make_label(N);
  std::shared_ptr<ExpressionTree> expr_this  = expression_ptr();
  std::shared_ptr<ExpressionTree> expr_other = as_expression(other);
  auto lthis                                 = expr_this->attach_letters<N>(label);
  auto lother                                = expr_other->attach_letters<N>(label);

  // Execute the operation
  auto mult = lt::expr::mult(lthis, lother);

  auto expr_ptr = std::make_shared<ExpressionTree>(
        mult.get_expr(), extract_expr_permutation(mult, label),
        merge(expr_this->keepalives, expr_other->keepalives));
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, expr_ptr);
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::divide(std::shared_ptr<Tensor> other) const {
  DIMENSIONALITY_CHECK(other);

  // Collect labelled expressions
  auto label                                 = make_label(N);
  std::shared_ptr<ExpressionTree> expr_this  = expression_ptr();
  std::shared_ptr<ExpressionTree> expr_other = as_expression(other);
  auto lthis                                 = expr_this->attach_letters<N>(label);
  auto lother                                = expr_other->attach_letters<N>(label);

  // Execute the operation
  auto div = lt::expr::div(lthis, lother);

  auto expr_ptr = std::make_shared<ExpressionTree>(
        div.get_expr(), extract_expr_permutation(div, label),
        merge(expr_this->keepalives, expr_other->keepalives));
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, expr_ptr);
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::copy() const {
  if (needs_evaluation()) {
    // Return deep copy to the expression
    return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, m_expr_ptr);
  } else {
    // Actually make a deep copy of the tensor
    auto ret_ptr = empty_like();
    auto lt_ptr  = std::static_pointer_cast<TensorImpl<N>>(ret_ptr)->libtensor_ptr();
    lt::btod_copy<N>(*libtensor_ptr()).perform(*lt_ptr);
    return ret_ptr;
  }
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::transpose(std::vector<size_t> permutation) const {
  if (permutation.size() != N) {
    throw invalid_argument(
          "Number of indices in provided transposition axes (== " +
          std::to_string(permutation.size()) +
          ") does not agree with tensor dimension (== " + std::to_string(N) + ").");
  }

  // Reorder the axes
  std::vector<AxisInfo> newaxes;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (permutation[i] == permutation[j]) {
        throw invalid_argument("Duplicate index in transposition axes (" +
                               std::to_string(permutation[i]) + ") at indices " +
                               std::to_string(i) + " and " + std::to_string(j) + ".");
      }
    }
    if (permutation[i] >= N) {
      throw invalid_argument("Invalid axes specifier " + std::to_string(permutation[i]) +
                             ". Exceeds tensor dimension -1 (==" + std::to_string(N - 1) +
                             ").");
    }

    newaxes.push_back(m_axes[permutation[i]]);
  }

  // Chain permutations
  std::shared_ptr<ExpressionTree> expr_this = expression_ptr();
  std::vector<size_t> result_permutation;
  for (size_t i = 0; i < N; ++i) {
    result_permutation.push_back(expr_this->permutation[permutation[i]]);
  }
  auto expr_ptr = std::make_shared<ExpressionTree>(
        *expr_this->tree_ptr, result_permutation, expr_this->keepalives);
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, newaxes, expr_ptr);
}

namespace {
template <size_t R, size_t N, size_t M>
std::shared_ptr<Tensor> execute_direct_sum(
      std::shared_ptr<const AdcMemory> adcmem_ptr,
      const std::vector<std::shared_ptr<const lt::letter>>& label_result,
      const std::vector<std::shared_ptr<const lt::letter>>& label_first,
      const std::vector<std::shared_ptr<const lt::letter>>& label_second,
      std::shared_ptr<ExpressionTree> expr_first,
      std::shared_ptr<ExpressionTree> expr_second, std::vector<AxisInfo> axes_result) {
  auto lfirst  = expr_first->attach_letters<N>(label_first);
  auto lsecond = expr_second->attach_letters<M>(label_second);

  // Execute the operation (dirsum)
  auto res      = lt::dirsum(lfirst, lsecond);
  auto expr_ptr = std::make_shared<ExpressionTree>(
        res.get_expr(), extract_expr_permutation(res, label_result),
        merge(expr_first->keepalives, expr_second->keepalives));
  return std::make_shared<TensorImpl<R>>(adcmem_ptr, axes_result, expr_ptr);
}
}  // namespace

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::direct_sum(std::shared_ptr<Tensor> other) const {
  typedef std::vector<std::shared_ptr<const lt::letter>> lalvec_t;
  lalvec_t label_first  = make_label(N);
  lalvec_t label_second = make_label(other->ndim());

  lalvec_t label_result;
  for (auto& v : label_first) label_result.push_back(v);
  for (auto& v : label_second) label_result.push_back(v);
  std::vector<AxisInfo> axes_result;
  for (auto& ax : m_axes) axes_result.push_back(ax);
  for (auto& ax : other->axes()) axes_result.push_back(ax);

  std::shared_ptr<ExpressionTree> expr_first  = expression_ptr();
  std::shared_ptr<ExpressionTree> expr_second = as_expression(other);
#define IF_MATCHES_EXECUTE(DIMA, DIMB)                                                   \
  if (DIMA == label_first.size() && DIMB == label_second.size()) {                       \
    constexpr size_t DIMOUT = DIMA + DIMB;                                               \
    static_assert((DIMOUT > 0 && DIMOUT < 100),                                          \
                  "Internal error with DIMOUT computation");                             \
    if (DIMOUT != label_result.size()) {                                                 \
      throw runtime_error(                                                               \
            "Internal error: Inconsistency with DIMOUT and label_contracted.size()");    \
    }                                                                                    \
    return execute_direct_sum<DIMOUT, DIMA, DIMB>(m_adcmem_ptr, label_result,            \
                                                  label_first, label_second, expr_first, \
                                                  expr_second, axes_result);             \
  }

  IF_MATCHES_EXECUTE(1, 1)  //
  IF_MATCHES_EXECUTE(1, 2)  //
  IF_MATCHES_EXECUTE(1, 3)  //
  IF_MATCHES_EXECUTE(2, 1)  //
  IF_MATCHES_EXECUTE(2, 2)  //
  IF_MATCHES_EXECUTE(3, 1)  //

  throw not_implemented_error(
        "Did not implement the case of a direct_sum of two tensors of dimension " +
        std::to_string(ndim()) + " and " + std::to_string(other->ndim()) + ".");
#undef IF_MATCHES_EXECUTE
}

template <>
double TensorImpl<1>::trace(std::string) const {
  throw runtime_error("Trace can only be applied to tensors of even rank.");
}

template <>
double TensorImpl<3>::trace(std::string) const {
  throw runtime_error("Trace can only be applied to tensors of even rank.");
}

template <size_t N>
double TensorImpl<N>::trace(std::string contraction) const {
  if (contraction.size() != N) {
    throw invalid_argument(
          "Number of passed contraction indices needs to match tensor dimensionality.");
  }

  std::vector<std::pair<size_t, size_t>> trace_pairs;
  std::vector<bool> index_done(N, false);
  for (size_t i = 0; i < N; ++i) {
    if (index_done[i]) continue;
    index_done[i]   = true;
    bool found_pair = false;
    for (size_t j = i + 1; j < N; ++j) {
      if (contraction[i] == contraction[j]) {
        if (m_axes[i] != m_axes[j]) {
          throw invalid_argument("Axes to be traced along do not agree: " +
                                 m_axes[i].label + " versus " + m_axes[j].label);
        }
        index_done[j] = true;
        trace_pairs.push_back({i, j});
        found_pair = true;
        break;
      }
    }
    if (!found_pair) {
      throw("Found no matching second index for '" + std::string(1, contraction[i]) +
            "'.");
    }
  }

  if (2 * trace_pairs.size() != N) {
    throw invalid_argument(
          "Expected to find half as many trace indices as there are tensor dimensions, "
          "i.e. " +
          std::to_string(N / 2) + " indices and not " +
          std::to_string(trace_pairs.size()) + ".");
  }

  typedef std::vector<std::shared_ptr<const lt::letter>> lalvec_t;
  lalvec_t label = make_label(N);
  lalvec_t tlal_first;
  lalvec_t tlal_second;
  for (const auto& p : trace_pairs) {
    tlal_first.push_back(label[p.first]);
    tlal_second.push_back(label[p.second]);
  }

  std::shared_ptr<ExpressionTree> expr_this = expression_ptr();
  auto lfirst                               = expr_this->attach_letters<N>(label);
  constexpr size_t K                        = N / 2;
  return lt::trace(strip_safe<K>(tlal_first), strip_safe<K>(tlal_second), lfirst);
}

namespace {
template <size_t R, size_t K, size_t N, size_t M>
TensorOrScalar execute_tensordot_contract(
      std::shared_ptr<const AdcMemory> adcmem_ptr,
      const std::vector<std::shared_ptr<const lt::letter>>& label_result,
      const std::vector<std::shared_ptr<const lt::letter>>& label_contracted,
      const std::vector<std::shared_ptr<const lt::letter>>& label_first,
      const std::vector<std::shared_ptr<const lt::letter>>& label_second,
      std::shared_ptr<ExpressionTree> expr_first,
      std::shared_ptr<ExpressionTree> expr_second, std::vector<AxisInfo> axes_result) {
  // Build labelled expressions:
  auto lfirst  = expr_first->attach_letters<N>(label_first);
  auto lsecond = expr_second->attach_letters<M>(label_second);

  // Execute the operation (contract)
  auto res      = lt::contract(strip_safe<K>(label_contracted), lfirst, lsecond);
  auto expr_ptr = std::make_shared<ExpressionTree>(
        res.get_expr(), extract_expr_permutation(res, label_result),
        merge(expr_first->keepalives, expr_second->keepalives));
  auto tensor_ptr = std::make_shared<TensorImpl<R>>(adcmem_ptr, axes_result, expr_ptr);
  return TensorOrScalar{tensor_ptr, 0.0};
}

template <size_t R, size_t N, size_t M>
TensorOrScalar execute_tensordot_tensorprod(
      std::shared_ptr<const AdcMemory> adcmem_ptr,
      const std::vector<std::shared_ptr<const lt::letter>>& label_result,
      const std::vector<std::shared_ptr<const lt::letter>>& label_first,
      const std::vector<std::shared_ptr<const lt::letter>>& label_second,
      std::shared_ptr<ExpressionTree> expr_first,
      std::shared_ptr<ExpressionTree> expr_second, std::vector<AxisInfo> axes_result) {
  auto lfirst  = expr_first->attach_letters<N>(label_first);
  auto lsecond = expr_second->attach_letters<M>(label_second);

  // Execute the operation (tensor product)
  auto res      = lfirst * lsecond;
  auto expr_ptr = std::make_shared<ExpressionTree>(
        res.get_expr(), extract_expr_permutation(res, label_result),
        merge(expr_first->keepalives, expr_second->keepalives));
  auto tensor_ptr = std::make_shared<TensorImpl<R>>(adcmem_ptr, axes_result, expr_ptr);
  return TensorOrScalar{tensor_ptr, 0.0};
}

}  // namespace

template <size_t N>
TensorOrScalar TensorImpl<N>::tensordot(
      std::shared_ptr<Tensor> other,
      std::pair<std::vector<size_t>, std::vector<size_t>> axes) const {
  const std::vector<size_t>& axes_first  = axes.first;
  const std::vector<size_t>& axes_second = axes.second;

  if (axes_first.size() != axes_second.size()) {
    throw invalid_argument(
          "Length of the passed axes does not agree "
          " (first == " +
          shape_to_string(axes_first) + " and second == " + shape_to_string(axes_second) +
          ")");
  }
  if (axes_first.size() > N) {
    throw invalid_argument(
          "Length of the passed axes overshoots dimensionality of the first "
          "tensor.");
  }
  if (axes_first.size() > other->ndim()) {
    throw invalid_argument(
          "Length of the passed axes overshoots dimensionality of the second "
          "tensor.");
  }

  // Build label for first and second tensor and contraction indices
  typedef std::vector<std::shared_ptr<const lt::letter>> lalvec_t;
  lalvec_t label_first  = make_label(N);
  lalvec_t label_second = make_label(other->ndim());
  lalvec_t label_contracted;
  for (size_t i = 0; i < axes_first.size(); ++i) {
    std::shared_ptr<const lt::letter> l_contraction = label_first[axes_first[i]];
    label_second[axes_second[i]]                    = l_contraction;

    if (m_axes[axes_first[i]] != other->axes()[axes_second[i]]) {
      throw invalid_argument(
            "tensordot can only contract equivalent axes together. The " +
            std::to_string(i) + "-th axis clashes (" + m_axes[axes_first[i]].label +
            " versus " + other->axes()[axes_second[i]].label + "). Tensor spaces are " +
            space() + " and " + other->space());
    }
    label_contracted.push_back(l_contraction);
  }

  // Build labels of the result
  lalvec_t label_result;
  std::vector<AxisInfo> axes_result;
  for (size_t i = 0; i < N; ++i) {
    auto it = std::find(axes_first.begin(), axes_first.end(), i);
    if (it == axes_first.end()) {
      label_result.push_back(label_first[i]);
      axes_result.push_back(m_axes[i]);
    }
  }
  for (size_t j = 0; j < other->ndim(); ++j) {
    auto it = std::find(axes_second.begin(), axes_second.end(), j);
    if (it == axes_second.end()) {
      label_result.push_back(label_second[j]);
      axes_result.push_back(other->axes()[j]);
    }
  }

  if (label_result.size() != N + other->ndim() - 2 * label_contracted.size()) {
    throw runtime_error(
          "Internal error: Result index count does not agree with expected "
          "number.");
  }

  // Build labelled expressions:
  std::shared_ptr<ExpressionTree> expr_first  = expression_ptr();
  std::shared_ptr<ExpressionTree> expr_second = as_expression(other);

  // Branch into the different cases
  if (label_result.size() == 0 && N == label_contracted.size() &&
      N == label_first.size() && N == label_second.size()) {
    // Full contraction => execute dot_product

    auto lfirst  = expr_first->attach_letters<N>(label_first);
    auto lsecond = expr_second->attach_letters<N>(label_second);
    return TensorOrScalar{nullptr, lt::dot_product(lfirst, lsecond)};
  } else if (label_contracted.size() == 0) {
#define IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD(DIMA, DIMB)                            \
  if (DIMA == label_first.size() && DIMB == label_second.size()) {                    \
    constexpr size_t DIMOUT = DIMB + DIMA;                                            \
    static_assert((DIMOUT > 0 && DIMOUT < 100),                                       \
                  "Internal error with DIMOUT computation");                          \
    if (DIMOUT != label_result.size()) {                                              \
      throw runtime_error(                                                            \
            "Internal error: Inconsistency with DIMOUT and label_contracted.size()"); \
    }                                                                                 \
    return execute_tensordot_tensorprod<DIMOUT, DIMA, DIMB>(                          \
          m_adcmem_ptr, label_result, label_first, label_second, expr_first,          \
          expr_second, axes_result);                                                  \
  }

    //
    // Instantiation generated from TensorImpl/instantiate_valid.py
    //
    IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD(1, 1)  //
    IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD(1, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD(1, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD(2, 1)  //
    IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD(2, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD(3, 1)  //

#undef IF_DIMENSIONS_MATCH_EXECUTE_TENSORPROD
  } else {
    // Other cases => normal contraction
#define IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(N_CONTR_IDCS, DIMA, DIMB)                \
  if (DIMA == label_first.size() && DIMB == label_second.size() &&                    \
      N_CONTR_IDCS == label_contracted.size()) {                                      \
    constexpr size_t DIMOUT = DIMB + DIMA - N_CONTR_IDCS - N_CONTR_IDCS;              \
    static_assert((DIMOUT > 0 && DIMOUT < 100),                                       \
                  "Internal error with DIMOUT computation");                          \
    if (DIMOUT != label_result.size()) {                                              \
      throw runtime_error(                                                            \
            "Internal error: Inconsistency with DIMOUT and label_contracted.size()"); \
    }                                                                                 \
    return execute_tensordot_contract<DIMOUT, N_CONTR_IDCS, DIMA, DIMB>(              \
          m_adcmem_ptr, label_result, label_contracted, label_first, label_second,    \
          expr_first, expr_second, axes_result);                                      \
  }

    //
    // Instantiation generated from TensorImpl/instantiate_valid.py
    //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 1, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 1, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 1, 4)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 2, 1)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 2, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 2, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 2, 4)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 3, 1)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 3, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 3, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 4, 1)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(1, 4, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 2, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 2, 4)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 3, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 3, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 3, 4)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 4, 2)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 4, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(2, 4, 4)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(3, 3, 4)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(3, 4, 3)  //
    IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT(3, 4, 4)  //

#undef IF_DIMENSIONS_MATCH_EXECUTE_CONTRACT
  }

  throw not_implemented_error(
        "Did not implement the case of a tensordot over " +
        std::to_string(label_contracted.size()) +
        " indices for two tensors of dimensions " + std::to_string(label_first.size()) +
        " and " + std::to_string(label_second.size()) +
        ", yielding a tensor of dimension " + std::to_string(label_result.size()) + ".");
}

template <size_t N>
std::vector<scalar_type> TensorImpl<N>::dot(
      std::vector<std::shared_ptr<Tensor>> tensors) const {
  std::vector<scalar_type> ret(tensors.size(), 0.0);

  for (size_t i = 0; i < tensors.size(); ++i) {
    auto tensor_ptr = std::static_pointer_cast<TensorImpl<N>>(tensors[i]);
    if (ndim() != tensor_ptr->ndim()) {
      throw dimension_mismatch(
            "Dimensionality of this tensor (" + std::to_string(ndim()) +
            ") does not agree with the dimensionality of the " + std::to_string(i) +
            "-th tensor passed, which has dimensionality " +
            std::to_string(tensor_ptr->ndim()) + ".");
    }
    if (shape() != tensor_ptr->shape()) {
      throw dimension_mismatch("Shape of this tensor (" + shape_to_string(shape()) +
                               ") does not agree with the shape of the " +
                               std::to_string(i) + "-th tensor passed, which has shape " +
                               shape_to_string(tensor_ptr->shape()) + ".");
    }

    ret[i] = lt::btod_dotprod<N>(*libtensor_ptr(), *(tensor_ptr->libtensor_ptr()))
                   .calculate();
  }
  return ret;
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::symmetrise(
      const std::vector<std::vector<size_t>>& permutations) const {
  if (permutations.size() == 0) {
    return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, m_libtensor_ptr,
                                           m_expr_ptr);  // Noop
  }

  // Label this expression
  auto label                                = make_label(N);
  std::shared_ptr<ExpressionTree> expr_this = expression_ptr();
  auto lthis                                = expr_this->attach_letters<N>(label);

  // Execute the operation
  auto symmetrised = [&permutations, &lthis, this, label]() {
    if (permutations.size() == 1) {
      auto parsed = parse_permutation<1>(m_axes, strip_safe<N>(label), permutations);
      return 0.5 * lt::expr::symm(parsed.first, parsed.second, lthis);
    } else if (permutations.size() == 2) {
      auto parsed = parse_permutation<2>(m_axes, strip_safe<N>(label), permutations);
      return 0.5 * lt::expr::symm(parsed.first, parsed.second, lthis);
    } else {
      throw runtime_error(
            "Antisymmetrisation not implemented for more than two index pairs.");
    }
  }();

  auto expr_ptr = std::make_shared<ExpressionTree>(
        symmetrised.get_expr(), extract_expr_permutation(symmetrised, label),
        expr_this->keepalives);
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, expr_ptr);
}

template <size_t N>
std::shared_ptr<Tensor> TensorImpl<N>::antisymmetrise(
      const std::vector<std::vector<size_t>>& permutations) const {
  if (permutations.size() == 0) {
    return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, m_libtensor_ptr,
                                           m_expr_ptr);  // Noop
  }

  // Label this expression
  auto label                                = make_label(N);
  std::shared_ptr<ExpressionTree> expr_this = expression_ptr();
  auto lthis                                = expr_this->attach_letters<N>(label);

  // Execute the operation
  auto antisymmetrised = [&permutations, &lthis, this, &label]() {
    if (permutations.size() == 1) {

      auto parsed = parse_permutation<1>(m_axes, strip_safe<N>(label), permutations);
      return 0.5 * lt::expr::asymm(parsed.first, parsed.second, lthis);
    } else if (permutations.size() == 2) {
      auto parsed = parse_permutation<2>(m_axes, strip_safe<N>(label), permutations);
      return 0.5 * lt::expr::asymm(parsed.first, parsed.second, lthis);
    } else {
      throw runtime_error(
            "Antisymmetrisation not implemented for more than two index pairs.");
    }
  }();

  auto expr_ptr = std::make_shared<ExpressionTree>(
        antisymmetrised.get_expr(), extract_expr_permutation(antisymmetrised, label),
        expr_this->keepalives);
  return std::make_shared<TensorImpl<N>>(m_adcmem_ptr, m_axes, expr_ptr);
}

template <size_t N>
void TensorImpl<N>::set_element(const std::vector<size_t>& tidx, scalar_type value) {
  if (!is_element_allowed(tidx)) {
    throw runtime_error("Setting tensor index (" + shape_to_string(tidx) +
                        ") not allowed, since zero by symmetry.");
  }

  lt::index<N> block_idx;
  lt::index<N> in_block_idx;
  std::tie(block_idx, in_block_idx) = assert_convert_tensor_index(*libtensor_ptr(), tidx);
  lt::btod_set_elem<N>{}.perform(*libtensor_ptr(), block_idx, in_block_idx, value);
}

template <size_t N>
scalar_type TensorImpl<N>::get_element(const std::vector<size_t>& tidx) const {
  lt::index<N> block_idx;
  lt::index<N> in_block_idx;
  std::tie(block_idx, in_block_idx) = assert_convert_tensor_index(*libtensor_ptr(), tidx);

  // The value we want to return
  scalar_type ret;

  // Make an orbit over the block index, i.e. find the canonical block
  lt::block_tensor_ctrl<N, scalar_type> ctrl(*libtensor_ptr());
  lt::dimensions<N> bidims(libtensor_ptr()->get_bis().get_block_index_dims());
  lt::orbit<N, scalar_type> obit(ctrl.req_const_symmetry(), block_idx);

  // If the orbit (i.e. the index) is not allowed by symmetry, than the element is
  // zero.
  if (!obit.is_allowed()) return 0;

  // Absolute index of the canonical block
  const lt::abs_index<N> idx_abscanon(obit.get_acindex(), bidims);

  // Check if we have hit a zero block, then return 0 as well.
  if (ctrl.req_is_zero_block(idx_abscanon.get_index())) return 0;

  // The transformation between the selected block and the canonical block
  const lt::tensor_transf<N, scalar_type>& tr = obit.get_transf(block_idx);

  // Transform in-block index.
  lt::index<N> pibidx(in_block_idx);  //< permuted in-block index
  pibidx.permute(tr.get_perm());

  // Get the actual value from the dense tensor representing the block
  {
    auto& block = ctrl.req_const_block(idx_abscanon.get_index());
    lt::dense_tensor_rd_ctrl<N, scalar_type> blkctrl(block);

    const scalar_type* p = blkctrl.req_const_dataptr();
    ret                  = p[lt::abs_index<N>(pibidx, block.get_dims()).get_abs_index()];
    blkctrl.ret_const_dataptr(p);
  }

  // Transform value
  tr.get_scalar_tr().apply(ret);

  ctrl.ret_const_block(idx_abscanon.get_index());
  return ret;
}

template <size_t N>
bool TensorImpl<N>::is_element_allowed(const std::vector<size_t>& tidx) const {
  lt::index<N> block_idx;
  std::tie(block_idx, std::ignore) = assert_convert_tensor_index(*libtensor_ptr(), tidx);

  // Check if the block is allowed in the symmetry of the guess
  lt::block_tensor_ctrl<N, scalar_type> ctrl(*libtensor_ptr());
  lt::orbit<N, scalar_type> orb(ctrl.req_const_symmetry(), block_idx);
  const bool block_allowed = orb.is_allowed();

  // TODO Even if the block is allowed, the index might not be
  // (e.g. diagonal of an anti-symmetric tensor).
  const bool index_allowed = true;

  return block_allowed && index_allowed;
}

template <size_t N>
std::vector<std::pair<std::vector<size_t>, scalar_type>> TensorImpl<N>::select_n_absmax(
      size_t n, bool unique_by_symmetry) const {
  return execute_select_n<lt::compare4absmax>(*libtensor_ptr(), n, unique_by_symmetry);
}

template <size_t N>
std::vector<std::pair<std::vector<size_t>, scalar_type>> TensorImpl<N>::select_n_absmin(
      size_t n, bool unique_by_symmetry) const {
  return execute_select_n<lt::compare4absmin>(*libtensor_ptr(), n, unique_by_symmetry);
}

template <size_t N>
std::vector<std::pair<std::vector<size_t>, scalar_type>> TensorImpl<N>::select_n_max(
      size_t n, bool unique_by_symmetry) const {
  return execute_select_n<lt::compare4max>(*libtensor_ptr(), n, unique_by_symmetry);
}

template <size_t N>
std::vector<std::pair<std::vector<size_t>, scalar_type>> TensorImpl<N>::select_n_min(
      size_t n, bool unique_by_symmetry) const {
  return execute_select_n<lt::compare4min>(*libtensor_ptr(), n, unique_by_symmetry);
}

template <size_t N>
void TensorImpl<N>::export_to(scalar_type* memptr, size_t size) const {
  if (this->size() != size) {
    throw invalid_argument("The memory provided (== " + std::to_string(size) +
                           ") does not agree with the number of tensor elements (== " +
                           std::to_string(this->size()) + ")");
  }
  lt::btod_export<N>(*libtensor_ptr()).perform(memptr);
}

template <size_t N>
void TensorImpl<N>::import_from(const scalar_type* memptr, size_t size,
                                scalar_type tolerance, bool symmetry_check) {
  if (this->size() != size) {
    throw invalid_argument("The memory size provided (== " + std::to_string(size) +
                           ") does not agree with the number of tensor elements (== " +
                           std::to_string(this->size()) + ")");
  }

  if (symmetry_check) {
    // Slow algorithm with symmetry check (via libtensor)

    // Zero this memory.
    // TODO Check this really needs to be done for proper functioning
    //      There is some indication that yes, unfortunately.
    lt::btod_set<N>(0.0).perform(*libtensor_ptr());

    scalar_type* noconst_mem = const_cast<scalar_type*>(memptr);
    libtensor::btod_import_raw<N>(noconst_mem, libtensor_ptr()->get_bis().get_dims(),
                                  tolerance)
          .perform(*libtensor_ptr());
  } else {
    // Fast algorithm without symmetry check (via import_from with generator)
    auto fast_importer = [this, memptr](
                               const std::vector<std::pair<size_t, size_t>>& range,
                               scalar_type* ptr) {
      if (range.size() != N) {
        throw runtime_error("Internal error: Dimension mismatch in fast_importer");
      }

      // Strides for accessing the full memptr
      std::array<size_t, N> strides_full;
      strides_full[N - 1] = 1;
      for (size_t idim = N - 1; idim > 0; --idim) {
        strides_full[idim - 1] = strides_full[idim] * shape()[idim];
      }

      // Dimensionalities for access into the small ptr
      std::array<size_t, N> dim_ptr;
      for (size_t idim = 0; idim != N; ++idim) {
        dim_ptr[idim] = range[idim].second - range[idim].first;
      }

      // Strides for accessing the small ptr
      std::array<size_t, N> strides_ptr;
      strides_ptr[N - 1] = 1;
      for (size_t idim = N - 1; idim > 0; --idim) {
        strides_ptr[idim - 1] = strides_ptr[idim] * dim_ptr[idim];
      }
      const size_t size_ptr = strides_ptr[0] * dim_ptr[0];

      for (size_t iabs = 0; iabs < size_ptr; ++iabs) {
        size_t full_abs = 0;
        for (size_t idim = 0; idim != N; ++idim) {
          size_t idx = range[idim].first + (iabs / strides_ptr[idim]) % dim_ptr[idim];
          full_abs += strides_full[idim] * idx;
        }
        ptr[iabs] = memptr[full_abs];
      }
    };
    import_from(fast_importer, tolerance, false);
  }
}

template <size_t N>
void TensorImpl<N>::import_from(
      std::function<void(const std::vector<std::pair<size_t, size_t>>&, scalar_type*)>
            generator,
      scalar_type tolerance, bool symmetry_check) {

  if (symmetry_check) {
    // Slow algorithm with symmetry check (via import_from with libtensor)

    // Get complete data via generator ...
    std::vector<std::pair<size_t, size_t>> full_range(N);
    for (size_t i = 0; i < N; ++i) {
      full_range[i].first  = 0;
      full_range[i].second = shape()[i];
    }
    std::vector<scalar_type> data(this->size());
    generator(full_range, data.data());

    // ... and import it
    import_from(data, tolerance, true);
  } else {
    // Fast algorithm by only considering canonical blocks
    // First zero out the tensor completely.
    lt::block_tensor_ctrl<N, scalar_type> ctrl(*libtensor_ptr());
    ctrl.req_zero_all_blocks();

    // Now loop over all orbits, which thus gives access to all
    // canonical blocks
    lt::orbit_list<N, scalar_type> orbitlist(ctrl.req_const_symmetry());
    lt::index<N> blk_idx;  // Holder for canonical-block indices
    for (auto it = orbitlist.begin(); it != orbitlist.end(); ++it) {
      orbitlist.get_index(it, blk_idx);

      // Get the dense tensor for the current canonical block
      // and import the data from the generator
      lt::dense_tensor_wr_i<N, scalar_type>& blk = ctrl.req_block(blk_idx);

      // Compute range to import
      std::vector<std::pair<size_t, size_t>> range(N);
      const lt::block_index_space<N>& bis = libtensor_ptr()->get_bis();
      lt::index<N> blk_start(bis.get_block_start(blk_idx));
      lt::dimensions<N> blk_dims(bis.get_block_dims(blk_idx));
      for (size_t i = 0; i < N; ++i) {
        range[i].first  = blk_start[i];
        range[i].second = blk_start[i] + blk_dims[i];
      }

      bool all_zero = false;
      {
        lt::dense_tensor_wr_ctrl<N, scalar_type> cblk(blk);
        cblk.req_prefetch();
        scalar_type* ptr = cblk.req_dataptr();

        generator(range, ptr);  // Import data from the generator

        if (tolerance > 0) {
          // Check that some are non-zero
          all_zero = std::all_of(
                ptr, ptr + blk_dims.get_size(),
                [tolerance](scalar_type e) { return std::abs(e) < tolerance; });
        }

        cblk.ret_dataptr(ptr);
      }
      ctrl.ret_block(blk_idx);

      if (all_zero) {
        ctrl.req_zero_block(blk_idx);
      }
    }
  }
}

template <size_t N>
std::string TensorImpl<N>::describe_symmetry() const {
  std::stringstream ss;

  lt::block_tensor_ctrl<N, scalar_type> ctrl(*libtensor_ptr());
  ss << ctrl.req_const_symmetry();

  return ss.str();
}

template <size_t N>
std::string TensorImpl<N>::describe_expression(std::string stage) const {
  if (needs_evaluation()) {
    std::stringstream ss;
    if (stage == "unoptimised") {
      ss << *m_expr_ptr;
    } else if (stage == "optimised") {
      lt::expr::print_tree(m_expr_ptr->optimised_tree(), ss, 2);
    } else if (stage == "evaluation") {
      auto newtensor_ptr =
            std::make_shared<lt::btensor<N, scalar_type>>(as_bispace<N>(m_axes));
      lt::expr::expr_tree evaltree =
            m_expr_ptr->evaluation_tree(*newtensor_ptr, /* add = */ false);
      lt::expr::print_tree(evaltree, ss, 2);
    } else {
      throw invalid_argument("Stage " + stage +
                             " not valid for describe_expression. Try 'unoptimised', "
                             "'optimised', 'evaluation' or 'evaluation'");
    }
    return ss.str();
  } else {
    return "btensor of shape " + shape_to_string(m_shape);
  }
}

template <size_t N>
std::shared_ptr<ExpressionTree> TensorImpl<N>::expression_ptr() const {
  if (m_expr_ptr != nullptr) {
    if (m_libtensor_ptr != nullptr) {
      throw runtime_error(
            "Internal error: m_libtensor_ptr is not a nullptr and neither is "
            "m_expr_ptr.");
    }
    return m_expr_ptr;  // Already have an expression for myself, just return it
  } else {
    if (m_libtensor_ptr == nullptr) {
      throw runtime_error(
            "Internal error: Both m_libtensor_ptr and m_expr_ptr are nullptrs.");
    }

    // Make a new expression tree with just this tensor in it
    return std::make_shared<ExpressionTree>(
          lt::expr::node_ident_any_tensor<N, scalar_type>(*m_libtensor_ptr),
          identity_permutation(N), std::vector<std::shared_ptr<void>>{m_libtensor_ptr});
  }
}

std::shared_ptr<ExpressionTree> as_expression(const std::shared_ptr<Tensor>& tensor) {
  std::shared_ptr<ExpressionTree> ret;
  if (tensor->ndim() == 1) {
    ret = std::static_pointer_cast<TensorImpl<1>>(tensor)->expression_ptr();
  } else if (tensor->ndim() == 2) {
    ret = std::static_pointer_cast<TensorImpl<2>>(tensor)->expression_ptr();
  } else if (tensor->ndim() == 3) {
    ret = std::static_pointer_cast<TensorImpl<3>>(tensor)->expression_ptr();
  } else if (tensor->ndim() == 4) {
    ret = std::static_pointer_cast<TensorImpl<4>>(tensor)->expression_ptr();
  } else {
    throw not_implemented_error("Only implemented for dimensionality <= 4.");
  }

  if (ret->permutation.size() != tensor->ndim()) {
    throw runtime_error("Internal error: Mismatch between permutation.size() == " +
                        std::to_string(ret->permutation.size()) +
                        " and tensor dimensionality " + std::to_string(tensor->ndim()) +
                        ".");
  }

  return ret;
}

template <size_t N>
lt::btensor<N, scalar_type>& as_btensor(const std::shared_ptr<Tensor>& in) {
  if (N != in->ndim()) {
    throw dimension_mismatch("Requested dimensionality " + std::to_string(N) +
                             ", but passed tensor has dimensionality " +
                             std::to_string(in->ndim()));
  }
  return static_cast<lt::btensor<N, scalar_type>&>(
        *std::static_pointer_cast<TensorImpl<N>>(in));
}

template <size_t N>
std::shared_ptr<lt::btensor<N, scalar_type>> as_btensor_ptr(
      const std::shared_ptr<Tensor>& in) {
  if (N != in->ndim()) {
    throw dimension_mismatch("Requested dimensionality " + std::to_string(N) +
                             ", but passed tensor has dimensionality " +
                             std::to_string(in->ndim()));
  }
  return std::static_pointer_cast<TensorImpl<N>>(in)->libtensor_ptr();
}

namespace {
template <size_t N>
std::shared_ptr<Tensor> make_tensor_inner(std::shared_ptr<Symmetry> symmetry) {
  auto ltsym_ptr = as_lt_symmetry<N>(*symmetry);
  auto newtensor_ptr =
        std::make_shared<lt::btensor<N, scalar_type>>(ltsym_ptr->get_bis());

  lt::block_tensor_ctrl<N, scalar_type> ctrl_to(*newtensor_ptr);
  lt::so_copy<N, scalar_type>(*ltsym_ptr).perform(ctrl_to.req_symmetry());
  return std::make_shared<TensorImpl<N>>(symmetry->adcmem_ptr(), symmetry->axes(),
                                         std::move(newtensor_ptr));
}
}  // namespace

std::shared_ptr<Tensor> make_tensor(std::shared_ptr<Symmetry> symmetry) {
  if (symmetry->ndim() == 1) {
    return make_tensor_inner<1>(symmetry);
  } else if (symmetry->ndim() == 2) {
    return make_tensor_inner<2>(symmetry);
  } else if (symmetry->ndim() == 3) {
    return make_tensor_inner<3>(symmetry);
  } else if (symmetry->ndim() == 4) {
    return make_tensor_inner<4>(symmetry);
  } else {
    throw not_implemented_error("Only implemented for dimensionality <= 4.");
  }
}

std::shared_ptr<Tensor> make_tensor(std::shared_ptr<const AdcMemory> adcmem_ptr,
                                    std::vector<AxisInfo> axes) {
  if (axes.size() == 1) {
    return std::make_shared<TensorImpl<1>>(adcmem_ptr, axes);
  } else if (axes.size() == 2) {
    return std::make_shared<TensorImpl<2>>(adcmem_ptr, axes);
  } else if (axes.size() == 3) {
    return std::make_shared<TensorImpl<3>>(adcmem_ptr, axes);
  } else if (axes.size() == 4) {
    return std::make_shared<TensorImpl<4>>(adcmem_ptr, axes);
  } else {
    throw not_implemented_error("Only implemented for dimensionality <= 4.");
  }
}

//
// Explicit instantiation
//

#define INSTANTIATE(DIM)                                                                 \
  template class TensorImpl<DIM>;                                                        \
                                                                                         \
  template lt::btensor<DIM, scalar_type>& as_btensor(const std::shared_ptr<Tensor>& in); \
                                                                                         \
  template std::shared_ptr<lt::btensor<DIM, scalar_type>> as_btensor_ptr(                \
        const std::shared_ptr<Tensor>& in);

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)

#undef INSTANTIATE

}  // namespace libadcc
