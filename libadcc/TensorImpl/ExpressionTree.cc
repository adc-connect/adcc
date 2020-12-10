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

// This needs to be the first include ... libtensor reasons
#pragma GCC visibility push(default)
#include <libtensor/expr/eval/eval_exception.h>
#pragma GCC visibility pop
//
#include "../exceptions.hh"
#include "ExpressionTree.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/expr/btensor/eval_btensor_double.h>
#include <libtensor/expr/btensor/impl/eval_tree_builder_btensor.h>
#include <libtensor/expr/btensor/impl/tensor_from_node.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_assign.h>
#include <libtensor/expr/dag/node_ident.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/opt/opt_add_before_transf.h>
#include <libtensor/expr/opt/opt_merge_adjacent_add.h>
#include <libtensor/expr/opt/opt_merge_adjacent_transf.h>
#include <libtensor/expr/opt/opt_merge_equiv_ident.h>
#include <libtensor/linalg/BlasSequential.h>
#include <libtensor/symmetry/so_copy.h>
#pragma GCC visibility pop

namespace libadcc {
namespace lt = libtensor;

template <size_t N>
lt::expr::expr_rhs<N, scalar_type> ExpressionTree::attach_letters(
      const std::vector<std::shared_ptr<const lt::letter>>& letters) const {

  if (N != permutation.size()) {
    throw runtime_error("Internal error: Mismatch between permutation.size() == " +
                        std::to_string(permutation.size()) +
                        " and expr_rhs dimensionality " + std::to_string(N) + ".");
  }
  if (letters.size() != permutation.size()) {
    throw runtime_error("Internal error: Mismatch between permutation.size() == " +
                        std::to_string(permutation.size()) + " and letters size " +
                        std::to_string(letters.size()) + ".");
  }

  // The permutation is to be applied *after* the expression is evaluated,
  // so these letters need to pick up the inverse permutation.
  std::vector<const lt::letter*> let_ptrs;
  for (size_t i = 0; i < N; ++i) {
    auto it = std::find(permutation.begin(), permutation.end(), i);
    if (it == permutation.end()) {
      throw runtime_error("Internal error: Could not build inverse permutation");
    }
    const size_t inv_i = static_cast<size_t>(it - permutation.begin());

    let_ptrs.push_back(letters[inv_i].get());
  }

  // This copies both the expression as well as the letters to internal storage.
  return lt::expr::expr_rhs<N, scalar_type>(*tree_ptr, let_ptrs);
}

namespace {
lt::expr::expr_tree optimise_tree(const lt::expr::expr_tree& in) {
  lt::expr::expr_tree ret(in);
  lt::expr::opt_merge_equiv_ident(ret);
  lt::expr::opt_merge_adjacent_transf(ret);
  lt::expr::opt_add_before_transf(ret);
  lt::expr::opt_merge_adjacent_transf(ret);
  lt::expr::opt_merge_adjacent_add(ret);
  return ret;
}

bool is_linear_combination(const lt::expr::expr_tree& assigntree) {
  using namespace libtensor::expr;

  auto is_valid_summand = [&assigntree](size_t id) {
    if (assigntree.get_vertex(id).check_type<node_ident>()) return true;

    if (assigntree.get_vertex(id).check_type<node_transform_base>()) {
      const node_transform<scalar_type>& nt =
            assigntree.get_vertex(id).template recast_as<node_transform<scalar_type>>();
      for (size_t i = 0; i < nt.get_perm().size(); ++i) {
        if (i != nt.get_perm()[i]) return false;  // Need identity permutation
      }

      // Only one subnode for transformation.
      if (assigntree.get_edges_out(id).size() != 1) return false;
      const size_t tid = assigntree.get_edges_out(id)[0];

      if (assigntree.get_vertex(tid).check_type<node_ident>()) {
        return true;
      }
    }
    return false;
  };

  bool found_add = false;
  for (auto it = assigntree.begin(); it != assigntree.end(); ++it) {
    if (assigntree.get_vertex(it).check_type<node_add>()) {
      if (found_add) return false;  // There should be exactly one add
      auto add_edges = assigntree.get_edges_out(it);
      for (auto aed = add_edges.begin(); aed != add_edges.end(); ++aed) {
        if (!is_valid_summand(*aed)) return false;
      }
      found_add = true;
    } else if (assigntree.get_vertex(it).check_type<node_assign>()) {
      if (assigntree.get_root() != assigntree.get_id(it)) return false;
    } else if (!is_valid_summand(assigntree.get_id(it))) {
      return false;
    }
  }

  return found_add;
}

template <size_t N>
void evaluate_linear_combination(const lt::expr::expr_tree& assigntree,
                                 lt::btensor<N, scalar_type>& result, bool add) {
  using namespace libtensor::expr;
  if (!is_linear_combination(assigntree)) {
    throw invalid_argument(
          "evaluate_linear_combination got a tree which is not a linear combination "
          "tree");
  }

  lt::btensor<N, scalar_type>* first_btensor = nullptr;
  std::unique_ptr<lt::btod_add<N>> operator_ptr;
  auto add_summand = [&assigntree, &operator_ptr, &first_btensor](size_t id) {
    eval_btensor_double::btensor_from_node<N, scalar_type> extract(assigntree, id);
    auto& btensor = dynamic_cast<lt::btensor<N, scalar_type>&>(extract.get_btensor());
    // Notice: No permutation allowed here
    const lt::permutation<N>& perm = extract.get_transf().get_perm();
    if (!perm.is_identity()) {
      throw runtime_error("Internal error: Caught non-identity permutation.");
    }

    const double c = extract.get_transf().get_scalar_tr().get_coeff();
    if (!operator_ptr) {
      // Operator not yet set up -> do so with first summand
      operator_ptr.reset(new lt::btod_add<N>(btensor, c));
      first_btensor = &btensor;
    } else {
      // Operator is set up -> just add summand
      operator_ptr->add_op(btensor, c);
    }
  };

  auto root_edges = assigntree.get_edges_out(assigntree.get_root());
  for (auto it = root_edges.begin(); it != root_edges.end(); ++it) {
    if (assigntree.get_vertex(*it).check_type<node_add>()) {
      auto add_edges = assigntree.get_edges_out(*it);
      for (auto aed = add_edges.begin(); aed != add_edges.end(); ++aed) {
        add_summand(*aed);
      }
      break;  // Just one add tree
    }
  }

  // Copy symmmetry over
  if (first_btensor == nullptr) {
    throw runtime_error("Internal error: Got nullptr where set pointer was expected.");
  }
  lt::block_tensor_ctrl<N, scalar_type> ctrl_to(result);
  lt::block_tensor_ctrl<N, scalar_type> ctrl_from(*first_btensor);
  lt::so_copy<N, scalar_type>(ctrl_from.req_const_symmetry())
        .perform(ctrl_to.req_symmetry());

  if (!add) lt::btod_set<N>(0.0).perform(result);
  operator_ptr->perform(result, 1.0);
}

/** Return the expression tree resulting from assigning tree to
 *  the passed result. Optionally the assignment adds elements instead of setting them.
 */
template <size_t N>
libtensor::expr::expr_tree assignment_tree(const ExpressionTree& tree,
                                           libtensor::btensor<N, scalar_type>& result,
                                           bool add) {
  using namespace libtensor::expr;
  if (add) throw not_implemented_error("add = true not tested so far");

  // Form the RHS of the assignment
  std::vector<std::shared_ptr<const lt::letter>> label_safe;
  for (size_t i = 0; i < N; ++i) {
    label_safe.push_back(std::make_shared<lt::letter>());
  }
  expr_rhs<N, scalar_type> rhs = tree.attach_letters<N>(label_safe);

  // Form the labels for libtensor
  std::vector<const lt::letter*> label_unsafe;
  for (size_t i = 0; i < N; ++i) {
    label_unsafe.push_back(label_safe[i].get());
  }
  label<N> label(label_unsafe);

  // Build the assignment tree, starting from the assignment node
  node_assign nassign(N, add);
  expr_tree assigntree(nassign);
  expr_tree::node_id_t id = assigntree.get_root();
  node_ident_any_tensor<N, scalar_type> nresult(result);  // LHS of assignment
  assigntree.add(id, nresult);

  // Check if permutations are needed
  lt::permutation<N> px = label.permutation_of(rhs.get_label());
  if (!px.is_identity()) {
    std::vector<size_t> perm(N);
    for (size_t i = 0; i < N; i++) perm[i] = px[i];

    node_transform<scalar_type> n3(perm, lt::scalar_transf<scalar_type>());
    id = assigntree.add(id, n3);
  }

  // Add permuted or unpermuted RHS
  assigntree.add(id, rhs.get_expr());

  return assigntree;
}
}  // namespace

/** Return an optimised form of the internal expression tree */
lt::expr::expr_tree ExpressionTree::optimised_tree() { return optimise_tree(*tree_ptr); }

template <size_t N>
libtensor::expr::expr_tree ExpressionTree::evaluation_tree(
      libtensor::btensor<N, scalar_type>& result, bool add) const {
  // This does exactly the same stuff as
  // lt::expr::eval_btensor<double>().evaluate(assigntree) except doing the evaluation.
  lt::expr::expr_tree assigntree = assignment_tree(*this, result, add);
  lt::expr::eval_tree_builder_btensor bld(assigntree);
  bld.build();
  return bld.get_tree();
}

template <size_t N>
void ExpressionTree::evaluate_to(lt::btensor<N, scalar_type>& result, bool add) const {
  lt::expr::expr_tree assigntree = optimise_tree(assignment_tree(*this, result, add));

  lt::BlasSequential seq;  // Switch to sequential BLAS for evaluations

  if (is_linear_combination(assigntree)) {
    evaluate_linear_combination(assigntree, result, add);
    return;
  }

  // Normal evaluation via libtensor
  lt::expr::eval_btensor<double>().evaluate(assigntree);
}

//
// Explicit instantiation
//

#define INSTANTIATE(DIM)                                               \
  template void ExpressionTree::evaluate_to(                           \
        libtensor::btensor<DIM, scalar_type>& result, bool add) const; \
                                                                       \
  template libtensor::expr::expr_tree ExpressionTree::evaluation_tree( \
        libtensor::btensor<DIM, scalar_type>& result, bool add) const;

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)

#undef INSTANTIATE

}  // namespace libadcc
