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
#include "../config.hh"
#include <ostream>

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/expr/btensor/btensor.h>
#include <libtensor/expr/dag/print_tree.h>
#pragma GCC visibility pop

namespace libadcc {

// Expression representing a libtensor btensor object upon evaluation.
class ExpressionTree {
 public:
  std::shared_ptr<libtensor::expr::expr_tree> tree_ptr;  //< Libtensor expression tree

  /// Permutation to be applied to the result of evaluating the expression above
  std::vector<size_t> permutation;

  /// List of all objects to keep alive as long as expression exists
  std::vector<std::shared_ptr<void>> keepalives;

  ExpressionTree(const libtensor::expr::node& node, std::vector<size_t> permutation_,
                 std::vector<std::shared_ptr<void>> keepalives_)
        : tree_ptr(new libtensor::expr::expr_tree(node)),
          permutation(permutation_),
          keepalives(keepalives_) {}
  ExpressionTree(libtensor::expr::expr_tree expr_tree, std::vector<size_t> permutation_,
                 std::vector<std::shared_ptr<void>> keepalives_)
        : tree_ptr(new libtensor::expr::expr_tree(std::move(expr_tree))),
          permutation(permutation_),
          keepalives(keepalives_) {}

  template <size_t N>
  libtensor::expr::expr_rhs<N, scalar_type> attach_letters(
        const std::vector<std::shared_ptr<const libtensor::letter>>& letters) const;

  /** Return an optimised form of the internal expression tree */
  libtensor::expr::expr_tree optimised_tree();

  /** Return the expression tree libtensor will use for running the actual evaluation.
   *  This includes optimisations and formed intermediates. */
  template <size_t N>
  libtensor::expr::expr_tree evaluation_tree(libtensor::btensor<N, scalar_type>& result,
                                             bool add = false) const;

  /** Evaluate tree into a pre-allocated btensor. Optimise should be called first.
   *  Optionally (iff add is true) add the result of the evaluation to the tensor */
  template <size_t N>
  void evaluate_to(libtensor::btensor<N, scalar_type>& result, bool add = false) const;
};

inline std::ostream& operator<<(std::ostream& o, const ExpressionTree& expr) {
  libtensor::expr::print_tree(*expr.tree_ptr, o, 2);
  return o;
}

}  // namespace libadcc
