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

#include "setup_point_group_table.hh"
#include "../exceptions.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/symmetry/point_group_table.h>
#pragma GCC visibility pop

namespace libadcc {
namespace lt        = libtensor;
using irrep_label_t = lt::point_group_table::label_t;

namespace {
std::map<size_t, std::string> table_to_map(const lt::point_group_table& pgt) {
  std::map<size_t, std::string> ret;
  for (irrep_label_t l = 0; l < pgt.get_n_labels(); l++) {
    ret[l] = pgt.get_irrep_name(l);
  }
  return ret;
}
}  // namespace

std::map<size_t, std::string> setup_point_group_table(lt::product_table_container& ptc,
                                                      const std::string& point_group) {
  // Details:
  //    - adcman/adcman/qchem/import_ao_data.{h,C}
  //    - liblegacy/liblegacy/qcimport_pgsymmetry.{h,C}

  if (point_group == "C1") {
    if (!ptc.table_exists(point_group)) {
      const std::vector<std::string> irreps{"A"};
      lt::point_group_table pgt("C1", irreps, irreps[0]);
      ptc.add(pgt);
      return table_to_map(pgt);
    }
  } else if (point_group == "C2v") {
    if (!ptc.table_exists(point_group)) {
      const std::vector<std::string> irreps{"A1", "A2", "B1", "B2"};
      lt::point_group_table pgt("C2v", irreps, irreps[0]);
      pgt.add_product(1, 1, 0);
      pgt.add_product(1, 2, 3);
      pgt.add_product(1, 3, 2);
      pgt.add_product(2, 2, 0);
      pgt.add_product(2, 3, 1);
      pgt.add_product(3, 3, 0);
      pgt.check();
      ptc.add(pgt);
      return table_to_map(pgt);
    }
  } else if (point_group == "D2") {
    if (!ptc.table_exists(point_group)) {
      const std::vector<std::string> irreps{"A", "B1", "B2", "B3"};
      lt::point_group_table pgt("D2", irreps, irreps[0]);
      pgt.add_product(1, 1, 0);
      pgt.add_product(1, 2, 3);
      pgt.add_product(1, 3, 2);
      pgt.add_product(2, 2, 0);
      pgt.add_product(2, 3, 1);
      pgt.add_product(3, 3, 0);
      pgt.check();
      ptc.add(pgt);
      return table_to_map(pgt);
    }
  } else {
    throw not_implemented_error("Point group " + point_group + " not implemented.");
  }
  return table_to_map(ptc.req_const_table<lt::point_group_table>(point_group));
}

}  // namespace libadcc
