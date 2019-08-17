//
// Copyright (C) 2018 by the adcc authors
//
// This file is part of adcc.
//
// adcc is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// adcc is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with adcc. If not, see <http://www.gnu.org/licenses/>.
//

#include <adcc/Tensor.hh>
#include <adcc/config.hh>
#include <ctx/CtxMap.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

#define IF_AT_TYPE_RETURN(TYPE)                                                      \
  if (self->at_raw_value(key).type_name_raw() == typeid(TYPE).name()) {              \
    return py::cast(self->at<TYPE>(key));                                            \
  }                                                                                  \
  if (self->at_raw_value(key).type_name_raw() == typeid(std::vector<TYPE>).name()) { \
    return py::cast(self->at<std::vector<TYPE>>(key));                               \
  }

#define IF_ATPTR_TYPE_RETURN(TYPE)                                      \
  if (self->at_raw_value(key).type_name_raw() == typeid(TYPE).name()) { \
    return py::cast(self->at_ptr<TYPE>(key));                           \
  }

static py::object CtxMap__at(std::shared_ptr<ctx::CtxMap> self, std::string key) {
  IF_AT_TYPE_RETURN(bool)
  IF_AT_TYPE_RETURN(int)
  IF_AT_TYPE_RETURN(long)
  IF_AT_TYPE_RETURN(unsigned long)
  IF_AT_TYPE_RETURN(unsigned int)
  IF_AT_TYPE_RETURN(std::string)
  IF_AT_TYPE_RETURN(scalar_type)
  IF_ATPTR_TYPE_RETURN(Tensor)

  throw not_implemented_error("Generic at() not implemented for type " +
                              self->at_raw_value(key).type_name() + ".");
}

static py::object CtxMap__at_def(std::shared_ptr<ctx::CtxMap> self, std::string key,
                                 py::object def) {
  if (self->exists(key)) {
    return CtxMap__at(self, key);
  } else {
    return def;
  }
}

#undef IF_AT_TYPE_RETURN
#undef IF_ATPTR_TYPE_RETURN

static py::list CtxMap__keys(const ctx::CtxMap& self) {
  py::list keys;
  for (auto& kv : self) {
    keys.append(kv.key());
  }
  return keys;
}

static void CtxMap__erase(ctx::CtxMap& ctx, const std::string& path,
                          bool recursive = false) {
  if (recursive) {
    ctx.erase_recursive(path);
  } else {
    ctx.erase(path);
  }
}

void export_CtxMap(py::module& m) {
  using namespace ctx;

  py::class_<ctx::CtxMap, std::shared_ptr<ctx::CtxMap>>(
        m, "CtxMap", "Class holding a CtxMap context map object.")
        .def(py::init<>())
        .def("keys", &CtxMap__keys)
        .def("exists", &CtxMap::exists)
        //
        .def("get", &CtxMap__at)
        .def("get", &CtxMap__at_def)
        .def("__getitem__", &CtxMap__at)
        .def("__delitem__",
             [](ctx::CtxMap& ctx, const std::string& key) { ctx.erase(key); })
        .def("erase", &CtxMap__erase)
        .def("describe",
             [](ctx::CtxMap& self) {
               std::stringstream ss;
               ss << self << std::endl;
               return ss.str();
             })
        //
        .def("update_tensor", [](ctx::CtxMap& self, std::string key,
                                 std::shared_ptr<Tensor> val) { self.update(key, val); })
        .def("update_scalar", [](ctx::CtxMap& self, std::string key,
                                 scalar_type val) { self.update(key, val); })
        .def("update_bool",
             [](ctx::CtxMap& self, std::string key, bool val) { self.update(key, val); })
        .def("update_unsigned_long", [](ctx::CtxMap& self, std::string key,
                                        unsigned long val) { self.update(key, val); })
        .def("update_unsigned_int", [](ctx::CtxMap& self, std::string key,
                                       unsigned int val) { self.update(key, val); })
        .def("update_string", [](ctx::CtxMap& self, std::string key,
                                 std::string val) { self.update(key, val); })
        .def("update_scalar_list",
             [](ctx::CtxMap& self, std::string key, std::vector<scalar_type> val) {
               self.update_copy(key, val);
             })
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
