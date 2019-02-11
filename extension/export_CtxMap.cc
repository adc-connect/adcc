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

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::shared_ptr<Tensor> CtxMap__at_tensor(std::shared_ptr<ctx::CtxMap> self,
                                                 std::string key) {
  return self->at_ptr<Tensor>(key);
}

static std::shared_ptr<Tensor> CtxMap__at_tensor_def(std::shared_ptr<ctx::CtxMap> self,
                                                     std::string key,
                                                     std::shared_ptr<Tensor> def) {
  return self->at_ptr<Tensor>(key, def);
}

static scalar_type CtxMap__at_scalar(const ctx::CtxMap& self, std::string key) {
  return self.at<scalar_type>(key);
}

static scalar_type CtxMap__at_scalar_def(const ctx::CtxMap& self, std::string key,
                                         scalar_type def) {
  return self.at<scalar_type>(key, def);
}

static bool CtxMap__at_bool(const ctx::CtxMap& self, std::string key) {
  return self.at<bool>(key);
}

static bool CtxMap__at_bool_def(const ctx::CtxMap& self, std::string key, bool def) {
  return self.at<bool>(key, def);
}

static unsigned long CtxMap__at_unsigned_long(const ctx::CtxMap& self, std::string key) {
  return self.at<unsigned long>(key);
}

static unsigned long CtxMap__at_unsigned_long_def(const ctx::CtxMap& self,
                                                  std::string key, unsigned long def) {
  return self.at<unsigned long>(key, def);
}

static unsigned int CtxMap__at_unsigned_int_def(const ctx::CtxMap& self, std::string key,
                                                unsigned int def) {
  return self.at<unsigned int>(key, def);
}

static unsigned int CtxMap__at_unsigned_int(const ctx::CtxMap& self, std::string key) {
  return self.at<unsigned int>(key);
}

static py::list CtxMap__keys(const ctx::CtxMap& self) {
  py::list keys;
  for (auto& kv : self) {
    keys.append(kv.key());
  }
  return keys;
}

void export_CtxMap(py::module& m) {
  using namespace ctx;

  py::class_<ctx::CtxMap, std::shared_ptr<ctx::CtxMap>>(
        m, "CtxMap", "Class holding a CtxMap context map object.")
        .def("keys", &CtxMap__keys)
        .def("exists", &CtxMap::exists)
        .def("at_tensor", &CtxMap__at_tensor)
        .def("at_tensor", &CtxMap__at_tensor_def)
        .def("at_scalar", &CtxMap__at_scalar)
        .def("at_scalar", &CtxMap__at_scalar_def)
        .def("at_bool", &CtxMap__at_bool)
        .def("at_bool", &CtxMap__at_bool_def)
        .def("at_unsigned_long", &CtxMap__at_unsigned_long)
        .def("at_unsigned_long", &CtxMap__at_unsigned_long_def)
        .def("at_unsigned_int", &CtxMap__at_unsigned_int)
        .def("at_unsigned_int", &CtxMap__at_unsigned_int_def)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
