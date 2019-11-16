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

#include <sstream>

#include <adcc/ThreadPool.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

void export_threading(py::module& m) {
  auto threadpool_ptr = std::make_shared<adcc::ThreadPool>();
  auto set_threads    = [threadpool_ptr](size_t n_threads) {
    threadpool_ptr->reinit(n_threads, 2 * n_threads - 1);
  };
  try {
    set_threads(py::module::import("os").attr("cpu_count")().cast<size_t>());
  } catch (py::cast_error& c) {
    // Single-threaded setup
  }

  m.def(
        "get_n_threads", [threadpool_ptr]() { return threadpool_ptr->n_running(); },
        "Get the number of running worker threads used by adcc.");
  m.def("set_n_threads", set_threads,
        "Set the number of running worker threads used by adcc");
  m.def(
        "set_n_threads_total",
        [threadpool_ptr](size_t n_total) {
          const size_t n_running = threadpool_ptr->n_running();
          threadpool_ptr->reinit(n_running, n_total);
        },
        "Set the total number of threads (running and sleeping) used by adcc. This will "
        "disappear in the future. Do not rely on it.");
  m.def(
        "get_n_threads_total", [threadpool_ptr]() { return threadpool_ptr->n_total(); },
        "Get the total number of threads (running and sleeping) used by adcc. This will "
        "disappear in the future. Do not rely on it.");
}

}  // namespace py_iface
}  // namespace adcc
