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

#include "ThreadPool.hh"
#include "exceptions.hh"
#include <libutil/thread_pool/thread_pool.h>

namespace libadcc {
namespace {
void kill_thread_pool(std::shared_ptr<void>& holder_ptr) {
  if (holder_ptr == nullptr) return;

  auto* tp_ptr = static_cast<libutil::thread_pool*>(holder_ptr.get());
  tp_ptr->terminate();
  tp_ptr->dissociate();
  holder_ptr.reset();
}
}  // namespace

void ThreadPool::reinit(size_t n_running, size_t n_total) {
  if (n_running == 0 || n_total == 0) {
    throw invalid_argument("n_running and n_total need to be larger than zero.");
  }
  if (n_total < n_running) {
    throw invalid_argument("n_running cannot be larger than n_total.");
  }
  kill_thread_pool(m_holder_ptr);

  // Setup new thread pool
  m_holder_ptr.reset(new libutil::thread_pool(n_total, n_running));
  auto* tp_ptr = static_cast<libutil::thread_pool*>(m_holder_ptr.get());
  tp_ptr->associate();

  m_n_running = n_running;
  m_n_total   = n_total;
}

ThreadPool::~ThreadPool() { kill_thread_pool(m_holder_ptr); }
}  // namespace libadcc
