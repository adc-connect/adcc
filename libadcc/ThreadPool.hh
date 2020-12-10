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
#include <memory>

namespace libadcc {
/**
 *  \addtogroup Utilities
 */
///@{

/** Pool managing how many threads may be used for calculations. */
class ThreadPool {
 public:
  /** Initialise the thread pool.
   *
   * \param  n_running  The total number of running threads to employ
   * \param  n_total    The total number of worker threads to use
   *                    (Either running or ready)
   */
  ThreadPool(size_t n_running, size_t n_total) : m_holder_ptr{nullptr} {
    reinit(n_running, n_total);
  }

  /** Reinitialise the thread pool.
   *
   * \param  n_running  The total number of running threads to employ
   * \param  n_total    The total number of worker threads to use
   *                    (Either running or idle)
   */
  void reinit(size_t n_running, size_t n_total);

  /** Initialise a thread pool without parallelisation */
  ThreadPool() : ThreadPool(1, 1) {}

  /** Return the number of running threads. */
  size_t n_running() const { return m_n_running; }

  /** Return the total number of worker threads (running or ready) */
  size_t n_total() const { return m_n_total; }

  // Avoid copying or copy-assinging
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  ~ThreadPool();

 private:
  // Hack to avoid the libutil data structures in the interface
  std::shared_ptr<void> m_holder_ptr;
  size_t m_n_running;
  size_t m_n_total;
};

///@}
}  // namespace libadcc
