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

#include "BlasSequential.hh"

#ifdef HAVE_OPENBLAS
extern "C" int openblas_get_num_threads();
extern "C" void openblas_set_num_threads(int num_threads);
#elif HAVE_MKL
#include <mkl.h>
#define openblas_get_num_threads mkl_get_max_threads
#define openblas_set_num_threads mkl_set_num_threads
#else  // no openblas, no MKL
namespace {
int openblas_get_num_threads() { return 1; }
void openblas_set_num_threads(int) {}
}  // namespace
#endif

namespace adcc {

BlasSequential::BlasSequential() : blas_num_threads(openblas_get_num_threads()) {
  openblas_set_num_threads(1);
}

BlasSequential::~BlasSequential() { openblas_set_num_threads(blas_num_threads); }

}  // namespace adcc

