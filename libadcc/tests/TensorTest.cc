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

#include "../Tensor.hh"
#include "../AdcMemory.hh"
#include "../TensorImpl.hh"
#include "../TensorImpl/get_block_starts.hh"
#include "TensorTestData.hh"
#include "output_tensor.hh"
#include "random_tensor.hh"
#include "wrap_libtensor.hh"
#include <algorithm>
#include <array>
#include <catch2/catch.hpp>
#include <libtensor/expr/bispace/bispace.h>

#define CHECK_ELEMENTWISE(TENSOR, EXPRESSION)       \
  {                                                 \
    {                                               \
      std::vector<double> exported;                 \
      TENSOR->export_to(exported);                  \
      for (size_t i = 0; i < TENSOR->size(); ++i) { \
        INFO("Index is " << i);                     \
        CHECK(exported[i] == EXPRESSION);           \
      }                                             \
    }                                               \
  }

namespace libadcc {
namespace tests {

TEST_CASE("Test Tensor interface", "[tensor]") {
  auto adcmem_ptr = std::shared_ptr<AdcMemory>(new AdcMemory());

  SECTION("Basic tests") {
    libtensor::bispace<1> bis5(5);
    libtensor::bispace<2> bis55(bis5 & bis5);  // Symmetric 5x5 space
    libtensor::bispace<2> bia55(bis5 | bis5);  // Non-symmetric 5x5 space
    std::vector<AxisInfo> ax5{{"x", 5}};
    std::vector<AxisInfo> ax55{{"x", 5}, {"x", 5}};

    // Put some data on the stack:
    double vector1[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double vector2[] = {1.0, 0.0, 1.0, 0.0, 1.0};
    double vector3[] = {0.0, 1.0, 0.0, 1.0, 0.0};
    double matrix[]  = {1.1, 1.2, 1.3, 1.4, 1.5,  //
                       1.2, 2.2, 2.3, 2.4, 2.5,  //
                       1.3, 2.3, 3.3, 3.4, 3.5,  //
                       1.4, 2.4, 3.4, 4.4, 4.5,  //
                       1.5, 2.5, 3.5, 4.5, 5.5};
    double batrix[]  = {0.0,  1.2,  1.3,  1.4,  1.5,  //
                       -1.2, 0.0,  2.3,  2.4,  2.5,  //
                       -1.3, -2.3, 0.0,  3.4,  3.5,  //
                       -1.4, -2.4, -3.4, 0.0,  4.5,  //
                       -1.5, -2.5, -3.5, -4.5, 0.0};

    // And import it
    auto v1_ptr = std::make_shared<libtensor::btensor<1>>(bis5);
    auto v2_ptr = std::make_shared<libtensor::btensor<1>>(bis5);
    auto v3_ptr = std::make_shared<libtensor::btensor<1>>(bis5);
    auto m_ptr  = std::make_shared<libtensor::btensor<2>>(bis55);
    auto b_ptr  = std::make_shared<libtensor::btensor<2>>(bia55);

    libtensor::btod_import_raw<1>(vector1, bis5.get_bis().get_dims()).perform(*v1_ptr);
    libtensor::btod_import_raw<1>(vector2, bis5.get_bis().get_dims()).perform(*v2_ptr);
    libtensor::btod_import_raw<1>(vector3, bis5.get_bis().get_dims()).perform(*v3_ptr);
    libtensor::btod_import_raw<2>(matrix, bis55.get_bis().get_dims()).perform(*m_ptr);
    libtensor::btod_import_raw<2>(batrix, bia55.get_bis().get_dims()).perform(*b_ptr);

    // Enwrap them into a Tensor object
    std::shared_ptr<Tensor> v1_tensor_ptr = wrap_libtensor(adcmem_ptr, ax5, v1_ptr);
    std::shared_ptr<Tensor> v2_tensor_ptr = wrap_libtensor(adcmem_ptr, ax5, v2_ptr);
    std::shared_ptr<Tensor> v3_tensor_ptr = wrap_libtensor(adcmem_ptr, ax5, v3_ptr);
    std::shared_ptr<Tensor> m_tensor_ptr  = wrap_libtensor(adcmem_ptr, ax55, m_ptr);
    std::shared_ptr<Tensor> b_tensor_ptr  = wrap_libtensor(adcmem_ptr, ax55, b_ptr);

    INFO("vector v1: \n" << *v1_tensor_ptr);
    INFO("vector v2: \n" << *v2_tensor_ptr);
    INFO("vector v3: \n" << *v3_tensor_ptr);
    INFO("matrix m:  \n" << *m_tensor_ptr);
    INFO("matrix b:  \n" << *b_tensor_ptr);

    SECTION("Tensor size properties.") {
      CHECK(v1_tensor_ptr->ndim() == 1);
      CHECK(v1_tensor_ptr->shape()[0] == 5);
      CHECK(v1_tensor_ptr->size() == 5);

      CHECK(v2_tensor_ptr->ndim() == 1);
      CHECK(v2_tensor_ptr->shape()[0] == 5);
      CHECK(v2_tensor_ptr->size() == 5);

      CHECK(m_tensor_ptr->ndim() == 2);
      CHECK(m_tensor_ptr->shape()[0] == 5);
      CHECK(m_tensor_ptr->shape()[1] == 5);
      CHECK(m_tensor_ptr->size() == 25);
    }

    SECTION("export_results") {
      auto export_and_check = [](std::shared_ptr<Tensor> tensor, double* comparison) {
        {
          std::vector<double> exported(tensor->size());
          tensor->export_to(exported.data(), exported.size());

          for (size_t i = 0; i < tensor->size(); ++i) {
            INFO("Index is " << i);
            CHECK(exported[i] == comparison[i]);
          }
        }

        {
          std::vector<double> empty;
          tensor->export_to(empty);
          for (size_t i = 0; i < tensor->size(); ++i) {
            INFO("Index is " << i);
            CHECK(empty[i] == comparison[i]);
          }
        }
      };

      export_and_check(v1_tensor_ptr, vector1);
      export_and_check(v2_tensor_ptr, vector2);
      export_and_check(v3_tensor_ptr, vector3);
      export_and_check(m_tensor_ptr, matrix);
    }

    SECTION("import_from") {
      SECTION("v1 import") {
        auto v1new_ptr = v1_tensor_ptr->empty_like();
        v1new_ptr->import_from(vector1, 5);
        CHECK_ELEMENTWISE(v1new_ptr, vector1[i]);
      }

      SECTION("m import") {
        auto new_ptr = m_tensor_ptr->empty_like();
        new_ptr->import_from(matrix, 25);
        CHECK_ELEMENTWISE(new_ptr, matrix[i]);
      }

      SECTION("m import with error") {
        auto new_ptr = m_tensor_ptr->empty_like();

        // Tilt the data
        matrix[23] += 1e-8;
        matrix[5] -= 1e-8;
        new_ptr->import_from(matrix, 25, 1e-8);
        CHECK_ELEMENTWISE(new_ptr, matrix[i]);
      }
    }

    SECTION("scale function") {
      auto scale_and_check = [](std::shared_ptr<Tensor> tensor, double scalar,
                                double* comparison) {
        std::vector<double> exported;
        auto res = tensor->scale(scalar);

        res->export_to(exported);
        for (size_t i = 0; i < res->size(); ++i) {
          INFO("Index is " << i);
          CHECK(exported[i] == scalar * comparison[i]);
        }
      };

      scale_and_check(v1_tensor_ptr, 15, vector1);
      scale_and_check(v1_tensor_ptr, 0, vector1);
      scale_and_check(v2_tensor_ptr, -1, vector2);
      scale_and_check(v3_tensor_ptr, 3, vector3);
      scale_and_check(m_tensor_ptr, 0.1223, matrix);
    }

    SECTION("dot function") {
      CHECK(v1_tensor_ptr->dot({v2_tensor_ptr})[0] == 9);
      CHECK(v1_tensor_ptr->dot({v3_tensor_ptr})[0] == 6);
      CHECK(v2_tensor_ptr->dot({v1_tensor_ptr})[0] == 9);

      const std::vector<double> ret =
            v1_tensor_ptr->dot({v1_tensor_ptr, v2_tensor_ptr, v3_tensor_ptr});
      CHECK(ret[0] == 55);
      CHECK(ret[1] == 9);
      CHECK(ret[2] == 6);
    }

    SECTION("copy function") {
      auto copy_and_check = [](std::shared_ptr<Tensor> tensor, double* comparison) {
        std::vector<double> exported;
        std::vector<double> exported_orig;

        std::shared_ptr<Tensor> copy = tensor->copy();
        copy->export_to(exported);
        for (size_t i = 0; i < tensor->size(); ++i) {
          INFO("copy_check index is " << i);
          CHECK(exported[i] == comparison[i]);
        }

        const double scalar = 0.8;
        copy->scale(scalar)->export_to(exported);
        tensor->export_to(exported_orig);
        for (size_t i = 0; i < tensor->size(); ++i) {
          INFO("scale index is " << i);
          CHECK(exported[i] == scalar * comparison[i]);
          CHECK(exported_orig[i] == comparison[i]);
        }
      };

      copy_and_check(v1_tensor_ptr, vector1);
      copy_and_check(v2_tensor_ptr, vector2);
      copy_and_check(v3_tensor_ptr, vector3);
      copy_and_check(m_tensor_ptr, matrix);
    }

    SECTION("transpose function") {
      SECTION("transpose (symmetric) m") {
        auto res = m_tensor_ptr->transpose({1, 0});
        CHECK_ELEMENTWISE(res, matrix[i]);
      }

      SECTION("transpose (antisymmetric) b") {
        auto res = b_tensor_ptr->transpose({1, 0});
        CHECK_ELEMENTWISE(res, -batrix[i]);
      }

      SECTION("transpose arbitrary tensor") {
        std::shared_ptr<Tensor> tensor_ptr =
              random_tensor<4>(adcmem_ptr, {{5, 5, 5, 5}}, {{"a", "b", "c", "d"}});
        std::vector<scalar_type> t;
        tensor_ptr->export_to(t);

        auto ret_ptr = tensor_ptr->transpose({0, 2, 3, 1});

        auto compute_ref = [&t](size_t i) {
          const std::vector<size_t> strides{125, 25, 5, 1};
          // Compute actual indices from an absolute index i
          std::vector<size_t> indices{i / 125, (i / 25) % 5, (i / 5) % 5, i % 5};
          const size_t transposed = indices[0] * strides[0] + indices[1] * strides[2] +
                                    indices[2] * strides[3] + indices[3] * strides[1];
          return t[transposed];
        };
        CHECK_ELEMENTWISE(ret_ptr, compute_ref(i));
      }
    }

    SECTION("add function") {
      SECTION("add m += m") {
        auto res = m_tensor_ptr->add(m_tensor_ptr);
        CHECK_ELEMENTWISE(res, 2 * matrix[i]);
      }

      SECTION("add v1 += v2") {
        auto res = v1_tensor_ptr->add(v2_tensor_ptr);
        CHECK_ELEMENTWISE(res, vector1[i] + vector2[i]);
      }

      SECTION("add v1 += v2 + v3") {
        auto res = v1_tensor_ptr->add(v2_tensor_ptr->add(v3_tensor_ptr));
        CHECK_ELEMENTWISE(res, vector1[i] + vector2[i] + vector3[i]);
      }

      SECTION("add v1 += c * v2") {
        auto res = v1_tensor_ptr->add(v2_tensor_ptr->scale(-0.5));
        CHECK_ELEMENTWISE(res, vector1[i] - 0.5 * vector2[i]);
      }

      SECTION("add v1 += c * v2 + d * v3") {
        auto res = v1_tensor_ptr->add(v2_tensor_ptr->scale(-0.7))
                         ->add(v3_tensor_ptr->scale(28.3));
        CHECK_ELEMENTWISE(res, vector1[i] - 0.7 * vector2[i] + 28.3 * vector3[i]);
      }

      SECTION("add_lincomb v1 += c * v2 + d * v3") {
        auto res = v1_tensor_ptr->copy();
        res->add_linear_combination({{-0.7, 28.3}}, {{v2_tensor_ptr, v3_tensor_ptr}});
        CHECK_ELEMENTWISE(res, vector1[i] - 0.7 * vector2[i] + 28.3 * vector3[i]);
      }
    }

    SECTION("transpose add") {
      SECTION("transpose b.T + m") {
        auto res = b_tensor_ptr->transpose({1, 0})->add(m_tensor_ptr);
        CHECK_ELEMENTWISE(res, -batrix[i] + matrix[i]);
      }

      SECTION("transpose add arbitrary") {
        std::shared_ptr<Tensor> t1_ptr =
              random_tensor<4>(adcmem_ptr, {{5, 5, 5, 5}}, {{"a", "b", "c", "d"}});
        std::shared_ptr<Tensor> t2_ptr =
              random_tensor<4>(adcmem_ptr, {{5, 5, 5, 5}}, {{"d", "c", "b", "a"}});

        std::vector<scalar_type> t1;
        std::vector<scalar_type> t2;
        t1_ptr->export_to(t1);
        t2_ptr->export_to(t2);

        auto ret_ptr =
              t1_ptr->transpose({0, 2, 3, 1})->add(t2_ptr->transpose({3, 1, 0, 2}));

        auto compute_ref = [&t1, &t2](size_t i) {
          const std::vector<size_t> strides{125, 25, 5, 1};
          // Compute actual indices from an absolute index i
          std::vector<size_t> indices{i / 125, (i / 25) % 5, (i / 5) % 5, i % 5};
          const size_t idx1 = indices[0] * strides[0] + indices[1] * strides[2] +
                              indices[2] * strides[3] + indices[3] * strides[1];
          const size_t idx2 = indices[0] * strides[3] + indices[1] * strides[1] +
                              indices[2] * strides[0] + indices[3] * strides[2];
          return t1[idx1] + t2[idx2];
        };
        CHECK_ELEMENTWISE(ret_ptr, compute_ref(i));
      }
    }

    SECTION("multiply function") {
      SECTION("multiply m * m") {
        auto out_ptr = m_tensor_ptr->multiply(m_tensor_ptr);
        CHECK_ELEMENTWISE(out_ptr, matrix[i] * matrix[i]);
      }

      SECTION("multiply b * b") {
        auto out_ptr = b_tensor_ptr->multiply(b_tensor_ptr);
        CHECK_ELEMENTWISE(out_ptr, batrix[i] * batrix[i]);
      }

      SECTION("multiply v1 * v2") {
        auto out_ptr = v1_tensor_ptr->multiply(v2_tensor_ptr);
        CHECK_ELEMENTWISE(out_ptr, vector1[i] * vector2[i]);
      }

      SECTION("multiply m * b") {
        auto out_ptr = m_tensor_ptr->multiply(b_tensor_ptr);
        CHECK_ELEMENTWISE(out_ptr, matrix[i] * batrix[i]);
      }
    }

    SECTION("divide function") {
      SECTION("divide m / m") {
        auto out_ptr = m_tensor_ptr->divide(m_tensor_ptr);
        CHECK_ELEMENTWISE(out_ptr, matrix[i] / matrix[i]);
      }

      SECTION("divide v2 / v1") {
        auto out_ptr = v2_tensor_ptr->divide(v1_tensor_ptr);
        CHECK_ELEMENTWISE(out_ptr, vector2[i] / vector1[i]);
      }

      SECTION("divide b / m") {
        auto out_ptr = b_tensor_ptr->divide(m_tensor_ptr);
        CHECK_ELEMENTWISE(out_ptr, batrix[i] / matrix[i]);
      }
    }

    SECTION("set_mask function") {
      SECTION("b[ii] = 15") {
        b_tensor_ptr->set_mask("aa", 15);
        INFO("b after \n" << *m_tensor_ptr);
        auto compute_ref = [&batrix](size_t i) {
          // b is a 5x5 matrix, so the diagonal stride is 6
          return i % 6 == 0 ? 15 : batrix[i];
        };
        CHECK_ELEMENTWISE(b_tensor_ptr, compute_ref(i));
      }

      SECTION("m[ii] = 100") {
        m_tensor_ptr->set_mask("ii", 100);
        INFO("m after \n" << *m_tensor_ptr);
        auto compute_ref = [&matrix](size_t i) {
          // m is a 5x5 matrix, so the diagonal stride is 6
          return i % 6 == 0 ? 100 : matrix[i];
        };
        CHECK_ELEMENTWISE(m_tensor_ptr, compute_ref(i));
      }

      SECTION("m[ij] = -10") {
        m_tensor_ptr->set_mask("ij", -10);
        INFO("m after \n" << *m_tensor_ptr);
        CHECK_ELEMENTWISE(m_tensor_ptr, -10);
      }

      SECTION("tensor[ijki] = -0.1514") {
        std::shared_ptr<Tensor> tensor_ptr =
              random_tensor<4>(adcmem_ptr, {{5, 5, 5, 5}}, {{"a", "b", "c", "a"}});
        std::vector<scalar_type> t;
        tensor_ptr->export_to(t);

        tensor_ptr->set_mask("ijki", -0.1514);

        auto compute_ref = [&t](size_t i) {
          // Compute actual indices from an absolute index i
          std::vector<size_t> indices{i / 125, (i / 25) % 5, (i / 5) % 5, i % 5};
          if (indices[0] == indices[3]) {
            return -0.1514;
          } else {
            return t[i];
          }
        };
        CHECK_ELEMENTWISE(tensor_ptr, compute_ref(i));
      }
    }

    SECTION("empty_like function") {
      std::shared_ptr<Tensor> v1_empty = v1_tensor_ptr->empty_like();
      CHECK(v1_empty->ndim() == 1);
      CHECK(v1_empty->shape()[0] == 5);
      CHECK(v1_empty->size() == 5);

      std::shared_ptr<Tensor> m_empty = m_tensor_ptr->empty_like();
      CHECK(m_empty->ndim() == 2);
      CHECK(m_empty->shape()[0] == 5);
      CHECK(m_empty->shape()[1] == 5);
      CHECK(m_empty->size() == 25);
    }

    SECTION("zeros_like function") {
      std::shared_ptr<Tensor> v1_zeros = v1_tensor_ptr->zeros_like();
      CHECK(v1_zeros->ndim() == 1);
      CHECK(v1_zeros->shape()[0] == 5);
      CHECK(v1_zeros->size() == 5);
      CHECK_ELEMENTWISE(v1_zeros, 0);

      std::shared_ptr<Tensor> m_zeros = m_tensor_ptr->zeros_like();
      CHECK(m_zeros->ndim() == 2);
      CHECK(m_zeros->shape()[0] == 5);
      CHECK(m_zeros->shape()[1] == 5);
      CHECK(m_zeros->size() == 25);
      CHECK_ELEMENTWISE(v1_zeros, 0);

      std::shared_ptr<Tensor> b_zeros = b_tensor_ptr->zeros_like();
      CHECK(b_zeros->ndim() == 2);
      CHECK(b_zeros->shape()[0] == 5);
      CHECK(b_zeros->shape()[1] == 5);
      CHECK(b_zeros->size() == 25);
      CHECK_ELEMENTWISE(v1_zeros, 0);
    }

    SECTION("ones_like function") {
      auto check_ones = [](std::shared_ptr<Tensor> tensor) {
        std::vector<double> exported;
        tensor->export_to(exported);
        for (size_t i = 0; i < tensor->size(); ++i) {
          INFO("Index is " << i);
          CHECK(exported[i] == 1);
        }
      };

      std::shared_ptr<Tensor> v1_ones = v1_tensor_ptr->ones_like();
      CHECK(v1_ones->ndim() == 1);
      CHECK(v1_ones->shape()[0] == 5);
      CHECK(v1_ones->size() == 5);
      check_ones(v1_ones);

      std::shared_ptr<Tensor> m_ones = m_tensor_ptr->ones_like();
      CHECK(m_ones->ndim() == 2);
      CHECK(m_ones->shape()[0] == 5);
      CHECK(m_ones->shape()[1] == 5);
      CHECK(m_ones->size() == 25);
      check_ones(m_ones);

      std::shared_ptr<Tensor> b_ones = b_tensor_ptr->ones_like();
      CHECK(b_ones->ndim() == 2);
      CHECK(b_ones->shape()[0] == 5);
      CHECK(b_ones->shape()[1] == 5);
      CHECK(b_ones->size() == 25);
      check_ones(b_ones);
    }

  }  // SECTION basic tests

  SECTION("symmetrise and antisymmetrisation") {
    libtensor::bispace<1> bis3(TensorTestData::N);
    libtensor::bispace<4> bia3333(bis3 | bis3 | bis3 | bis3);
    std::vector<AxisInfo> ax3333(4, {"x", TensorTestData::N});

    auto in_ptr = std::make_shared<libtensor::btensor<4, scalar_type>>(bia3333);
    libtensor::btod_import_raw<4>(TensorTestData::a.data(), bia3333.get_bis().get_dims())
          .perform(*in_ptr);

    std::shared_ptr<Tensor> in_tensor_ptr = wrap_libtensor(adcmem_ptr, ax3333, in_ptr);

    SECTION("symmetrise_to function") {
      SECTION("symmetrise_to (0,1)") {
        std::shared_ptr<Tensor> out_tensor_ptr = in_tensor_ptr->symmetrise({{0, 1}});
        CHECK_ELEMENTWISE(out_tensor_ptr, TensorTestData::a_sym_01[i]);
      }

      SECTION("symmetrise_to {(0,1), (2,3)}") {
        std::shared_ptr<Tensor> out_tensor_ptr =
              in_tensor_ptr->symmetrise({{0, 1}, {2, 3}});
        CHECK_ELEMENTWISE(out_tensor_ptr, TensorTestData::a_sym_01_23[i]);
      }
    }

    SECTION("antisymmetrise_to function") {
      SECTION("antisymmetrise_to (0,1)") {
        std::shared_ptr<Tensor> out_tensor_ptr = in_tensor_ptr->antisymmetrise({{0, 1}});
        CHECK_ELEMENTWISE(out_tensor_ptr, TensorTestData::a_asym_01[i]);
      }

      SECTION("antisymmetrise_to {(0,1), (2,3)}") {
        std::shared_ptr<Tensor> out_tensor_ptr =
              in_tensor_ptr->antisymmetrise({{0, 1}, {2, 3}});
        CHECK_ELEMENTWISE(out_tensor_ptr, TensorTestData::a_asym_01_23[i]);
      }
    }
  }  // SECTION symmetrise antisymmetrise

  SECTION("direct_sum") {
    auto m1_ptr = make_tensor(adcmem_ptr, {{"a", 3}, {"b", 4}});
    auto m2_ptr = make_tensor(adcmem_ptr, {{"c", 2}, {"d", 3}});
    m1_ptr->import_from({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    m2_ptr->import_from({-1, 2, -3, 4, -5, 6});

    std::vector<double> ref1{-2, 1,  -4, 3,  -6,  5, 1, 4, -1, 6,  -3, 8,
                             -4, -1, -6, 1,  -8,  3, 3, 6, 1,  8,  -1, 10,
                             -6, -3, -8, -1, -10, 1, 5, 8, 3,  10, 1,  12};
    std::vector<double> ref2{0.,  3.,  -2., 5.,  -4., 7.,  1.,  4.,  -1., 6.,  -3., 8.,
                             2.,  5.,  0.,  7.,  -2., 9.,  3.,  6.,  1.,  8.,  -1., 10.,
                             4.,  7.,  2.,  9.,  0.,  11., 5.,  8.,  3.,  10., 1.,  12.,
                             6.,  9.,  4.,  11., 2.,  13., 7.,  10., 5.,  12., 3.,  14.,
                             8.,  11., 6.,  13., 4.,  15., 9.,  12., 7.,  14., 5.,  16.,
                             10., 13., 8.,  15., 6.,  17., 11., 14., 9.,  16., 7.,  18.};

    std::shared_ptr<Tensor> res_1 = m2_ptr->direct_sum(m2_ptr);
    std::shared_ptr<Tensor> res_2 = m1_ptr->direct_sum(m2_ptr);
    CHECK(res_1->size() == ref1.size());
    CHECK(res_2->size() == ref2.size());
    CHECK_ELEMENTWISE(res_1, ref1[i]);
    CHECK_ELEMENTWISE(res_2, ref2[i]);
  }  // SECTION direct_sum

  SECTION("tensordot") {
    SECTION("Scalar answer") {
      std::shared_ptr<Tensor> v1_ptr = random_tensor<1>(adcmem_ptr, {{4}}, {{"i"}});
      std::shared_ptr<Tensor> v2_ptr = random_tensor<1>(adcmem_ptr, {{4}}, {{"i"}});
      std::shared_ptr<Tensor> t1_ptr =
            random_tensor<4>(adcmem_ptr, {{2, 7, 8, 4}}, {{"i", "j", "k", "l"}});
      std::shared_ptr<Tensor> t2_ptr =
            random_tensor<4>(adcmem_ptr, {{8, 4, 7, 2}}, {{"k", "l", "j", "i"}});

      std::vector<double> v1_buffer;
      std::vector<double> v2_buffer;
      std::vector<double> t1_buffer;
      std::vector<double> t2_buffer;
      v1_ptr->export_to(v1_buffer);
      v2_ptr->export_to(v2_buffer);
      t1_ptr->export_to(t1_buffer);
      t2_ptr->transpose({3, 2, 0, 1})->export_to(t2_buffer);
      scalar_type ref_v = std::inner_product(v1_buffer.begin(), v1_buffer.end(),
                                             v2_buffer.begin(), 0.0);
      scalar_type ref_t = std::inner_product(t1_buffer.begin(), t1_buffer.end(),
                                             t2_buffer.begin(), 0.0);

      TensorOrScalar res_v1 = v1_ptr->tensordot(v2_ptr, {{0}, {0}});
      TensorOrScalar res_v2 = v2_ptr->tensordot(v1_ptr, {{0}, {0}});
      TensorOrScalar res_t1 = t1_ptr->tensordot(t2_ptr, {{0, 1, 2, 3}, {3, 2, 0, 1}});
      TensorOrScalar res_t2 = t2_ptr->tensordot(t1_ptr, {{0, 1, 2, 3}, {2, 3, 1, 0}});
      TensorOrScalar res_t3 = t2_ptr->tensordot(t1_ptr, {{2, 1, 0, 3}, {1, 3, 2, 0}});
      CHECK(res_v1.tensor_ptr == nullptr);
      CHECK(res_v2.tensor_ptr == nullptr);
      CHECK(res_t1.tensor_ptr == nullptr);
      CHECK(res_t2.tensor_ptr == nullptr);
      CHECK(res_t3.tensor_ptr == nullptr);
      CHECK(res_v1.scalar == Approx(ref_v));
      CHECK(res_v2.scalar == Approx(ref_v));
      CHECK(res_t1.scalar == Approx(ref_t));
      CHECK(res_t2.scalar == Approx(ref_t));
      CHECK(res_t3.scalar == Approx(ref_t));
    }

    SECTION("direct product") {
      auto m1_ptr = make_tensor(adcmem_ptr, {{"a", 3}, {"b", 4}});
      auto m2_ptr = make_tensor(adcmem_ptr, {{"c", 2}, {"d", 3}});
      m1_ptr->import_from({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
      m2_ptr->import_from({-1, 2, -3, 4, -5, 6});

      std::vector<double> ref1{1, -2,  3,  -4,  5,  -6,  -2, 4,  -6,  8,  -10, 12,
                               3, -6,  9,  -12, 15, -18, -4, 8,  -12, 16, -20, 24,
                               5, -10, 15, -20, 25, -30, -6, 12, -18, 24, -30, 36};
      std::vector<double> ref2{-1,  2,  -3,  4,  -5,  6,  -2,  4,  -6,  8,  -10, 12,
                               -3,  6,  -9,  12, -15, 18, -4,  8,  -12, 16, -20, 24,
                               -5,  10, -15, 20, -25, 30, -6,  12, -18, 24, -30, 36,
                               -7,  14, -21, 28, -35, 42, -8,  16, -24, 32, -40, 48,
                               -9,  18, -27, 36, -45, 54, -10, 20, -30, 40, -50, 60,
                               -11, 22, -33, 44, -55, 66, -12, 24, -36, 48, -60, 72};

      TensorOrScalar tos_1 = m2_ptr->tensordot(m2_ptr, {{}, {}});
      TensorOrScalar tos_2 = m1_ptr->tensordot(m2_ptr, {{}, {}});
      CHECK_FALSE(tos_1.tensor_ptr == nullptr);
      CHECK_FALSE(tos_2.tensor_ptr == nullptr);
      auto res_1 = tos_1.tensor_ptr;
      auto res_2 = tos_2.tensor_ptr;

      CHECK(res_1->size() == ref1.size());
      CHECK(res_2->size() == ref2.size());
      CHECK_ELEMENTWISE(res_1, ref1[i]);
      CHECK_ELEMENTWISE(res_2, ref2[i]);
    }

    SECTION("(1, 2, 2) == matrix-matrix multiplication") {
      std::shared_ptr<Tensor> a_ptr =
            random_tensor<2>(adcmem_ptr, {{4, 3}}, {{"a", "b"}});
      std::shared_ptr<Tensor> b_ptr =
            random_tensor<2>(adcmem_ptr, {{3, 2}}, {{"b", "c"}});

      TensorOrScalar res = a_ptr->tensordot(b_ptr, {{1}, {0}});
      CHECK_FALSE(res.tensor_ptr == nullptr);
      std::shared_ptr<Tensor> out_ptr = res.tensor_ptr;

      // Compute reference
      std::vector<scalar_type> out(out_ptr->size());
      {
        std::vector<scalar_type> a;
        std::vector<size_t> astr{3, 1};
        a_ptr->export_to(a);

        std::vector<scalar_type> b;
        std::vector<size_t> bstr{2, 1};
        b_ptr->export_to(b);

        std::vector<size_t> outstr{2, 1};
        for (size_t i = 0; i < 4; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 2; ++k) {
              const size_t a_idx   = astr[0] * i + astr[1] * j;
              const size_t b_idx   = bstr[0] * j + bstr[1] * k;
              const size_t out_idx = outstr[0] * i + outstr[1] * k;
              out[out_idx] += a[a_idx] * b[b_idx];
            }
          }
        }

        CHECK_ELEMENTWISE(out_ptr, Approx(out[i]));
      }
    }  // SECTION (1, 2, 2) == matrix-matrix multiplication

    SECTION("(1, 2, 2) perm") {
      std::shared_ptr<Tensor> a_ptr =
            random_tensor<2>(adcmem_ptr, {{3, 4}}, {{"b", "a"}});
      std::shared_ptr<Tensor> b_ptr =
            random_tensor<2>(adcmem_ptr, {{3, 2}}, {{"b", "c"}});

      TensorOrScalar res = a_ptr->tensordot(b_ptr, {{0}, {0}});
      CHECK_FALSE(res.tensor_ptr == nullptr);
      std::shared_ptr<Tensor> out_ptr = res.tensor_ptr;

      // Compute reference
      std::vector<scalar_type> out(out_ptr->size());
      {
        std::vector<scalar_type> a;
        std::vector<size_t> astr{4, 1};
        a_ptr->export_to(a);

        std::vector<scalar_type> b;
        std::vector<size_t> bstr{2, 1};
        b_ptr->export_to(b);

        std::vector<size_t> outstr{2, 1};
        for (size_t i = 0; i < 4; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 2; ++k) {
              const size_t a_idx   = astr[1] * i + astr[0] * j;
              const size_t b_idx   = bstr[0] * j + bstr[1] * k;
              const size_t out_idx = outstr[0] * i + outstr[1] * k;
              out[out_idx] += a[a_idx] * b[b_idx];
            }
          }
        }

        CHECK_ELEMENTWISE(out_ptr, Approx(out[i]));
      }
    }  // SECTION (1, 2, 2) perm

    SECTION("(1, 2, 4)") {
      size_t ni = 4, nj = 3, nk = 2, nl = 2, nm = 3;

      std::shared_ptr<Tensor> a_ptr =
            random_tensor<2>(adcmem_ptr, {{ni, nj}}, {{"ni", "nj"}});
      std::shared_ptr<Tensor> b_ptr =
            random_tensor<4>(adcmem_ptr, {{nk, nj, nl, nm}}, {{"nk", "nj", "nl", "nm"}});

      TensorOrScalar res = a_ptr->tensordot(b_ptr, {{1}, {1}});
      CHECK_FALSE(res.tensor_ptr == nullptr);
      std::shared_ptr<Tensor> out_ptr = res.tensor_ptr;

      // Compute reference
      std::vector<scalar_type> out(out_ptr->size());
      {
        std::vector<scalar_type> a;
        std::vector<size_t> astr{nj, 1};
        a_ptr->export_to(a);

        std::vector<scalar_type> b;
        std::vector<size_t> bstr{nj * nl * nm, nl * nm, nm, 1};
        b_ptr->export_to(b);

        std::vector<size_t> outstr{nk * nl * nm, nl * nm, nm, 1};
        for (size_t i = 0; i < ni; ++i) {
          for (size_t j = 0; j < nj; ++j) {
            for (size_t k = 0; k < nk; ++k) {
              for (size_t l = 0; l < nl; ++l) {
                for (size_t m = 0; m < nm; ++m) {
                  const size_t a_idx = astr[0] * i + astr[1] * j;
                  const size_t b_idx =
                        bstr[0] * k + bstr[1] * j + bstr[2] * l + bstr[3] * m;
                  const size_t out_idx =
                        outstr[0] * i + outstr[1] * k + outstr[2] * l + outstr[3] * m;
                  out.at(out_idx) += a.at(a_idx) * b.at(b_idx);
                }  // m
              }    // l
            }      // k
          }        // j
        }          // i

        CHECK_ELEMENTWISE(out_ptr, Approx(out[i]));
      }
    }  // SECTION (1, 2, 4)
  }    // SECTION tensordot

  SECTION("trace") {
    SECTION("Matrix trace") {
      std::shared_ptr<Tensor> m_ptr =
            random_tensor<2>(adcmem_ptr, {{3, 3}}, {{"a", "a"}});
      std::vector<double> buf;
      m_ptr->export_to(buf);
      const double ref = buf[0] + buf[4] + buf[8];
      CHECK(m_ptr->trace("aa") == ref);
    }

    SECTION("Tensor trace iijj") {
      std::shared_ptr<Tensor> t_ptr =
            random_tensor<4>(adcmem_ptr, {{3, 3, 4, 4}}, {{"a", "a", "b", "b"}});
      std::vector<double> buf;
      t_ptr->export_to(buf);
      double ref = 0;
      const std::vector<size_t> strides{48, 16, 4, 1};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          ref += buf[(strides[0] + strides[1]) * i + (strides[2] + strides[3]) * j];
        }
      }
      CHECK(t_ptr->trace("iijj") == ref);
    }

    SECTION("Tensor trace ijij") {
      std::shared_ptr<Tensor> t_ptr =
            random_tensor<4>(adcmem_ptr, {{3, 4, 3, 4}}, {{"a", "b", "a", "b"}});
      std::vector<double> buf;
      t_ptr->export_to(buf);
      double ref = 0;
      const std::vector<size_t> strides{48, 12, 4, 1};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          ref += buf[(strides[0] + strides[2]) * i + (strides[1] + strides[3]) * j];
        }
      }
      CHECK(t_ptr->trace("ijij") == ref);
    }

    SECTION("Tensor trace ijji") {
      std::shared_ptr<Tensor> t_ptr =
            random_tensor<4>(adcmem_ptr, {{3, 4, 4, 3}}, {{"a", "b", "b", "a"}});
      std::vector<double> buf;
      t_ptr->export_to(buf);
      double ref = 0;
      const std::vector<size_t> strides{48, 12, 3, 1};
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 4; ++j) {
          ref += buf[(strides[0] + strides[3]) * i + (strides[1] + strides[2]) * j];
        }
      }
      CHECK(t_ptr->trace("ijji") == ref);
    }
  }  // SECTION trace

  SECTION("element access") {
    libtensor::bispace<1> bis6(6);
    bis6.split(2).split(4);
    libtensor::bispace<2> bis66(bis6 | bis6);  // Non-symmetric 5x5 space
    std::vector<std::vector<size_t>> splits = get_block_starts(bis66.get_bis());
    std::vector<AxisInfo> ax55{{"x", 6, 0, splits[0]}, {"y", 6, 0, splits[1]}};

    // Import some data on the stack:
    double matrix[] = {0.0,  1.1,  1.2,  1.3,  1.4,  1.5,  //
                       -1.2, 0.0,  0.0,  2.3,  5.4,  2.5,  //
                       -1.3, -1.3, -2.3, 5.0,  3.4,  3.5,  //
                       -1.4, -1.4, -2.4, -3.4, 0.0,  4.5,  //
                       -1.5, -1.5, -2.5, -3.5, -4.5, 0.0,  //
                       2.2,  2.2,  4.4,  -1.3, 3.1,  -1.1};

    auto m_ptr = std::make_shared<libtensor::btensor<2>>(bis66);
    libtensor::btod_import_raw<2>(matrix, bis66.get_bis().get_dims()).perform(*m_ptr);
    std::shared_ptr<Tensor> m_tensor_ptr = wrap_libtensor(adcmem_ptr, ax55, m_ptr);

    SECTION("get_element") {
      CHECK(m_tensor_ptr->get_element({2, 3}) == 5);
      CHECK(m_tensor_ptr->get_element({1, 1}) == 0);
      CHECK(m_tensor_ptr->get_element({5, 2}) == 4.4);
      CHECK(m_tensor_ptr->get_element({1, 5}) == 2.5);
    }  // SECTION get_element

    SECTION("set_element") {
      m_tensor_ptr->set_element({2, 3}, 99);
      m_tensor_ptr->set_element({1, 1}, -20);
      m_tensor_ptr->set_element({5, 2}, -2.3);
      m_tensor_ptr->set_element({1, 5}, 5.5);

      double ref[] = {0.0,  1.1,   1.2,  1.3,  1.4,  1.5,  //
                      -1.2, -20.0, 0.0,  2.3,  5.4,  5.5,  //
                      -1.3, -1.3,  -2.3, 99.0, 3.4,  3.5,  //
                      -1.4, -1.4,  -2.4, -3.4, 0.0,  4.5,  //
                      -1.5, -1.5,  -2.5, -3.5, -4.5, 0.0,  //
                      2.2,  2.2,   -2.3, -1.3, 3.1,  -1.1};
      CHECK_ELEMENTWISE(m_tensor_ptr, ref[i]);
    }  // SECTION set_element

    SECTION("select_n_min") {
      std::vector<std::pair<std::vector<size_t>, scalar_type>> ref = {
            {{4, 4}, -4.5},  //
            {{4, 3}, -3.5},  //
            {{3, 3}, -3.4},  //
            {{4, 2}, -2.5},  //
            {{3, 2}, -2.4},  //
            {{2, 2}, -2.3},  //
      };

      for (size_t n = 1; n <= ref.size(); ++n) {
        auto res = m_tensor_ptr->select_n_min(n);
        for (size_t i = 0; i < n; ++i) {
          INFO("Element " << i + 1 << "/" << n);
          CHECK(res[i].first == ref[i].first);
          CHECK(res[i].second == ref[i].second);
        }
      }
    }  // select_n_min

    SECTION("select_n_max") {
      std::vector<std::pair<std::vector<size_t>, scalar_type>> ref = {
            {{1, 4}, 5.4},  //
            {{2, 3}, 5.0},  //
            {{3, 5}, 4.5},  //
            {{5, 2}, 4.4},  //
            {{2, 5}, 3.5},  //
            {{2, 4}, 3.4},  //
      };

      for (size_t n = 1; n <= ref.size(); ++n) {
        auto res = m_tensor_ptr->select_n_max(n);
        for (size_t i = 0; i < n; ++i) {
          INFO("Element " << i + 1 << "/" << n);
          CHECK(res[i].first == ref[i].first);
          CHECK(res[i].second == ref[i].second);
        }
      }
    }  // select_n_absmax

    SECTION("select_n_absmin") {
      std::vector<std::pair<std::vector<size_t>, scalar_type>> ref = {
            {{0, 1}, 1.1},   //
            {{5, 5}, -1.1},  //
            {{1, 0}, -1.2},  //
            {{0, 2}, 1.2},   //
            {{0, 3}, 1.3},   //
            {{2, 0}, -1.3},  //
            {{2, 1}, -1.3},  //
            {{5, 3}, -1.3},  //
            {{0, 4}, 1.4},   //
      };

      for (size_t n = 1; n <= ref.size(); ++n) {
        auto res = m_tensor_ptr->select_n_absmin(n);
        for (size_t i = 0; i < n; ++i) {
          INFO("Element " << i + 1 << "/" << n);
          CHECK(res[i].first == ref[i].first);
          CHECK(res[i].second == ref[i].second);
        }
      }
    }  // select_n_absmin

    SECTION("select_n_absmax") {
      std::vector<std::pair<std::vector<size_t>, scalar_type>> ref = {
            {{1, 4}, 5.4},   //
            {{2, 3}, 5.0},   //
            {{3, 5}, 4.5},   //
            {{4, 4}, -4.5},  //
            {{5, 2}, 4.4},   //
            {{2, 5}, 3.5},   //
            {{4, 3}, -3.5},  //
            {{3, 3}, -3.4},  //
            {{2, 4}, 3.4},   //
      };

      for (size_t n = 1; n <= ref.size(); ++n) {
        auto res = m_tensor_ptr->select_n_absmax(n);
        for (size_t i = 0; i < n; ++i) {
          INFO("Element " << i + 1 << "/" << n);
          CHECK(res[i].first == ref[i].first);
          CHECK(res[i].second == ref[i].second);
        }
      }
    }  // select_n_absmax

    // TODO is_element_allowed
  }  // SECTION element access

  //
  // New tests for lazy evaluation
  //

  SECTION("Lazy sum tests") {
    std::vector<AxisInfo> axes{{"a", 3, 3, {0, 3}}, {"b", 4, 4, {0, 4}}};

    auto A = make_tensor(adcmem_ptr, axes);
    auto B = make_tensor(adcmem_ptr, axes);
    auto C = make_tensor(adcmem_ptr, axes);
    A->set_random();
    B->set_random();
    C->set_random();

    std::vector<double> a_data;
    std::vector<double> b_data;
    std::vector<double> c_data;
    A->export_to(a_data);
    B->export_to(b_data);
    C->export_to(c_data);

    SECTION("Lazy sum properties.") {
      auto AB = A->add(B);
      CHECK(AB->ndim() == 2);
      CHECK(AB->size() == 48);
      CHECK(AB->shape()[0] == 6);
      CHECK(AB->shape()[1] == 8);
      CHECK(AB->needs_evaluation());
    }

    SECTION("Lazy sum values.") {
      auto AB = A->add(B);
      CHECK_ELEMENTWISE(AB, a_data[i] + b_data[i]);
      CHECK_FALSE(AB->needs_evaluation());
    }

    SECTION("Lazy sum evaluate values.") {
      auto AB = evaluate(A->add(B));
      CHECK_FALSE(AB->needs_evaluation());
      CHECK_ELEMENTWISE(AB, a_data[i] + b_data[i]);
    }
  }  // SECTION Lazy sum tests

  SECTION("Lazy scale tests") {
    std::vector<AxisInfo> axes{{"a", 3, 3, {0, 3}}, {"b", 4, 4, {0, 4}}};
    auto A = make_tensor(adcmem_ptr, axes);
    A->set_random();
    std::vector<double> a_data;
    A->export_to(a_data);

    SECTION("Lazy scale properties.") {
      auto cA = A->scale(1.2);

      CHECK(cA->ndim() == 2);
      CHECK(cA->size() == 48);
      CHECK(cA->shape()[0] == 6);
      CHECK(cA->shape()[1] == 8);
      CHECK(cA->needs_evaluation());
    }

    SECTION("Lazy scale values.") {
      auto cA = A->scale(1.2);
      CHECK_ELEMENTWISE(cA, 1.2 * a_data[i]);
      CHECK_FALSE(cA->needs_evaluation());
    }

    SECTION("Lazy scale evaluate values.") {
      auto cA = evaluate(A->scale(1.2));
      CHECK_FALSE(cA->needs_evaluation());
      CHECK_ELEMENTWISE(cA, 1.2 * a_data[i]);
    }
  }  // SECTION Lazy sum tests

}  // TEST_CASE
}  // namespace tests
}  // namespace libadcc
