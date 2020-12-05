#pragma once
#include <string>
#include <vector>

namespace adcc {
/**
 *  \defgroup Information about the tensor backend
 */
///@{

struct TensorBackend {
  std::string name;
  std::string version;
  std::vector<std::string> features;
  std::string blas;
  std::string authors;
};

/** Get some info about libtensor */
TensorBackend tensor_backend();

///@}
}  // namespace adcc
