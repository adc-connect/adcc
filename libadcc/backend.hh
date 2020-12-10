#pragma once
#include <string>
#include <vector>

namespace libadcc {
/**
 *  \addtogroup Tensor
 */
///@{

/** Structure to hold information about the tensor backends */
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
}  // namespace libadcc
