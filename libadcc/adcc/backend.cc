#include "backend.hh"
#include <libtensor/metadata.h>

namespace adcc {

TensorBackend tensor_backend() {
  return TensorBackend{
        "libtensorlight",  // name
        libtensor::metadata::version_string(),
        libtensor::metadata::features(),
        libtensor::metadata::blas(),
        libtensor::metadata::authors(),
  };
}

}  // namespace adcc
