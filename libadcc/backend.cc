#include "backend.hh"
#include <libtensor/metadata.h>

namespace libadcc {

TensorBackend tensor_backend() {
  return TensorBackend{
        "libtensorlight",  // name
        libtensor::metadata::version_string(),
        libtensor::metadata::features(),
        libtensor::metadata::blas(),
        libtensor::metadata::authors(),
  };
}

}  // namespace libadcc
