#include "../exceptions.hh"
#include "util.hh"
#include <cstddef>
#include <pybind11/pybind11.h>

namespace libadcc {

namespace py = pybind11;

// Small wrapper around py::array_t to enable the correct
// type hints for the dimensionality of numpy arrays in the stub file.
// Important difference: In contrast to py::array_t<T> other types are not
// silently casted (and copied) to T if necessary (e.g. int -> double).
// Instead this class throws if it is constructed from e.g. std::array_t<int>
template <typename T, size_t Ndim>
class NDArray : public py::array_t<T> {
 public:
  // Needed by the type caster: it builds the value via reinterpret_borrow /
  // reinterpret_steal, which require the (handle, borrowed_t/stolen_t) ctors.
  using py::array_t<T>::array_t;

  // Converting constructor
  NDArray(py::array array)
        : py::array_t<T>(py::reinterpret_borrow<py::array_t<T>>(array)) {
    if (!py::isinstance<py::array_t<T>>(array)) throw runtime_error("Invalid array type");
  }
};
}  // namespace libadcc

namespace pybind11 {
namespace detail {

// Create a string of Ndim "int, int, ..." as shape for the np.ndarray type hint
template <size_t Ndim>
struct NdimName {
  static constexpr auto value = NdimName<Ndim - 1>::value + const_name(", int");
};
template <>
struct NdimName<1> {
  static constexpr auto value = const_name("int");
};

template <typename T, std::size_t Ndim>
struct handle_type_name<libadcc::NDArray<T, Ndim>> {
  static constexpr auto name = const_name("numpy.ndarray[tuple[") +
                               NdimName<Ndim>::value + const_name("], numpy.dtype[") +
                               npy_format_descriptor<T>::name + const_name("]]");
};

}  // namespace detail
}  // namespace pybind11
