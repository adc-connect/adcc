#include "metadata.hh"
#include <algorithm>
#include <libtensor/version.h>
#include <sstream>
#include <vector>

#ifdef ADCC_WITH_LIBXM
#include <adcc/exceptions.hh>
#include <libvmm/version.h>
#endif  // ADCC_WITH_LIBXM

namespace adcc {
namespace {
static const std::string static_version_string = "0.14.4";

static const std::vector<std::string> version_split = [](const std::string& in) {
  std::vector<std::string> parts;
  std::stringstream ss(in);
  std::string item;
  while (std::getline(ss, item, '.')) parts.push_back(item);
  return parts;
}(static_version_string);

static int get_version_part(size_t part) {
  int ret;
  std::stringstream ss(version_split[part]);
  ss >> ret;
  return ret;
}

std::string join_authors(std::list<std::string> li) {
  const size_t n_items = li.size();
  size_t i             = 0;
  std::string ret;
  for (auto& item : li) {
    ret.append(item);
    if (i == n_items - 2) {
      ret.append(" and ");
    } else if (i < n_items - 1) {
      ret.append(", ");
    }
    ++i;
  }
  return ret;
}

}  // namespace

int version::major_part() { return get_version_part(0); }
int version::minor_part() { return get_version_part(1); }
int version::patch_part() { return get_version_part(2); }
bool version::is_debug() {
#ifdef NDEBUG
  return false;
#else
  return true;
#endif  // NDEBUG
}

std::string version::version_string() { return static_version_string; }

std::vector<std::string> __features__() {
  std::vector<std::string> ret;
#ifdef ADCC_WITH_LIBXM
  ret.push_back("libxm");
#endif
  return ret;
}

// Feel free to add yourself below.
std::string __authors__() { return "Michael F. Herbst and Maximilian Scheurer"; }

std::string __email__() { return "developers@adc-connect.org"; }

/** Get the list of components compiled into adccore */
std::vector<Component> __components__() {
  std::vector<Component> ret;

  ret.push_back(Component{"libtensor", libtensor::version::get_string(),
                          join_authors(libtensor::version::get_authors()),
                          "C++ library for tensor computations.", "10.1002/jcc.23377",
                          "https://github.com/epifanovsky/libtensor",
                          "Boost Software License 1.0"});

#ifdef ADCC_WITH_LIBXM
  throw not_implemented_error("libxm version not implemented");
  ret.push_back(Component{"libxm", "?.?.?", "Ilya Kaliman", "Libxm Tensor Library.",
                          "10.1002/jcc.24713", "https://github.com/ilyak/libxm",
                          "ISC License"});
#endif  // ADCC_WITH_LIBXM

  std::sort(ret.begin(), ret.end(), [](const Component& lhs, const Component& rhs) {
    return lhs.name < rhs.name;
  });
  return ret;
}

}  // namespace adcc
