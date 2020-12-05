#pragma once
#include <string>
#include <vector>

namespace adcc {
/**
 *  \defgroup Metadata Metadata about adccore
 */
///@{

/**Version information */
struct version {
  /** Return the major part of the version */
  static int major_part();

  /** Return the minor part of the version */
  static int minor_part();

  /** Return the patch part of the version */
  static int patch_part();

  /** Is the compiled version a Debug version */
  static bool is_debug();

  /**  Return the version as a string */
  static std::string version_string();
};

/**Data about a library or third-party component*/
struct Component {
  std::string name;         //!< The name of the component
  std::string version;      //!< The version string
  std::string authors;      //!< The authors
  std::string description;  //!< A brief description
  std::string doi;          //!< DOI to a publication
  std::string website;      //!< Website of the upstream source
  std::string licence;      //!< A short identifier of the licence
};

/** Return the list of compiled-in features */
std::vector<std::string> __features__();

/** Return the authors string */
std::string __authors__();

/** Return the email string */
std::string __email__();

/** Get the list of components compiled into adccore */
std::vector<Component> __components__();

///@}
}  // namespace adcc
