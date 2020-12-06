# user-provided customisations for libadcc
#
# Available entities for configuration (for the defaults see the setup.py)

# libraries = []           # Libraries to link into libadcc (e.g. 'libtensorlight')
# library_dirs = []        # Directories to search for libraries
# include_dirs = []        # Extra directories to search for headers
# extra_link_args = []     # Extra arguments for the linker
# extra_compile_args = []  # Extra arguments for the compiler
# runtime_library_dirs = []  # Runtime library search directories
# extra_objects = []       # Extra objects to link in
# define_macros = []       # Extra macros to define
# search_system = True     # Search the system for libtensor or not
# download_missing = True  # Download libtensor automatically if missing on system
#
# Place to install libtensor to if missing on the system.
# Set to None to disable feature.
# libtensor_autoinstall = "~/.local"
import os

# Specify additional directories for pkg-config.
if False:
    # Typical use case: If libtensor has been installed to the prefix
    # /usr/local then this would make sure that it is automatically
    # found by the setup.py
    libtensor_prefix = "/usr/local"
    os.environ["PKG_CONFIG_PATH"] = os.path.join(libtensor_prefix, "lib/pkgconfig")
