#
# pybind11
#
add_subdirectory(pybind11)

#
# adccore
#
include(adccore/share/adccore/AdcCore.cmake)

option(ADCCORE_INSTALL "Install adcccore or just compile" ON)
if (ADCCORE_INSTALL AND adccore_SHARED_BUILD)
	message(FATAL_ERROR
"This version of adcc is not installable, \
since adcccore.so is compiled as a shared library."
	)
endif()

if (NOT adccore_EXPECTED_VERSION VERSION_EQUAL adccore_VERSION)
	message(FATAL_ERROR
"AdcCore expected version (== ${adccore_EXPECTED_VERSION}) and actual \
adcc version (== ${adccore_VERSION}) do not match."
	)
endif()
