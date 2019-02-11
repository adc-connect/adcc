## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------

cmake_minimum_required(VERSION 3.2.0)

##########################################################################
# Project setup

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "The type of build")
set(BUILD_SHARED_LIBS ON CACHE
	BOOL "Build shared libraries (forced in this project)")
if(NOT BUILD_SHARED_LIBS)
	message("Notice: BUILD_SHARED_LIBS == ON is forced for adcc.")
endif()

project(adcc VERSION 0.6.0)
set(adccore_EXPECTED_VERSION "${PROJECT_VERSION}")
enable_testing()
if (CMAKE_VERSION VERSION_GREATER 3.5.0)
	set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

##########################################################################
# Setup compiler flags
include(CheckCXXCompilerFlag)

# C++ standard of this project is 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

macro(enable_if_cxx_compiles VARIABLE FLAG)
        # Checks whether a cxx compiler supports a flag and if yes
        # adds it to the variable provided.
        #
        string(REGEX REPLACE "[^a-zA-Z0-9]" "" FLAG_CLEAN "${FLAG}")
        CHECK_CXX_COMPILER_FLAG("-Werror ${FLAG}" DRB_HAVE_FLAG_${FLAG_CLEAN})
        if (DRB_HAVE_FLAG_${FLAG_CLEAN})
                set(${VARIABLE} "${${VARIABLE}} ${FLAG}")
        endif()
        unset(FLAG_CLEAN)
endmacro(enable_if_cxx_compiles)

# Standard flags, used without check
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall -Wextra -pedantic")

# Really useful and sensible flags to check for common errors
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wnon-virtual-dtor")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Woverloaded-virtual")
# enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wshadow")  # TODO some issues in ctx
# enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wold-style-cast") # Due to pybind11
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wcast-align")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wconversion")
# enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wuseless-cast")   # Due to pybind11
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wsign-conversion")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wmisleading-indentation")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wduplicated-cond")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wduplicated-branches")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wlogical-op")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wnull-dereference")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wdouble-promotion")
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wformat=2")

# These we rather want as warnings, not errors
enable_if_cxx_compiles(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations")

include_directories("${CMAKE_CURRENT_LIST_DIR}")

##########################################################################
# Externals and targets

# This directory sets up the external projects
add_subdirectory(external)

set(ADCC_PYTHON_SOURCES
	ExportAdcc.cc
	export_AdcIntermediates.cc
	export_AdcMatrix.cc
	export_AdcMemory.cc
	export_AmplitudeVector.cc
	export_amplitude_vector_enforce_spin_kind.cc
	export_compute_one_particle_densities.cc
	export_CtxMap.cc
	export_HartreeFockProvider.cc
	export_HfData.cc
	export_LazyMp.cc
	export_OneParticleDensityMatrix.cc
	export_ReferenceState.cc
	export_solve_adcman_davidson.cc
	export_Tensor.cc
	export_ThreadPool.cc
	export_tmp_run_prelim.cc
	util.cc
)

link_directories(${adccore_LINK_DIRECTORIES})
add_library(adcc SHARED ${ADCC_PYTHON_SOURCES})
target_link_libraries(adcc
	PUBLIC Adcc::adccore ${PYTHON_LIBRARIES}
	PRIVATE pybind11::module
)
set_target_properties(adcc
	PROPERTIES
	PREFIX "lib${PYTHON_MODULE_PREFIX}"
	SUFFIX "${PYTHON_MODULE_EXTENSION}"
	LINKER_LANGUAGE "CXX"
	VERSION "${PROJECT_VERSION}"
)
