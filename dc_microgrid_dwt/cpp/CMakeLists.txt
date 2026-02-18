cmake_minimum_required(VERSION 3.14)
project(microgrid_dsp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags for DSP performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find pybind11
find_package(pybind11 REQUIRED)

# Build the Python module
pybind11_add_module(microgrid_dsp
    dsp_core.cpp
    bindings.cpp
)

target_include_directories(microgrid_dsp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Install to the project root so Python can import it
install(TARGETS microgrid_dsp
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/..
)
