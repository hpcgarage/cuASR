cmake_minimum_required(VERSION 3.13)
project(cuASR CUDA CXX)

# RELEASE config by default if none is provided:
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
  set(BUILD_TYPE_INFERRED_RELEASE TRUE)
endif()

# first convert build type string to uppercase, and then compare
string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE|RELWITHDEBINFO)$")
  message(FATAL_ERROR "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
endif()

# always dump compiler invocation commands to compile_commands.json
# note that this does not include nvcc invocations >:(
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Switches for testing, benchmarks and examples
option(CUASR_TEST    "Build cuASR test suite. Use with CUASR_TEST_LEVEL={0|1|2}." ON)
option(CUASR_BENCH   "Build cuASR benchmark suite."  ON)
option(CUASR_EXAMPLE "Build cuASR examples."  ON)

# By default, build fat binaries. TODO add sm_80 here
option(CUASR_CUDA_ARCHS "List of CUDA architectures to compile for." "80")

# CUDA native compiler (nvcc) only supports upto C++14 for now
find_package(CUDA REQUIRED)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# C++ compiler flags for target compile options
set(cuASR_CXX_FLAGS -Wall -Wextra -Wno-unused-parameter -Wno-uninitialized -Wno-strict-aliasing)
set(cuASR_CXX_FLAGS_DEBUG          -O0 -g3 -DDEBUG ${cuASR_CXX_FLAGS})
set(cuASR_CXX_FLAGS_RELEASE        -O3 -DNDEBUG ${cuASR_CXX_FLAGS})
set(cuASR_CXX_FLAGS_RELWITHDEBINFO -O3 -g3 -DNDEBUG ${cuASR_CXX_FLAGS})

# CUDA compiler flags for target compile options
set(cuASR_CUDA_FLAGS --expt-relaxed-constexpr)
set(cuASR_CUDA_FLAGS_DEBUG          -G ${cuASR_CUDA_FLAGS})
set(cuASR_CUDA_FLAGS_RELEASE        -O3 ${cuASR_CUDA_FLAGS})
set(cuASR_CUDA_FLAGS_RELWITHDEBINFO -G ${cuASR_CUDA_FLAGS})
set(CMAKE_CUDA_ARCHITECTURES ${CUASR_CUDA_ARCHS})

# the sub-modules update themselves with git, so find git
find_package(Git QUIET)

# make sure we have cutlass checked-out and is up-to-date
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  message(STATUS "Checking submodule version for NVIDIA/cutlass")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} submodule update --init ${PROJECT_SOURCE_DIR}/cutlass
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_SUBMOD_STDOUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE GIT_SUBMOD_STDERR ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE GIT_SUBMOD_RESULT
  )
  if(NOT GIT_SUBMOD_RESULT EQUAL "0")
    message(FATAL_ERROR "git submodule update --init failed with ${GIT_SUBMOD_RESULT}, please checkout cutlass manually. Git stdout was ${GIT_SUBMOD_STDOUT}. Git stderr was ${GIT_SUBMOD_STDERR}.")
  elseif(NOT ${GIT_SUBMOD_STDOUT} STREQUAL "")
    message(STATUS ${GIT_SUBMOD_STDOUT})
  endif()
endif()

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include")
  message(FATAL_ERROR "Cutlass submodule is not present and automatic checkout failed, please checkout cutlass manually.")
endif()

if(CUASR_TEST)
  enable_testing()
  add_subdirectory(test)
  if(NOT DEFINED CUASR_TEST_LEVEL)
    set(CUASR_TEST_LEVEL 0)
  endif()
endif()

if(CUASR_BENCH)
  add_subdirectory(bench)
  if(NOT DEFINED CUASR_BENCH_LEVEL)
    set(CUASR_BENCH_LEVEL 0)
  endif()
endif()

if(CUASR_EXAMPLE)
  add_subdirectory(examples)
endif()

message(STATUS "")
message(STATUS "BUILD SUMMARY:")
message(STATUS "  Build type           : ${uppercase_CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  C++ Compiler         : ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Compiler version : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA Compiler        : ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  Build tests          : ${CUASR_TEST}")
message(STATUS "  Test level           : ${CUASR_TEST_LEVEL}")
message(STATUS "  Build benchmarks     : ${CUASR_BENCH}")
message(STATUS "  Bench level          : ${CUASR_BENCH_LEVEL}")
message(STATUS "  Build examples       : ${CUASR_EXAMPLE}")
message(STATUS "  Found CUDA?          : ${CUDA_FOUND}")
message(STATUS "  CXX flags            : ${cuASR_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}")
message(STATUS "  CUDA flags           : ${cuASR_CUDA_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}")
if (BUILD_TYPE_INFERRED_RELEASE)
  message(WARNING "No build type provided, defaulted to RELEASE configuration.")
endif()
