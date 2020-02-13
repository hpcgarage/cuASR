cmake_minimum_required(VERSION 3.9)
project(SuperFWGPUKernels)

# RELEASE config by default:
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "No build type provided, default to RELEASE.")
  set(CMAKE_BUILD_TYPE "RELEASE")
endif ()

# Switches for testing, benchmarks or static lib builds
option(FWGPU_TEST   "FWGPU_TEST"   ON)
option(FWGPU_BENCH  "FWGPU_BENCH"  ON)
option(FWGPU_STATIC "FWGPU_STATIC" OFF)

# static or dynamic lib build targets
if (FWGPU_STATIC)
  set(FWGPU_LIB_NAME fwgpu_static)
  set(FWGPU_LIB_TYPE STATIC)
else ()
  set(FWGPU_LIB_NAME fwgpu)
  set(FWGPU_LIB_TYPE SHARED)
endif()

# CUDA build flags
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  if (NOT CMAKE_CUDA_FLAGS)
    set(CMAKE_CUDA_FLAGS "-arch=sm_60")
  endif()

  if (NOT CMAKE_CUDA_FLAGS_RELEASE)
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Wall ${CMAKE_CUDA_FLAGS}")
  endif()

  if (NOT CMAKE_CUDA_FLAGS_DEBUG)
    set(CMAKE_CUDA_FLAGS_DEBUG   "-O0 -DDEBUG -g ${CMAKE_CUDA_FLAGS}")
  endif()
endif()

# Source directory structure
add_subdirectory(src)
include_directories(${PROJECT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})

if (FWGPU_TEST)
  enable_testing()
  add_subdirectory(test)
endif ()

if (FWGPU_BENCH)
  add_subdirectory(bench)
endif ()

# C++11 Project for now, bump up later if needed
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fno-rtti -fno-exceptions -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -Wall -Wextra -fno-exceptions -DDEBUG -g")

message(STATUS "")
message(STATUS "BUILD SUMMARY")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  Compiler ID          : ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library name         : ${FWGPU_LIB_NAME}")
message(STATUS "  Library type         : ${FWGPU_LIB_TYPE}")
message(STATUS "  Build tests          : ${FWGPU_TEST}")
message(STATUS "  Build benchmarks     : ${FWGPU_BENCH}")
message(STATUS "  FoundCUDA            : ${CUDA_FOUND}")
message(STATUS "  CXX release flags    : ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  CXX debug flags      : ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "  CUDA flags           : ${CMAKE_CUDA_FLAGS}")
message(STATUS "  CUDA release flags   : ${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "  CUDA debug flags     : ${CMAKE_CUDA_FLAGS_DEBUG}")
message(STATUS "WARNING: No build type provided, default to RELEASE.")
message(STATUS "")
