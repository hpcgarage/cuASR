cmake_minimum_required(VERSION 3.9)
project(SuperFWGPUKernels CUDA CXX)

# RELEASE config by default:
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
  set(BUILD_TYPE_INFERRED_RELEASE TRUE)
endif ()

# Switches for testing, benchmarks or static lib builds
option(FWGPU_TEST   "FWGPU_TEST"   ON)
option(FWGPU_BENCH  "FWGPU_BENCH"  ON)
option(FWGPU_STATIC "FWGPU_STATIC" ON)

# static or dynamic lib build targets
if (FWGPU_STATIC)
  set(FWGPU_LIB_NAME fwgpu_static)
  set(FWGPU_LIB_TYPE STATIC)
else ()
  set(FWGPU_LIB_NAME fwgpu)
  set(FWGPU_LIB_TYPE SHARED)
endif()

# CUDA native compiler (nvcc) only supports upto C++14 for now
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -fno-exceptions -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -Wall -Wextra -fno-exceptions -DDEBUG -g")

# CUDA build flags
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
  if (NOT CMAKE_CUDA_FLAGS)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
  endif()
  set(CUDA_NVCC_FLAGS_RELEASE "-O3 --expt-relaxed-constexpr -DNDEBUG ${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
  set(CUDA_NVCC_FLAGS_DEBUG "-O0 --expt-relaxed-constexpr -DDEBUG -g ${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
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

# fwgpu library configuration
set(fwgpu_src
  ./src/cublas_sgemm.cu
  ./src/cutlass_sgemm.cu
  ./src/cutlass_srgemm.cu
  ./src/utils.cu
)

cuda_add_library(${FWGPU_LIB_NAME} ${FWGPU_LIB_TYPE} ${fwgpu_src})

target_link_libraries(
  ${FWGPU_LIB_NAME}
  ${CUDA_CUBLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

include_directories(
  ${PROJECT_SOURCE_DIR}
  "${PROJECT_SOURCE_DIR}/cutlass/include"
  ${CUDA_INCLUDE_DIRS})

install(TARGETS ${FWGPU_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ${FWGPU_LIB_NAME}/include DESTINATION include)

message(STATUS "")
message(STATUS "BUILD SUMMARY:")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  C++ Compiler         : ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Compiler version : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA Compiler        : ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  Build type           : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Library name         : ${FWGPU_LIB_NAME}")
message(STATUS "  Library type         : ${FWGPU_LIB_TYPE}")
message(STATUS "  Build tests          : ${FWGPU_TEST}")
message(STATUS "  Build benchmarks     : ${FWGPU_BENCH}")
message(STATUS "  Found CUDA?          : ${CUDA_FOUND}")
if("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
  message(STATUS "  CXX flags            : ${CMAKE_CXX_FLAGS_RELEASE}")
  message(STATUS "  CUDA flags           : ${CUDA_NVCC_FLAGS_RELEASE}")
else()
  message(STATUS "  CXX flags            : ${CMAKE_CXX_FLAGS_DEBUG}")
  message(STATUS "  CUDA flags           : ${CUDA_NVCC_FLAGS_DEBUG}")
endif()
if (BUILD_TYPE_INFERRED_RELEASE)
  message(STATUS "WARNING: No build type provided, defaulted to RELEASE configuration.")
endif()
message(STATUS "")
