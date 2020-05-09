cmake_minimum_required(VERSION 3.9)
project(SrgemmCudaKernels CUDA CXX)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

# RELEASE config by default ifnone is provided:
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE")
  set(BUILD_TYPE_INFERRED_RELEASE TRUE)
elseif()
  # otherwise, first convert build type string to uppercase, and then compare
  if(NOT uppercase_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELEASE|RELWITHDEBINFO)$")
    message(FATAL_ERROR "Invalid value for CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
  endif()
  set(BUILD_TYPE_INFERRED_RELEASE FALSE)
endif()

# always dump compiler invocation commands to compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Switches for testing, benchmarks or static lib builds
option(FWGPU_TEST   "FWGPU_TEST"   ON)
option(FWGPU_BENCH  "FWGPU_BENCH"  ON)
option(FWGPU_STATIC "FWGPU_STATIC" ON)

# static or dynamic lib build targets
if(FWGPU_STATIC)
  set(FWGPU_LIB_NAME fwgpu_static)
  set(FWGPU_LIB_TYPE STATIC)
else()
  set(FWGPU_LIB_NAME fwgpu)
  set(FWGPU_LIB_TYPE SHARED)
endif()

# CUDA native compiler (nvcc) only supports upto C++14 for now
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g3 -DDEBUG -Wall -Wextra -Wno-unused-parameter -fno-exceptions")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wall -fno-exceptions")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG -Wall -fno-exceptions")
# set(CMAKE_CXX_CLANG_TIDY clang-tidy -checks='bugprone-*,cert-*,clang-analyzer-core*,misc-*,performance-*')

# CUDA build flags
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  if(NOT CMAKE_CUDA_FLAGS)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS Auto)
  endif()
  set(CUDA_NVCC_FLAGS_DEBUG "-O0 -g -G -DDEBUG --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
  set(CUDA_NVCC_FLAGS_RELEASE "-O3 -DNDEBUG --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
  set(CUDA_NVCC_FLAGS_RELWITHDEBINFO "-O3 -g -G -DNDEBUG --expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS}")
endif()

# Source directory structure
include_directories(${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS})

if(FWGPU_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

if(FWGPU_BENCH)
  add_subdirectory(bench)
endif()

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
)

target_include_directories(${FWGPU_LIB_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}
  "${PROJECT_SOURCE_DIR}/cutlass/include"
  ${CUDA_INCLUDE_DIRS})

install(TARGETS ${FWGPU_LIB_NAME} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY ${FWGPU_LIB_NAME}/include DESTINATION include)

message(STATUS "")
message(STATUS "BUILD SUMMARY:")
message(STATUS "  Build type           : ${uppercase_CMAKE_BUILD_TYPE}")
message(STATUS "  CMAKE_GENERATOR      : ${CMAKE_GENERATOR}")
message(STATUS "  C++ Compiler         : ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Compiler version : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CUDA Compiler        : ${CMAKE_CUDA_COMPILER}")
message(STATUS "  CUDA Compiler version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "  Library name         : ${FWGPU_LIB_NAME}")
message(STATUS "  Library type         : ${FWGPU_LIB_TYPE}")
message(STATUS "  Build tests          : ${FWGPU_TEST}")
message(STATUS "  Build benchmarks     : ${FWGPU_BENCH}")
message(STATUS "  Found CUDA?          : ${CUDA_FOUND}")
message(STATUS "  CXX flags            : ${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}")
message(STATUS "  CUDA flags           : ${CUDA_NVCC_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}")
if (BUILD_TYPE_INFERRED_RELEASE)
  message(WARNING "No build type provided, defaulted to RELEASE configuration.")
endif()