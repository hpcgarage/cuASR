# cuasr library configuration
add_library(deprecated_libfwgpu ${cuASR_LIB_TYPE}
  ./src/cutlass_srgemm.cu
  ./src/utils.cu
)
target_include_directories(deprecated_libfwgpu
  PUBLIC ${PROJECT_SOURCE_DIR}/include ${CUDA_INCLUDE_DIRS}
  PRIVATE ${PROJECT_SOURCE_DIR}/cutlass/include
  PRIVATE ${PROJECT_SOURCE_DIR}/test/regress/include
)
target_compile_options(deprecated_libfwgpu
  PUBLIC
  # C++ compiler flags
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:${uppercase_CMAKE_BUILD_TYPE}>>:
    ${cuASR_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}>

  # CUDA compiler flags
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:${uppercase_CMAKE_BUILD_TYPE}>>:
    ${cuASR_CUDA_FLAGS_${uppercase_CMAKE_BUILD_TYPE}}>
)

### Matrix tests
add_executable(Matrix_tests ${PROJECT_SOURCE_DIR}/test/harness.cpp Matrix_test.cpp)
target_include_directories(Matrix_tests
  PRIVATE gtest/googletest/include
  PRIVATE ${PROJECT_SOURCE_DIR}/test/regress/include)
target_link_libraries(Matrix_tests gtest deprecated_libfwgpu)
add_test(
  NAME Matrix_tests
  COMMAND Matrix_tests
)

### SemiRing GEMM tests
add_executable(tropical_gemm_tests ${PROJECT_SOURCE_DIR}/test/harness.cpp Srgemm_test.cu)
target_include_directories(tropical_gemm_tests
  PRIVATE gtest/googletest/include
  PRIVATE ${PROJECT_SOURCE_DIR}/test/regress/include)
target_link_libraries(tropical_gemm_tests gtest deprecated_libfwgpu)
add_test(
  NAME tropical_gemm_tests
  COMMAND tropical_gemm_tests
)
