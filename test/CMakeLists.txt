add_subdirectory(gtest)

include_directories(gtest/googletest/include)

### Matrix tests
set(Matrix_test_src
  harness.cpp
  Matrix_test.cpp
)
add_executable(Matrix_tests ${Matrix_test_src})
target_link_libraries(Matrix_tests
  gtest
  ${FWGPU_LIB_NAME}
)
add_test(
  NAME Matrix_tests
  COMMAND Matrix_tests
)

### GEMM tests
set(Gemm_test_src
  harness.cpp
  GemmEntry.cu
  Gemm_test.cpp
)
add_executable(Gemm_tests ${Gemm_test_src})
target_link_libraries(Gemm_tests
  gtest
  ${FWGPU_LIB_NAME}
)
add_test(
  NAME Gemm_tests
  COMMAND Gemm_tests
)

### SemiRing GEMM tests
set(Srgemm_test_src
  harness.cpp
  GemmEntry.cu
  Srgemm_test.cpp
)
add_executable(Srgemm_tests ${Srgemm_test_src})
target_link_libraries(Srgemm_tests
  gtest
  ${FWGPU_LIB_NAME}
)
add_test(
  NAME Srgemm_tests
  COMMAND Srgemm_tests
)
