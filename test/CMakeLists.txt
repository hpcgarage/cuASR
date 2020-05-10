add_subdirectory(gtest)

### Matrix tests
cuda_add_executable(Matrix_tests harness.cpp Matrix_test.cpp)
target_include_directories(Matrix_tests PRIVATE gtest/googletest/include)
target_link_libraries(Matrix_tests gtest ${FWGPU_LIB_NAME})
add_test(
  NAME Matrix_tests
  COMMAND Matrix_tests
)

### GEMM tests
cuda_add_executable(Gemm_tests harness.cpp Gemm_test.cu)
target_include_directories(Gemm_tests PRIVATE gtest/googletest/include)
target_link_libraries(Gemm_tests gtest ${FWGPU_LIB_NAME})
add_test(
  NAME Gemm_tests
  COMMAND Gemm_tests
)

### SemiRing GEMM tests
cuda_add_executable(Srgemm_tests harness.cpp Srgemm_test.cu)
target_include_directories(Srgemm_tests PRIVATE gtest/googletest/include)
target_link_libraries(Srgemm_tests gtest ${FWGPU_LIB_NAME})
add_test(
  NAME Srgemm_tests
  COMMAND Srgemm_tests
)
