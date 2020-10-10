### Matrix tests
add_executable(Matrix_tests ${PROJECT_SOURCE_DIR}/test/harness.cpp Matrix_test.cpp)
target_include_directories(Matrix_tests PRIVATE gtest/googletest/include)
target_link_libraries(Matrix_tests gtest ${cuASR_LIB_NAME})
add_test(
  NAME Matrix_tests
  COMMAND Matrix_tests
)

### SemiRing GEMM tests
add_executable(tropical_gemm_tests ${PROJECT_SOURCE_DIR}/test/harness.cpp Srgemm_test.cu)
target_include_directories(tropical_gemm_tests PRIVATE gtest/googletest/include)
target_link_libraries(tropical_gemm_tests gtest ${cuASR_LIB_NAME})
add_test(
  NAME tropical_gemm_tests
  COMMAND tropical_gemm_tests
)
