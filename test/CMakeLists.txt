add_subdirectory(gtest)

include_directories(gtest/googletest/include)

set(test_src
  harness.cpp
  GemmEntry.cu
  Matrix_test.cpp
  Gemm_test.cpp
)

add_executable(test_fwgpu ${test_src})

target_link_libraries(test_fwgpu
  gtest
  ${FWGPU_LIB_NAME}
)

add_test(
  NAME tests
  COMMAND test_fwgpu
)
