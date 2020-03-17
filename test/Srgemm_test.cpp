#include "gtest/gtest.h"

#include "GemmEntry.cuh"

TEST(FWGPU_Srgemm, GpuNaiveEqCpuNaive) {
  auto a = fwgpu::Matrix<float>(128, 32, 0xCAFED00D, 1.0, 1000.0);
  auto b = fwgpu::Matrix<float>(32, 128, 0xCAFED00D, 1.0, 1000.0);

  auto c_cpu_naive = fwgpu::testing::cpu_srgemm_naive_entry(a, b);
  auto c_gpu_naive = fwgpu::testing::gpu_srgemm_naive_entry(a, b);

  EXPECT_EQ(c_cpu_naive.size(), c_gpu_naive.size());
  EXPECT_EQ(c_cpu_naive.num_rows(), c_gpu_naive.num_rows());
  EXPECT_EQ(c_cpu_naive.num_cols(), c_gpu_naive.num_cols());
  for (auto i = 0ull; i < c_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu_naive[i], c_gpu_naive[i]);
  }
}

TEST(FWGPU_Srgemm, GpuNaiveEqCutlass) {
  auto a = fwgpu::Matrix<float>(1 << 7, 1 << 7, 0xCAFED00D, 1.0, 1000.0);
  auto b = fwgpu::Matrix<float>(1 << 7, 1 << 7, 0xCAFED00D, 1.0, 1000.0);

  auto c_gpu_naive = fwgpu::testing::gpu_srgemm_naive_entry(a, b);
  auto c_gpu_cutlass = fwgpu::testing::cutlass_srsgemm_entry(a, b);

  EXPECT_EQ(c_gpu_naive.size(), c_gpu_cutlass.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_gpu_cutlass.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_gpu_cutlass.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_gpu_cutlass[i]);
  }
}
