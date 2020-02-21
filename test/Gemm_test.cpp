#include "include/Matrix.hpp"
#include "include/cpu_gemm.hpp"
#include "include/gpu_gemm_entry.cuh"
#include "gtest/gtest.h"

TEST(FWGPU_Gemm, CpuNaiveCorrect) {
  // [8.0   3.0   0.0   1.0]    [5.0 8.0 0.0 6.6]   [ 53    82.5    11.5    60.5]
  // [2.0   5.0   4.0   9.0]  * [4.0 6.0 3.5 0.1] = [ 51    78.5    36.1   118.3]
  // [7.0   6.0   10.   13.]    [3.0 7.0 2.4 9.5]   [102   168.5    58.0   238.0]
  //                            [1.0 0.5 1.0 7.4]
  // >>> import numpy as np
  // >>> a = [[8.0, 3.0, 0.0, 1.0], [2.0, 5.0, 4.0, 9.0], [7.0, 6.0, 10., 13.]]
  // >>> b = [[5.0, 8.0, 0.0, 6.6], [4.0, 6.0, 3.5, 0.1], [3.0, 7.0, 2.4, 9.5], [1.0,
  // 0.5, 1.0, 7.4]]
  // >>> np.matmul(a, b)

  auto a = fwgpu::Matrix<float>(
      3, 4, { 8.0f, 2.0f, 7.0f, 3.0f, 5.0f, 6.0f, 0.0f, 4.0f, 10.0f, 1.0f, 9.0f, 13.0f });

  auto b = fwgpu::Matrix<float>(
      4, 4,
      { 5.0f, 4.0f, 3.0f, 1.0f, 8.0f, 6.0f, 7.0f, 0.5f, 0.0f, 3.5f, 2.4f, 1.0f, 6.6f,
        0.1f, 9.5f, 7.4f });

  auto c = fwgpu::cpu_gemm_naive_entry(a, b);

  EXPECT_EQ(size_t { 12 }, c.size());
  EXPECT_EQ(size_t { 3 }, c.num_rows());
  EXPECT_EQ(size_t { 4 }, c.num_cols());
  EXPECT_FLOAT_EQ(53.0f, c(0, 0));
  EXPECT_FLOAT_EQ(78.5f, c(1, 1));
  EXPECT_FLOAT_EQ(58.0f, c(2, 2));
}

TEST(FWGPU_Gemm, CublasCorrect) {
  // [1.8   7.0   2.8   3.0]    [5.1 2.4]   [56.04,  98.74]
  // [2.0   5.2   4.7   4.1]  * [4.6 6.1] = [56.95, 115.85]
  // [4.0   1.2   5.0   8.0]    [3.2 9.9]   [57.12, 130.42]
  //                            [1.9 8.0]

  auto a = fwgpu::Matrix<float>(
      3, 4, { 1.8f, 2.0f, 4.0f, 7.0f, 5.2f, 1.2f, 2.8f, 4.7f, 5.0f, 3.0f, 4.1f, 8.0f });

  auto b = fwgpu::Matrix<float>(4, 2, { 5.1f, 4.6f, 3.2f, 1.9f, 2.4f, 6.1f, 9.9f, 8.0f });

  auto c = fwgpu::cublas_sgemm_entry(a, b);

  EXPECT_EQ(size_t { 6 }, c.size());
  EXPECT_EQ(size_t { 3 }, c.num_rows());
  EXPECT_EQ(size_t { 2 }, c.num_cols());
  EXPECT_FLOAT_EQ(56.04f, c(0, 0));
  EXPECT_FLOAT_EQ(115.85f, c(1, 1));
  EXPECT_FLOAT_EQ(130.42f, c(2, 1));
}

TEST(FWGPU_Gemm, CpuNaiveEqCublas) {
  // two random matrices
  auto a = fwgpu::Matrix<float>(10, 8, 42, 1.5, 2.5);
  auto b = fwgpu::Matrix<float>(8, 20, 42, 2.5, 5.0);

  auto c_cpu_naive = fwgpu::cpu_gemm_naive_entry(a, b);
  auto c_cublas    = fwgpu::cublas_sgemm_entry(a, b);

  EXPECT_EQ(c_cpu_naive.size(), c_cublas.size());
  EXPECT_EQ(c_cpu_naive.num_rows(), c_cublas.num_rows());
  EXPECT_EQ(c_cpu_naive.num_cols(), c_cublas.num_cols());
  for (auto i = 0ull; i < c_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu_naive[i], c_cublas[i]);
  }
}

TEST(FWGPU_Gemm, GpuNaiveEqCpuNaive) {
  // two random matrices
  auto a = fwgpu::Matrix<float>(10, 8, 42, 1.5, 2.5);
  auto b = fwgpu::Matrix<float>(8, 20, 42, 2.5, 5.0);

  auto c_cpu_naive = fwgpu::cpu_gemm_naive_entry(a, b);
  auto c_gpu_naive = fwgpu::gpu_sgemm_naive_entry(a, b);

  EXPECT_EQ(c_cpu_naive.size(), c_gpu_naive.size());
  EXPECT_EQ(c_cpu_naive.num_rows(), c_gpu_naive.num_rows());
  EXPECT_EQ(c_cpu_naive.num_cols(), c_gpu_naive.num_cols());
  for (auto i = 0ull; i < c_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu_naive[i], c_gpu_naive[i]);
  }
}

TEST(FWGPU_Gemm, GpuNaiveEqCublas) {
  // two random matrices
  auto a = fwgpu::Matrix<float>(10, 8, 42, 1.5, 2.5);
  auto b = fwgpu::Matrix<float>(8, 20, 42, 2.5, 5.0);

  auto c_gpu_naive = fwgpu::gpu_sgemm_naive_entry(a, b);
  auto c_cublas    = fwgpu::cublas_sgemm_entry(a, b);

  EXPECT_EQ(c_gpu_naive.size(), c_cublas.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_cublas.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_cublas.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_cublas[i]);
  }
}
