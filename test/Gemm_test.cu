#include "fwgpu/Matrix.hpp"
#include "fwgpu/cpu_gemm.hpp"
#include "fwgpu/gpu_gemm.cuh"
#include "fwgpu/internal/utils.cuh"
#include "fwgpu/utils.hpp"

#include "gtest/gtest.h"

TEST(cuASR_Gemm, CpuNaiveCorrect) {
  // [8.0   3.0   0.0   1.0]    [5.0 8.0 0.0 6.6]   [ 53    82.5    11.5    60.5]
  // [2.0   5.0   4.0   9.0]  * [4.0 6.0 3.5 0.1] = [ 51    78.5    36.1   118.3]
  // [7.0   6.0   10.   13.]    [3.0 7.0 2.4 9.5]   [102   168.5    58.0   238.0]
  //                            [1.0 0.5 1.0 7.4]
  // >>> import numpy as np
  // >>> a = [[8.0, 3.0, 0.0, 1.0], [2.0, 5.0, 4.0, 9.0], [7.0, 6.0, 10., 13.]]
  // >>> b = [[5.0, 8.0, 0.0, 6.6], [4.0, 6.0, 3.5, 0.1], [3.0, 7.0, 2.4, 9.5], [1.0,
  // 0.5, 1.0, 7.4]]
  // >>> np.matmul(a, b)

  auto m = 3;
  auto k = 4;
  auto n = 4;

  auto a = fwgpu::Matrix<float>(
      m, k,
      { 8.0f, 2.0f, 7.0f, 3.0f, //
        5.0f, 6.0f, 0.0f, 4.0f, //
        10.0f, 1.0f, 9.0f, 13.0f });

  auto b = fwgpu::Matrix<float>(
      k, n,
      { 5.0f, 4.0f, 3.0f, 1.0f, //
        8.0f, 6.0f, 7.0f, 0.5f, //
        0.0f, 3.5f, 2.4f, 1.0f, //
        6.6f, 0.1f, 9.5f, 7.4f });

  auto c = fwgpu::Matrix<float>(m, n);
  fwgpu::cpu_gemm_naive(m, n, k, a.get_buf(), b.get_buf(), c.get_buf());

  EXPECT_EQ(size_t { 12 }, c.size());
  EXPECT_EQ(size_t { 3 }, c.num_rows());
  EXPECT_EQ(size_t { 4 }, c.num_cols());
  EXPECT_FLOAT_EQ(53.0f, c(0, 0));
  EXPECT_FLOAT_EQ(78.5f, c(1, 1));
  EXPECT_FLOAT_EQ(58.0f, c(2, 2));
}

TEST(cuASR_Gemm, CublasCorrect) {
  // [1.8   7.0   2.8   3.0]    [5.1 2.4]   [56.04,  98.74]
  // [2.0   5.2   4.7   4.1]  * [4.6 6.1] = [56.95, 115.85]
  // [4.0   1.2   5.0   8.0]    [3.2 9.9]   [57.12, 130.42]
  //                            [1.9 8.0]

  auto a = fwgpu::Matrix<float>(
      3, 4,
      { 1.8f, 2.0f, 4.0f, 7.0f, //
        5.2f, 1.2f, 2.8f, 4.7f, //
        5.0f, 3.0f, 4.1f, 8.0f });

  auto b = fwgpu::Matrix<float>(
      4, 2,
      { 5.1f, 4.6f, 3.2f, 1.9f, //
        2.4f, 6.1f, 9.9f, 8.0f });

  auto c = fwgpu::Matrix<float>(3, 2);

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  fwgpu::cublas_sgemm(d_A, d_B, d_C, 3, 4, 2);
  fwgpu::memcpy_d2h(c.get_buf(), d_C, c.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(size_t { 6 }, c.size());
  EXPECT_EQ(size_t { 3 }, c.num_rows());
  EXPECT_EQ(size_t { 2 }, c.num_cols());
  EXPECT_FLOAT_EQ(56.04f, c(0, 0));
  EXPECT_FLOAT_EQ(115.85f, c(1, 1));
  EXPECT_FLOAT_EQ(130.42f, c(2, 1));
}

TEST(cuASR_Gemm, CutlassCorrect) {
  auto a = fwgpu::Matrix<float>(
      3, 4,
      { 1.8f, 2.0f, 4.0f, 7.0f, //
        5.2f, 1.2f, 2.8f, 4.7f, //
        5.0f, 3.0f, 4.1f, 8.0f });

  auto b = fwgpu::Matrix<float>(
      4, 2,
      { 5.1f, 4.6f, 3.2f, 1.9f, //
        2.4f, 6.1f, 9.9f, 8.0f });

  auto c = fwgpu::Matrix<float>(3, 2);

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  auto m = 3;
  auto n = 2;
  auto k = 4;
  fwgpu::cutlass_sgemm_nn(m, n, k, 1, d_A, m, d_B, k, 0, d_C, m);
  fwgpu::memcpy_d2h(c.get_buf(), d_C, c.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(size_t { 6 }, c.size());
  EXPECT_EQ(size_t { 3 }, c.num_rows());
  EXPECT_EQ(size_t { 2 }, c.num_cols());
  EXPECT_FLOAT_EQ(56.04f, c(0, 0));
  EXPECT_FLOAT_EQ(115.85f, c(1, 1));
  EXPECT_FLOAT_EQ(130.42f, c(2, 1));
}

TEST(cuASR_Gemm, CpuNaiveEqCublas) {
  // two random matrices
  auto m           = 10;
  auto k           = 8;
  auto n           = 20;
  auto a           = fwgpu::Matrix<float>(m, k, 42, 1.5, 2.5);
  auto b           = fwgpu::Matrix<float>(k, n, 42, 2.5, 5.0);
  auto c_cpu_naive = fwgpu::Matrix<float>(m, n, 42, 2.5, 5.0);
  auto c_cublas    = c_cpu_naive;

  fwgpu::cpu_gemm_naive(m, n, k, a.get_buf(), b.get_buf(), c_cpu_naive.get_buf());

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_cublas);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  fwgpu::cublas_sgemm(d_A, d_B, d_C, m, k, n);
  fwgpu::memcpy_d2h(c_cublas.get_buf(), d_C, c_cublas.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cpu_naive.size(), c_cublas.size());
  EXPECT_EQ(c_cpu_naive.num_rows(), c_cublas.num_rows());
  EXPECT_EQ(c_cpu_naive.num_cols(), c_cublas.num_cols());
  for (auto i = 0ull; i < c_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu_naive[i], c_cublas[i]);
  }
}

TEST(cuASR_Gemm, GpuNaiveEqCpuNaive) {
  // two random matrices
  auto m           = 10;
  auto k           = 8;
  auto n           = 20;
  auto a           = fwgpu::Matrix<float>(m, k, 42, 1.5, 2.5);
  auto b           = fwgpu::Matrix<float>(k, n, 42, 2.5, 5.0);
  auto c_cpu_naive = fwgpu::Matrix<float>(m, n, 42, 2.5, 5.0);
  auto c_gpu_naive = c_cpu_naive;

  fwgpu::cpu_gemm_naive(m, n, k, a.get_buf(), b.get_buf(), c_cpu_naive.get_buf());

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_gpu_naive);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_gemm_naive<float><<<blocks, threads>>>(m, n, k, d_A, d_B, d_C);
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C, c_gpu_naive.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cpu_naive.size(), c_gpu_naive.size());
  EXPECT_EQ(c_cpu_naive.num_rows(), c_gpu_naive.num_rows());
  EXPECT_EQ(c_cpu_naive.num_cols(), c_gpu_naive.num_cols());
  for (auto i = 0ull; i < c_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu_naive[i], c_gpu_naive[i]);
  }
}

TEST(cuASR_Gemm, GpuNaiveEqCublas) {
  // two random matrices
  auto m           = 10;
  auto k           = 8;
  auto n           = 20;
  auto a           = fwgpu::Matrix<float>(m, k, 42, 1.5, 2.5);
  auto b           = fwgpu::Matrix<float>(k, n, 42, 2.5, 5.0);
  auto c_gpu_naive = fwgpu::Matrix<float>(m, n, 0.0f);
  auto c_cublas    = c_gpu_naive;

  auto dptrs
      = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_gpu_naive, c_cublas);
  float *d_A        = std::get<0>(dptrs);
  float *d_B        = std::get<1>(dptrs);
  float *d_C_naive  = std::get<2>(dptrs);
  float *d_C_cublas = std::get<3>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_gemm_naive<<<blocks, threads>>>(m, n, k, d_A, d_B, d_C_naive);
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C_naive, c_gpu_naive.bytesize());

  fwgpu::cublas_sgemm(d_A, d_B, d_C_cublas, m, k, n);
  fwgpu::memcpy_d2h(c_cublas.get_buf(), d_C_cublas, c_cublas.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_gpu_naive.size(), c_cublas.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_cublas.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_cublas.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_cublas[i]);
  }
}

TEST(cuASR_Gemm, CutlassEqCublas) {
  // two random matrices
  auto m         = 10;
  auto k         = 8;
  auto n         = 20;
  auto a         = fwgpu::Matrix<float>(m, k, 42, 1.5, 2.5);
  auto b         = fwgpu::Matrix<float>(k, n, 42, 2.5, 5.0);
  auto c_cutlass = fwgpu::Matrix<float>(m, n, 42, 2.5, 5.0);
  auto c_cublas  = c_cutlass;

  auto dptrs
      = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_cutlass, c_cublas);
  float *d_A         = std::get<0>(dptrs);
  float *d_B         = std::get<1>(dptrs);
  float *d_C_cutlass = std::get<2>(dptrs);
  float *d_C_cublas  = std::get<3>(dptrs);

  fwgpu::cutlass_sgemm_nn(m, n, k, 1, d_A, m, d_B, k, 0, d_C_cutlass, m);
  fwgpu::memcpy_d2h(c_cutlass.get_buf(), d_C_cutlass, c_cutlass.bytesize());

  fwgpu::cublas_sgemm(d_A, d_B, d_C_cublas, m, k, n);
  fwgpu::memcpy_d2h(c_cublas.get_buf(), d_C_cublas, c_cublas.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cutlass.size(), c_cublas.size());
  EXPECT_EQ(c_cutlass.num_rows(), c_cublas.num_rows());
  EXPECT_EQ(c_cutlass.num_cols(), c_cublas.num_cols());
  for (auto i = 0ull; i < c_cutlass.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cutlass[i], c_cublas[i]);
  }
}
