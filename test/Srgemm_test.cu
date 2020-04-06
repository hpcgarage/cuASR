#include "gtest/gtest.h"

#include "fwgpu/Matrix.hpp"
#include "fwgpu/cpu_srgemm.hpp"
#include "fwgpu/gpu_srgemm.cuh"
#include "fwgpu/gpu_srgemm.hpp"
#include "fwgpu/internal/utils.cuh"
#include "fwgpu/utils.hpp"

#include <tuple>

TEST(FWGPU_Srgemm, GpuNaiveEqCpuNaive) {
  auto m           = 128;
  auto k           = 32;
  auto n           = 128;
  auto a           = fwgpu::Matrix<float>(m, k, 0xCAFED00D, 1.0, 1000.0);
  auto b           = fwgpu::Matrix<float>(k, n, 0xCAFED00D, 1.0, 1000.0);
  auto c_cpu_naive = fwgpu::Matrix<float>(m, n, 0xCAFED00D, 1.0, 1000.0);
  auto c_gpu_naive = c_cpu_naive;

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_gpu_naive);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  fwgpu::cpu_srgemm_naive(
      m, n, k, a.get_buf(), m, b.get_buf(), k, c_cpu_naive.get_buf(), m);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(m, n, k, d_A, m, d_B, k, d_C, m);
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C, c_gpu_naive.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cpu_naive.size(), c_gpu_naive.size());
  EXPECT_EQ(c_cpu_naive.num_rows(), c_gpu_naive.num_rows());
  EXPECT_EQ(c_cpu_naive.num_cols(), c_gpu_naive.num_cols());
  for (auto i = 0ull; i < c_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu_naive[i], c_gpu_naive[i]);
  }
}

TEST(FWGPU_Srgemm, GpuNaiveEqCutlass) {
  auto N             = 128;
  auto a             = fwgpu::Matrix<float>(N, N, 0xCAFED00D, 1.0, 1000.0);
  auto b             = fwgpu::Matrix<float>(N, N, 0xCAFED00D, 1.0, 1000.0);
  auto c_gpu_naive   = fwgpu::Matrix<float>(N, N, 0.0f);
  auto c_gpu_cutlass = c_gpu_naive;

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(
      a, b, c_gpu_naive, c_gpu_cutlass);

  float *d_A         = std::get<0>(dptrs);
  float *d_B         = std::get<1>(dptrs);
  float *d_C_naive   = std::get<2>(dptrs);
  float *d_C_cutlass = std::get<3>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((N - 1) / 16 + 1, (N - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(N, N, N, d_A, N, d_B, N, d_C_naive, N);
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C_naive, c_gpu_naive.bytesize());

  fwgpu::cutlass_srsgemm_nn(N, N, N, d_A, N, d_B, N, d_C_cutlass, N, true);
  fwgpu::memcpy_d2h(c_gpu_cutlass.get_buf(), d_C_cutlass, c_gpu_cutlass.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_gpu_naive.size(), c_gpu_cutlass.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_gpu_cutlass.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_gpu_cutlass.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_gpu_cutlass[i]);
  }
}

TEST(FWGPU_Srgemm, CpuSubEqCutlassSub_TopLeft_128x128x8) {
  auto m     = 128;
  auto n     = 128;
  auto k     = 8;
  auto a     = fwgpu::Matrix<float>(256, 8, 0, 1.5f, 100.0f);
  auto b     = fwgpu::Matrix<float>(8, 256, 1, 1.5f, 100.0f);
  auto c_cpu = fwgpu::Matrix<float>(256, 256, 1, 1.5f, 100.0f);
  auto c_gpu = c_cpu;

  fwgpu::cpu_srgemm_naive(
      m, n, k,                   //
      a.get_buf(), a.num_rows(), //
      b.get_buf(), b.num_rows(), //
      c_cpu.get_buf(), c_cpu.num_rows());

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_gpu);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  fwgpu::cutlass_srsgemm_nn(
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C, c_gpu.num_rows(), true);
  fwgpu::memcpy_d2h(c_gpu.get_buf(), d_C, c_gpu.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cpu.size(), c_gpu.size());
  EXPECT_EQ(c_cpu.num_rows(), c_gpu.num_rows());
  EXPECT_EQ(c_cpu.num_cols(), c_gpu.num_cols());
  for (auto i = 0ull; i < c_cpu.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu[i], c_gpu[i]);
  }
}

TEST(FWGPU_Srgemm, CpuSubEqGpuNaiveSub_TopLeft_2x2x2) {
  auto m     = 2;
  auto n     = 2;
  auto k     = 2;
  auto a     = fwgpu::Matrix<float>(4, 2, 0, 1.5f, 100.0f);
  auto b     = fwgpu::Matrix<float>(2, 4, 1, 1.5f, 100.0f);
  auto c_cpu = fwgpu::Matrix<float>(4, 4, 1, 1.5f, 100.0f);
  auto c_gpu = c_cpu;

  fwgpu::cpu_srgemm_naive(
      m, n, k,                   //
      a.get_buf(), a.num_rows(), //
      b.get_buf(), b.num_rows(), //
      c_cpu.get_buf(), c_cpu.num_rows());

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_gpu);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C, a.num_rows());
  fwgpu::memcpy_d2h(c_gpu.get_buf(), d_C, c_gpu.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cpu.size(), c_gpu.size());
  EXPECT_EQ(c_cpu.num_rows(), c_gpu.num_rows());
  EXPECT_EQ(c_cpu.num_cols(), c_gpu.num_cols());
  for (auto i = 0ull; i < c_cpu.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu[i], c_gpu[i]);
  }
}

TEST(FWGPU_Srgemm, CpuSubEqGpuNaiveSub_TopLeft_128x128x8) {
  auto m     = 128;
  auto n     = 128;
  auto k     = 8;
  auto a     = fwgpu::Matrix<float>(256, 8, 0, 1.5f, 100.0f);
  auto b     = fwgpu::Matrix<float>(8, 256, 1, 1.5f, 100.0f);
  auto c_cpu = fwgpu::Matrix<float>(256, 256, 2, 1.5f, 100.0f);
  auto c_gpu = c_cpu;

  fwgpu::cpu_srgemm_naive(
      m, n, k,                   //
      a.get_buf(), a.num_rows(), //
      b.get_buf(), b.num_rows(), //
      c_cpu.get_buf(), c_cpu.num_rows());

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_gpu);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C, a.num_rows());
  fwgpu::memcpy_d2h(c_gpu.get_buf(), d_C, c_gpu.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cpu.size(), c_gpu.size());
  EXPECT_EQ(c_cpu.num_rows(), c_gpu.num_rows());
  EXPECT_EQ(c_cpu.num_cols(), c_gpu.num_cols());
  for (auto i = 0ull; i < c_cpu.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu[i], c_gpu[i]);
  }
}

TEST(FWGPU_Srgemm, GpuNaiveSubEqCutlassSub_TopLeft_128x128x8) {
  auto m             = 128;
  auto n             = 128;
  auto k             = 8;
  auto a             = fwgpu::Matrix<float>(256, 8, 0, 1.5f, 100.0f);
  auto b             = fwgpu::Matrix<float>(8, 256, 1, 1.5f, 100.0f);
  auto c_gpu_naive   = fwgpu::Matrix<float>(256, 256, 2, 1.5f, 100.0f);
  auto c_gpu_cutlass = c_gpu_naive;

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(
      a, b, c_gpu_naive, c_gpu_cutlass);
  float *d_A         = std::get<0>(dptrs);
  float *d_B         = std::get<1>(dptrs);
  float *d_C_naive   = std::get<2>(dptrs);
  float *d_C_cutlass = std::get<3>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C_naive, a.num_rows());
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C_naive, c_gpu_naive.bytesize());

  fwgpu::cutlass_srsgemm_nn(
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C_cutlass,
      c_gpu_cutlass.num_rows(), true);
  fwgpu::memcpy_d2h(c_gpu_cutlass.get_buf(), d_C_cutlass, c_gpu_cutlass.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_gpu_naive.size(), c_gpu_cutlass.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_gpu_cutlass.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_gpu_cutlass.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_gpu_cutlass[i]);
  }
}
