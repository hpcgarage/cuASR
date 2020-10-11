#include "gtest/gtest.h"

#include "fwgpu/Matrix.hpp"
#include "fwgpu/cpu_srgemm.hpp"
#include "fwgpu/gpu_srgemm.cuh"
#include "fwgpu/gpu_srgemm.hpp"
#include "fwgpu/utils.hpp"
#include "utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <tuple>

TEST(regress_cuASR_Srgemm, CpuNaiveCorrect) {
  auto a = fwgpu::Matrix<float>(
      4, 2,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.19755137, 0.553969979, 0.513400912   //
      });

  auto b = fwgpu::Matrix<float>(
      2, 4,
      {
          0.840187728, 0.911647379, //
          0.394382924, 0.19755137,  //
          0.729605675, 0.335222751, //
          0.798440039, 0.768229604  //
      });

  auto c = fwgpu::Matrix<float>(
      4, 4,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.197551370, 0.553969979, 0.513400912, //
          0.729605675, 0.335222751, 0.477397054, 0.952229738, //
          0.798440039, 0.768229604, 0.628870904, 0.916195095  //
      });

  auto correct = fwgpu::Matrix<float>(
      4, 4,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.197551370, 0.553969979, 0.513400912, //
          0.729605675, 0.335222751, 0.477397054, 0.848623633, //
          0.798440039, 0.768229604, 0.628870904, 0.916195095  //
      });

  fwgpu::cpu_srgemm_naive(
      4, 4, 2,                   //
      a.get_buf(), a.num_rows(), //
      b.get_buf(), b.num_rows(), //
      c.get_buf(), c.num_rows());

  EXPECT_EQ(correct.size(), c.size());
  EXPECT_EQ(correct.num_rows(), c.num_rows());
  EXPECT_EQ(correct.num_cols(), c.num_cols());
  for (auto i = 0ull; i < correct.size(); ++i) {
    EXPECT_FLOAT_EQ(correct[i], c[i]);
  }
}

TEST(regress_cuASR_Srgemm, CutlassCorrect) {
  auto a = fwgpu::Matrix<float>(
      4, 2,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.19755137, 0.553969979, 0.513400912   //
      });

  auto b = fwgpu::Matrix<float>(
      2, 4,
      {
          0.840187728, 0.911647379, //
          0.394382924, 0.19755137,  //
          0.729605675, 0.335222751, //
          0.798440039, 0.768229604  //
      });

  auto c = fwgpu::Matrix<float>(
      4, 4,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.197551370, 0.553969979, 0.513400912, //
          0.729605675, 0.335222751, 0.477397054, 0.952229738, //
          0.798440039, 0.768229604, 0.628870904, 0.916195095  //
      });

  auto correct = fwgpu::Matrix<float>(
      4, 4,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.197551370, 0.553969979, 0.513400912, //
          0.729605675, 0.335222751, 0.477397054, 0.848623633, //
          0.798440039, 0.768229604, 0.628870904, 0.916195095  //
      });

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  fwgpu::cutlass_srsgemm_nn(
      4, 4, 2, d_A, a.num_rows(), d_B, b.num_rows(), d_C, c.num_rows(), true);
  fwgpu::memcpy_d2h(c.get_buf(), d_C, c.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(correct.size(), c.size());
  EXPECT_EQ(correct.num_rows(), c.num_rows());
  EXPECT_EQ(correct.num_cols(), c.num_cols());
  for (auto i = 0ull; i < correct.size(); ++i) {
    EXPECT_FLOAT_EQ(correct[i], c[i]);
  }
}

TEST(regress_cuASR_Srgemm, GpuNaiveEqCpuNaive) {
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

TEST(regress_cuASR_Srgemm, GpuNaiveEqCutlass) {
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

TEST(regress_cuASR_Srgemm, GpuNaiveEqCutlass_TS_Inner) {
  auto m             = 125;
  auto n             = 125;
  auto k             = 1000;
  auto a             = fwgpu::Matrix<float>(m, k, 0xCAFED00D, 1.0, 1000.0);
  auto b             = fwgpu::Matrix<float>(k, n, 0xCAFED00D, 1.0, 1000.0);
  auto c_gpu_naive   = fwgpu::Matrix<float>(m, n, 0.0f);
  auto c_gpu_cutlass = c_gpu_naive;

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(
      a, b, c_gpu_naive, c_gpu_cutlass);

  float *d_A         = std::get<0>(dptrs);
  float *d_B         = std::get<1>(dptrs);
  float *d_C_naive   = std::get<2>(dptrs);
  float *d_C_cutlass = std::get<3>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(m, n, k, d_A, m, d_B, k, d_C_naive, m);
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C_naive, c_gpu_naive.bytesize());

  fwgpu::cutlass_srsgemm_nn(m, n, k, d_A, m, d_B, k, d_C_cutlass, m, true);
  fwgpu::memcpy_d2h(c_gpu_cutlass.get_buf(), d_C_cutlass, c_gpu_cutlass.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_gpu_naive.size(), c_gpu_cutlass.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_gpu_cutlass.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_gpu_cutlass.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_gpu_cutlass[i]);
  }
}

TEST(regress_cuASR_Srgemm, GpuNaiveEqCutlass_TS_Outer) {
  auto m             = 1000;
  auto n             = 1000;
  auto k             = 125;
  auto a             = fwgpu::Matrix<float>(m, k, 0xCAFED00D, 1.0, 1000.0);
  auto b             = fwgpu::Matrix<float>(k, n, 0xCAFED00D, 1.0, 1000.0);
  auto c_gpu_naive   = fwgpu::Matrix<float>(m, n, 0.0f);
  auto c_gpu_cutlass = c_gpu_naive;

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(
      a, b, c_gpu_naive, c_gpu_cutlass);

  float *d_A         = std::get<0>(dptrs);
  float *d_B         = std::get<1>(dptrs);
  float *d_C_naive   = std::get<2>(dptrs);
  float *d_C_cutlass = std::get<3>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(m, n, k, d_A, m, d_B, k, d_C_naive, m);
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C_naive, c_gpu_naive.bytesize());

  fwgpu::cutlass_srsgemm_nn(m, n, k, d_A, m, d_B, k, d_C_cutlass, m, true);
  fwgpu::memcpy_d2h(c_gpu_cutlass.get_buf(), d_C_cutlass, c_gpu_cutlass.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_gpu_naive.size(), c_gpu_cutlass.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_gpu_cutlass.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_gpu_cutlass.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_gpu_cutlass[i]);
  }
}

TEST(regress_cuASR_Srgemm, CpuNaiveSubEqCutlassSub_TopLeft_2x2x2) {
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

TEST(regress_cuASR_Srgemm, CpuNaiveSubEqCutlassSub_TopLeft_8x8x8) {
  auto m     = 8;
  auto n     = 8;
  auto k     = 8;
  auto a     = fwgpu::Matrix<float>(16, 8, 0, 1.5f, 100.0f);
  auto b     = fwgpu::Matrix<float>(8, 16, 1, 1.5f, 100.0f);
  auto c_cpu = fwgpu::Matrix<float>(16, 16, 1, 1.5f, 100.0f);
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

TEST(regress_cuASR_Srgemm, CpuNaiveSubEqCutlassSub_TopLeft_128x128x8) {
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

TEST(regress_cuASR_Srgemm, CpuNaiveSubEqGpuNaiveSub_TopLeft_2x2x2) {
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

TEST(regress_cuASR_Srgemm, CpuNaiveSubEqGpuNaiveSub_TopLeft_128x128x8) {
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

TEST(regress_cuASR_Srgemm, GpuNaiveSubEqCutlassSub_TopLeft_128x128x8) {
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

TEST(regress_cuASR_Srgemm, CpuNaiveSubEqGpuNaiveSub_BottomRight_128x128x8) {
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
      c_cpu.get_buf() + (256 * 128) + 128, c_cpu.num_rows());

  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats(a, b, c_gpu);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_srgemm_naive<<<blocks, threads>>>(
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C + (256 * 128) + 128,
      a.num_rows());
  fwgpu::memcpy_d2h(c_gpu.get_buf(), d_C, c_gpu.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_cpu.size(), c_gpu.size());
  EXPECT_EQ(c_cpu.num_rows(), c_gpu.num_rows());
  EXPECT_EQ(c_cpu.num_cols(), c_gpu.num_cols());
  for (auto i = 0ull; i < c_cpu.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu[i], c_gpu[i]);
  }
}

TEST(regress_cuASR_Srgemm, GpuNaiveSubEqCutlassSub_BottomRight_128x128x8) {
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
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C_naive + (256 * 128) + 128,
      a.num_rows());
  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_C_naive, c_gpu_naive.bytesize());

  fwgpu::cutlass_srsgemm_nn(
      m, n, k, d_A, a.num_rows(), d_B, b.num_rows(), d_C_cutlass + (256 * 128) + 128,
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

TEST(regress_cuASR_Srgemm, CpuNaiveEqCutlass_Small_17x27x17) {
  auto m     = 17;
  auto n     = 27;
  auto k     = 17;
  auto a     = fwgpu::Matrix<float>(17, 17, 0, 1.5f, 100.0f);
  auto b     = fwgpu::Matrix<float>(17, 27, 0, 1.5f, 100.0f);
  auto c_cpu = fwgpu::Matrix<float>(17, 27, 1, 1.5f, 100.0f);
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

TEST(regress_cuASR_Srgemm, CpuNaiveSubEqCutlassSub_Small_7x5x6) {
  auto m     = 7;
  auto n     = 6;
  auto k     = 5;
  auto a     = fwgpu::Matrix<float>(17, 17, 0, 1.5f, 100.0f);
  auto b     = fwgpu::Matrix<float>(17, 27, 0, 1.5f, 100.0f);
  auto c_cpu = fwgpu::Matrix<float>(17, 27, 1, 1.5f, 100.0f);
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
