#include "gtest/gtest.h"

#include "fwgpu/cpu_srgemm.hpp"
#include "fwgpu/gpu_srgemm.cuh"
#include "fwgpu/gpu_srgemm.hpp"
#include "fwgpu/internal/utils.cuh"
#include "fwgpu/utils.hpp"

#include <limits>

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
  auto N             = 1 << 7;
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

  fwgpu::cutlass_srsgemm_nn(N, N, N, d_A, N, d_B, N, d_C_cutlass, N, nullptr);
  fwgpu::memcpy_d2h(c_gpu_cutlass.get_buf(), d_C_cutlass, c_gpu_cutlass.bytesize());

  fwgpu::internal::dealloc_device_gemm_mats(dptrs);

  EXPECT_EQ(c_gpu_naive.size(), c_gpu_cutlass.size());
  EXPECT_EQ(c_gpu_naive.num_rows(), c_gpu_cutlass.num_rows());
  EXPECT_EQ(c_gpu_naive.num_cols(), c_gpu_cutlass.num_cols());
  for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_gpu_naive[i], c_gpu_cutlass[i]);
  }
}


// TEST(FWGPU_Srgemm, CpuNaiveSubblockCorrect) {
//   auto inf = std::numeric_limits<float>::infinity();
//   auto a   = fwgpu::Matrix<float>(
//       4, 4,
//       {
//           0.0, 5.0, 9.0, inf, //
//           inf, 0.0, 1.0, inf, //
//           inf, inf, 0.0, 2.0, //
//           inf, 3.0, inf, 0.0  //
//       });

//   auto b = a;
//   auto correct = fwgpu::Matrix<float>(
//       4, 4,
//       {
//           0.0, 5.0, 6.0, 8.0, //
//           inf, 0.0, 1.0, 3.0, //
//           inf, 5.0, 0.0, 2.0, //
//           inf, 3.0, 4.0, 0.0  //
//       });


//   EXPECT_EQ(c_gpu_naive.size(), c_gpu_cutlass.size());
//   EXPECT_EQ(c_gpu_naive.num_rows(), c_gpu_cutlass.num_rows());
//   EXPECT_EQ(c_gpu_naive.num_cols(), c_gpu_cutlass.num_cols());
//   for (auto i = 0ull; i < c_gpu_naive.size(); ++i) {
//     EXPECT_FLOAT_EQ(c_gpu_naive[i], c_gpu_cutlass[i]);
//   }
// }
