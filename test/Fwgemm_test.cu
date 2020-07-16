#include "gtest/gtest.h"

#include "fwgpu/Matrix.hpp"
#include "fwgpu/cpu_fwgemm.hpp"
#include "fwgpu/gpu_fwgemm.cuh"
#include "fwgpu/internal/utils.cuh"
#include "fwgpu/utils.hpp"


TEST(FWGPU_Fwgemm, GpuNaiveEqCpuNaive) {
  auto m           = 12;
  auto k           = 12;
  auto n           = 12;
  auto a           = fwgpu::Matrix<float>(m, k, 0xCAFED00D, 1.0, 1000.0);
  auto b           = fwgpu::Matrix<float>(k, n, 0xCAFED00D, 1.0, 1000.0);
  auto c_cpu_naive = fwgpu::Matrix<float>(m, n, 0xCAFED00D, 1.0, 1000.0);
  auto p_cpu_naive = fwgpu::Matrix<int>(m, n, 0xCAFED00D, 1, m * n * k);

  auto c_gpu_naive = c_cpu_naive;
  auto p_gpu_naive = p_cpu_naive;

  // alloc device buffers and move inputs to it
  float *d_A;
  fwgpu::malloc_device((void **)&d_A, a.bytesize());
  fwgpu::memcpy_h2d(d_A, a.get_buf(), a.bytesize());

  float *d_B;
  fwgpu::malloc_device((void **)&d_B, b.bytesize());
  fwgpu::memcpy_h2d(d_B, b.get_buf(), b.bytesize());

  float *d_dist;
  fwgpu::malloc_device((void **)&d_dist, c_gpu_naive.bytesize());
  fwgpu::memcpy_h2d(d_dist, c_gpu_naive.get_buf(), c_gpu_naive.bytesize());

  int *d_parent;
  fwgpu::malloc_device((void **)&d_parent, p_gpu_naive.bytesize());
  fwgpu::memcpy_h2d(d_parent, p_gpu_naive.get_buf(), p_gpu_naive.bytesize());
  int parent_offset = 0;

  fwgpu::cpu_fwgemm_naive(
      m,                      //
      n,                      //
      k,                      //
      a.get_buf(),            //
      a.num_cols(),           //
      b.get_buf(),            //
      b.num_cols(),           //
      c_cpu_naive.get_buf(),  //
      c_cpu_naive.num_cols(), //
      p_cpu_naive.get_buf(),  //
      parent_offset           //
  );

  dim3 threads(16, 16);
  dim3 blocks((m - 1) / 16 + 1, (n - 1) / 16 + 1);
  fwgpu::gpu_fwgemm_naive<float><<<blocks, threads>>>(
      m,                      //
      n,                      //
      k,                      //
      d_A,                    //
      a.num_cols(),           //
      d_B,                    //
      b.num_cols(),           //
      d_dist,                 //
      c_gpu_naive.num_cols(), //
      d_parent,               //
      parent_offset,          //
      true,                   //
      nullptr);               //

  fwgpu::memcpy_d2h(c_gpu_naive.get_buf(), d_dist, c_gpu_naive.bytesize());
  fwgpu::memcpy_d2h(p_gpu_naive.get_buf(), d_parent, p_gpu_naive.bytesize());


  EXPECT_EQ(c_cpu_naive.size(), c_gpu_naive.size());
  EXPECT_EQ(c_cpu_naive.num_rows(), c_gpu_naive.num_rows());
  EXPECT_EQ(c_cpu_naive.num_cols(), c_gpu_naive.num_cols());
  for (auto i = 0ull; i < c_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(c_cpu_naive[i], c_gpu_naive[i]);
  }

  EXPECT_EQ(p_cpu_naive.size(), p_gpu_naive.size());
  EXPECT_EQ(p_cpu_naive.num_rows(), p_gpu_naive.num_rows());
  EXPECT_EQ(p_cpu_naive.num_cols(), p_gpu_naive.num_cols());
  for (auto i = 0ull; i < p_cpu_naive.size(); ++i) {
    EXPECT_FLOAT_EQ(p_cpu_naive[i], p_gpu_naive[i]);
  }
}
