#include <chrono>
#include <tuple>

#include "benchmark/benchmark.h"
#include "include/gpu_gemm.cuh"
#include "include/gpu_gemm_entry.cuh"
#include "include/internal/utils.cuh"

static void BM_GpuGemmNaive(benchmark::State &state) {
  using namespace std::chrono;
  const auto N = state.range(0);

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B = fwgpu::Matrix<float>(N, N, 1.5f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_gemm_mats_on_gpu<float>(A, B);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  // loop over benchmark for this size
  dim3 threads(16, 16);
  dim3 blocks((N - 1) / 16 + 1, (N - 1) / 16 + 1);
  for (auto _ : state) {
    auto start   = high_resolution_clock::now();

    fwgpu::gpu_gemm_naive<float><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();
    auto d   = duration_cast<duration<double>>(end - start);
    state.SetIterationTime(d.count());
  }

  // free device buffers
  fwgpu::internal::dealloc_gemm_mats_on_gpu<float>(dptrs);
}
BENCHMARK(BM_GpuGemmNaive)
    ->RangeMultiplier(2)
    ->Range(64, 1024)
    ->UseManualTime();

static void BM_GpuGemmShReg(benchmark::State &state) {
  using namespace std::chrono;
  const auto N = state.range(0);

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B = fwgpu::Matrix<float>(N, N, 1.5f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_gemm_mats_on_gpu(A, B);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  // loop over benchmark for this size
  dim3 threads(16, 16);
  dim3 blocks((N - 1) / 16 + 1, (N - 1) / 16 + 1);
  for (auto _ : state) {
    auto start   = high_resolution_clock::now();

    fwgpu::gpu_gemm_sh_reg<float, 16, 16, 64>
        <<<blocks, threads>>>(N, N, N, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();
    auto d   = duration_cast<duration<double>>(end - start);
    state.SetIterationTime(d.count());
  }

  // free device buffers
  fwgpu::internal::dealloc_gemm_mats_on_gpu(dptrs);
}
BENCHMARK(BM_GpuGemmShReg)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->UseManualTime();

static void BM_CublasSgemm(benchmark::State &state) {
  using namespace std::chrono;
  const auto N = state.range(0);

  // allocate device buffers for this benchmark size N
  auto A     = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B     = fwgpu::Matrix<float>(N, N, 1.5f);
  auto dptrs = fwgpu::internal::alloc_gemm_mats_on_gpu(A, B);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  // loop over benchmark for this size
  for (auto _ : state) {
    auto start = high_resolution_clock::now();

    fwgpu::cublas_sgemm(d_A, d_B, d_C, N, N, N);
    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();
    auto d   = duration_cast<duration<double>>(end - start);
    state.SetIterationTime(d.count());
  }

  // free device buffers
  fwgpu::internal::dealloc_gemm_mats_on_gpu(dptrs);
}
BENCHMARK(BM_CublasSgemm)->RangeMultiplier(2)->Range(64, 4096)->UseManualTime();
