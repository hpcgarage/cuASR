#include <tuple>

#include "benchmark/benchmark.h"
#include "fwgpu/Matrix.hpp"
#include "fwgpu/gpu_gemm.cuh"
#include "fwgpu/gpu_srgemm.cuh"
#include "fwgpu/gpu_srgemm.hpp"
#include "fwgpu/internal/utils.cuh"


static void BM_GpuGemmCutlass(benchmark::State &state) {
  const auto N = state.range(0);

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B = fwgpu::Matrix<float>(N, N, 1.5f);
  auto C = fwgpu::Matrix<float>(N, N, 0.0f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats<float>(A, B, C);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  float milliseconds = 0.0;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // loop over benchmark for this size
  for (auto _ : state) {
    cudaEventRecord(start);
    fwgpu::cutlass_sgemm_nn(N, N, N, 1.0, d_A, N, d_B, N, 0.0, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
  }

  double flops_per_itr = 2 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // free device buffers
  fwgpu::internal::dealloc_device_gemm_mats<float>(dptrs);
}
BENCHMARK(BM_GpuGemmCutlass)->RangeMultiplier(2)->Range(64, 4096)->UseManualTime();

static void BM_GpuSrgemmCutlass(benchmark::State &state) {
  const auto N = state.range(0);

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B = fwgpu::Matrix<float>(N, N, 1.5f);
  auto C = fwgpu::Matrix<float>(N, N, 0.0f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats<float>(A, B, C);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  float milliseconds = 0.0;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // loop over benchmark for this size
  for (auto _ : state) {
    cudaEventRecord(start);
    fwgpu::cutlass_srsgemm_nn(N, N, N, d_A, N, d_B, N, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
  }

  double flops_per_itr = 2 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // free device buffers
  fwgpu::internal::dealloc_device_gemm_mats<float>(dptrs);
}
BENCHMARK(BM_GpuSrgemmCutlass)->RangeMultiplier(2)->Range(64, 4096)->UseManualTime();

static void BM_GpuSrgemmCutlass_TS_Outer(benchmark::State &state) {
  const auto N = state.range(0);

  auto m = N;
  auto n = N;
  auto k = N / 8;

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(m, k, 1.5f);
  auto B = fwgpu::Matrix<float>(k, n, 1.5f);
  auto C = fwgpu::Matrix<float>(m, n, 0.0f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats<float>(A, B, C);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  float milliseconds = 0.0;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // loop over benchmark for this size
  for (auto _ : state) {
    cudaEventRecord(start);
    fwgpu::cutlass_srsgemm_nn(m, n, k, d_A, m, d_B, k, d_C, m, true);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
  }

  double flops_per_itr = 2 * N * N * (N / 4);
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // free device buffers
  fwgpu::internal::dealloc_device_gemm_mats<float>(dptrs);
}
BENCHMARK(BM_GpuSrgemmCutlass_TS_Outer)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->UseManualTime();

static void BM_GpuSrgemmCutlass_TS_Inner(benchmark::State &state) {
  const auto N = state.range(0);

  auto m = N / 8;
  auto n = N / 8;
  auto k = N;

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(m, k, 1.5f);
  auto B = fwgpu::Matrix<float>(k, n, 1.5f);
  auto C = fwgpu::Matrix<float>(m, n, 0.0f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats<float>(A, B, C);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  float milliseconds = 0.0;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // loop over benchmark for this size
  for (auto _ : state) {
    cudaEventRecord(start);
    fwgpu::cutlass_srsgemm_nn(m, n, k, d_A, m, d_B, k, d_C, m, true);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
  }

  double flops_per_itr = 2 * (N / 4) * (N / 4) * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // free device buffers
  fwgpu::internal::dealloc_device_gemm_mats<float>(dptrs);
}
BENCHMARK(BM_GpuSrgemmCutlass_TS_Inner)
    ->RangeMultiplier(2)
    ->Range(64, 4096)
    ->UseManualTime();

static void BM_GpuSrgemmNaive(benchmark::State &state) {
  const auto N = state.range(0);

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B = fwgpu::Matrix<float>(N, N, 1.5f);
  auto C = fwgpu::Matrix<float>(N, N, 0.0f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats<float>(A, B, C);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  float milliseconds = 0.0;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // loop over benchmark for this size
  dim3 threads(16, 16);
  dim3 blocks((N - 1) / 16 + 1, (N - 1) / 16 + 1);
  for (auto _ : state) {
    cudaEventRecord(start);
    fwgpu::gpu_srgemm_naive<float><<<blocks, threads>>>(N, N, N, d_A, N, d_B, N, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
  }

  double flops_per_itr = 2 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // free device buffers
  fwgpu::internal::dealloc_device_gemm_mats<float>(dptrs);
}
BENCHMARK(BM_GpuSrgemmNaive)->RangeMultiplier(2)->Range(64, 4096)->UseManualTime();

static void BM_GpuGemmNaive(benchmark::State &state) {
  const auto N = state.range(0);

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B = fwgpu::Matrix<float>(N, N, 1.5f);
  auto C = fwgpu::Matrix<float>(N, N, 0.0f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats<float>(A, B, C);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  float milliseconds = 0.0;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // loop over benchmark for this size
  dim3 threads(16, 16);
  dim3 blocks((N - 1) / 16 + 1, (N - 1) / 16 + 1);
  for (auto _ : state) {
    cudaEventRecord(start);
    fwgpu::gpu_gemm_naive<float><<<blocks, threads>>>(N, N, N, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
  }

  double flops_per_itr = 2 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // free device buffers
  fwgpu::internal::dealloc_device_gemm_mats<float>(dptrs);
}
BENCHMARK(BM_GpuGemmNaive)->RangeMultiplier(2)->Range(64, 4096)->UseManualTime();

static void BM_CublasSgemm(benchmark::State &state) {
  const auto N = state.range(0);

  // init input matrices for this benchmark size N
  auto A = fwgpu::Matrix<float>(N, N, 1.5f);
  auto B = fwgpu::Matrix<float>(N, N, 1.5f);
  auto C = fwgpu::Matrix<float>(N, N, 0.0f);

  // allocate device buffers
  auto dptrs = fwgpu::internal::alloc_and_init_device_gemm_mats<float>(A, B, C);
  float *d_A = std::get<0>(dptrs);
  float *d_B = std::get<1>(dptrs);
  float *d_C = std::get<2>(dptrs);

  float milliseconds = 0.0;
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // loop over benchmark for this size
  for (auto _ : state) {
    cudaEventRecord(start);
    fwgpu::cublas_sgemm(d_A, d_B, d_C, N, N, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    state.SetIterationTime(milliseconds / 1000);
  }

  double flops_per_itr = 2 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // free device buffers
  fwgpu::internal::dealloc_device_gemm_mats<float>(dptrs);
}
BENCHMARK(BM_CublasSgemm)->RangeMultiplier(2)->Range(64, 4096)->UseManualTime();
