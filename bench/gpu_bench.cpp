#include "benchmark/benchmark.h"
// #include "include/cublas_gemm.cuh"
#include "include/gpu_gemm_entry.cuh"

static void BM_GpuGemmNaive(benchmark::State &state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const auto N = state.range(0);
    auto x       = fwgpu::Matrix<float>(N, N, 1.5f);
    auto y       = fwgpu::Matrix<float>(N, N, 1.5f);
    state.ResumeTiming();
    auto z = fwgpu::gpu_sgemm_naive_entry(x, y);
  }
}
BENCHMARK(BM_GpuGemmNaive)->RangeMultiplier(2)->Range(64, 1024);

static void BM_GpuGemmShReg(benchmark::State &state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const auto N = state.range(0);
    auto x       = fwgpu::Matrix<float>(N, N, 1.5f);
    auto y       = fwgpu::Matrix<float>(N, N, 1.5f);
    state.ResumeTiming();
    auto z = fwgpu::gpu_sgemm_sh_reg_entry(x, y);
  }
}
BENCHMARK(BM_GpuGemmShReg)->RangeMultiplier(2)->Range(64, 1024);

static void BM_CublasSgemm(benchmark::State &state) {
  while (state.KeepRunning()) {
    state.PauseTiming();
    const auto N = state.range(0);
    auto x       = fwgpu::Matrix<float>(N, N, 1.5f);
    auto y       = fwgpu::Matrix<float>(N, N, 1.5f);
    state.ResumeTiming();
    auto z = fwgpu::cublas_sgemm_entry(x, y);
  }
}
BENCHMARK(BM_CublasSgemm)->RangeMultiplier(2)->Range(64, 1024);
