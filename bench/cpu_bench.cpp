#include "benchmark/benchmark.h"
#include "include/cpu_gemm.hpp"

static void BM_CPUSimpleGemm(benchmark::State &state) {
  const auto N = state.range(0);
  auto x       = fwgpu::Matrix<float>(N, N, 1.5f);
  auto y       = fwgpu::Matrix<float>(N, N, 1.5f);
  auto z       = fwgpu::Matrix<float>(N, N);
  for (auto _ : state) {
    fwgpu::cpu_gemm_naive(N, N, N, x.get_buf(), y.get_buf(), z.get_buf());
  }
  double flops_per_itr = 2 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_CPUSimpleGemm)->RangeMultiplier(2)->Range(64, 1024);

static void BM_CPUNaiveGemm(benchmark::State &state) {
  const auto N = state.range(0);
  auto x       = fwgpu::Matrix<float>(N, N, 1.5f);
  auto y       = fwgpu::Matrix<float>(N, N, 1.5f);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fwgpu::naive_mm(x, y));
  }
  double flops_per_itr = 2 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_CPUNaiveGemm)->RangeMultiplier(2)->Range(64, 1024);
