#include "benchmark/benchmark.h"
#include "include/cpu_gemm.hpp"

static void BM_CPUSimpleGemm(benchmark::State &state) {
  const auto N = state.range(0);
  auto x       = fwgpu::Matrix<float>(N, N, 1.5f);
  auto y       = fwgpu::Matrix<float>(N, N, 1.5f);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fwgpu::cpu_gemm_naive_entry(x, y));
  }
}
BENCHMARK(BM_CPUSimpleGemm)->RangeMultiplier(2)->Range(64, 1024);

static void BM_CPUNaiveGemm(benchmark::State &state) {
  const auto N = state.range(0);
  auto x       = fwgpu::Matrix<float>(N, N, 1.5f);
  auto y       = fwgpu::Matrix<float>(N, N, 1.5f);
  for (auto _ : state) {
    benchmark::DoNotOptimize(fwgpu::naive_mm(x, y));
  }
}
BENCHMARK(BM_CPUNaiveGemm)->RangeMultiplier(2)->Range(64, 1024);
