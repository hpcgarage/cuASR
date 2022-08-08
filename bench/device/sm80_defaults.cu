/***************************************************************************************************
* Copyright (c) 2021, Vijay Thakkar (thakkarv@gatech.edu).
**************************************************************************************************/
//////////////////////////////////////////////////////////////////////
//  THIS BENCHMARK FILE IS GENERATED AUTOMATICALLY : DO NOT MODIFY  //
//////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

#include "harness.h"

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_mult_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::plus_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_plus_mult_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_plus_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_plus_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_plus_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_plus_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_max_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_max<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_max_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_min_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_min<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_min_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_min_mult_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::min_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_min_mult_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_max_mult_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::max_mult<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_max_mult_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_or_and_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;
  using RingOp = cuasr::or_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({ N, N, N });

  // benchmark loop
  for (auto _ : state) {
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_SM80_device_or_and_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);
