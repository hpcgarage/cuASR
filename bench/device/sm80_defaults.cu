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

static void BM_SM80_device_plus_multiplies_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_plus_multiplies_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::plus<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_plus_multiplies_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_plus_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_plus_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_plus_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::plus<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_plus_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_maximum_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::maximum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_maximum_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_minimum_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::minimum<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_minimum_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_minimum_multiplies_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::minimum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_minimum_multiplies_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_maximum_multiplies_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::maximum<precision>;
  using MultOp           = cuasr::multiplies<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_maximum_multiplies_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_dsrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = double;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_dsrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_tt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_tt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_tt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_tt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_tn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_tn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_tn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_tn_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_nt_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_nt_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_nt_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_nt_t)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_nn_n(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_nn_n)
    ->RangeMultiplier(2)->Range(256, 4096);

///////////////////////////////////////////////////////////////////////////////

static void BM_SM80_device_binary_or_binary_and_ssrgemm_nn_t(benchmark::State &state) {
  const auto N = static_cast<int>(state.range(0));
  using precision = float;
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm80;

  using AddOp            = cuasr::binary_or<precision>;
  using MultOp           = cuasr::binary_and<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::ColumnMajor,                   //
      precision, cutlass::layout::RowMajor,                   //
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
BENCHMARK(BM_SM80_device_binary_or_binary_and_ssrgemm_nn_t)
    ->RangeMultiplier(2)->Range(256, 4096);
