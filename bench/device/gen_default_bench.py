import os
import sys
import argparse

################################################################################
# This file creates all the possible semiring-gemm kernels for all tnspposes
# using just the defualt SRGEMM configurations for them.
################################################################################

precisions = [
    ["f64", "double"],
    ["f32", "float"],
    ["s32", "int"]
]

tnspposes = [
    [False, False, True],
    [False, False, False],
    [False, True, True],
    [False, True, False],
    [True, False, True],
    [True, False, False],
    [True, True, True],
    [True, True, False],
]

semiring_operators = [
    ["plus", "mult"],  # regular GEMM
    ["min",  "plus"],  # min-plus (tropical)
    ["max",  "plus"],  # max-plus
    ["min",  "max"],   # min-max
    ["max",  "min"],   # max-min
    ["min",  "mult"],  # min-multiplies
    ["max",  "mult"],  # max-multiplies
    ["or",   "and"]    # or-and
]

benchfile_header = """\
/***************************************************************************************************
* Copyright (c) 2022, Vijay Thakkar (thakkarv@gatech.edu).
**************************************************************************************************/
//////////////////////////////////////////////////////////////////////
//  THIS BENCHMARK FILE IS GENERATED AUTOMATICALLY : DO NOT MODIFY  //
//////////////////////////////////////////////////////////////////////

#include "benchmark/benchmark.h"

#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

#include "harness.h"
"""

bench_template = """\

///////////////////////////////////////////////////////////////////////////////

static void BM_SM{sm_arch}_default_{add_op}_{mult_op}_{precision_char}_srgemm_{tnspA}{tnspB}_{tnspC}(benchmark::State &state) {{
  const auto N = static_cast<int>(state.range(0));
  using precision = {precision_type};
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm{sm_arch};
  using RingOp = cuasr::{add_op}_{mult_op}<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::{tnsp_typeA}Major,                   //
      precision, cutlass::layout::{tnsp_typeB}Major,                   //
      precision, cutlass::layout::{tnsp_typeC}Major,
      precision, OpClass, SmArch>;

  // setup bench harness
  cuasr::bench::device::BenchHarness<Srgemm> bench({{ N, N, N }});

  // benchmark loop
  for (auto _ : state) {{
    benchmark::DoNotOptimize(bench.run());
    cudaDeviceSynchronize();
  }}

  double flops_per_itr = 2.0 * N * N * N;
  state.counters["Flop/s"]
      = benchmark::Counter(flops_per_itr, benchmark::Counter::kIsIterationInvariantRate);
}}
BENCHMARK(BM_SM{sm_arch}_default_{add_op}_{mult_op}_{precision_char}_srgemm_{tnspA}{tnspB}_{tnspC})
    ->RangeMultiplier(2)->Range(256, 4096);
"""


def write_benchmark_file_header(benchfile):
    benchfile.write(benchfile_header)


def write_benchmark_to_file(
        benchfile,
        sm_arch,
        add_op,
        mult_op,
        precision_char,
        precision_type,
        tnspA,
        tnspB,
        tnspC):
    tnsp_typeA = "Column" if tnspA == "n" else "Row"
    tnsp_typeB = "Column" if tnspB == "n" else "Row"
    tnsp_typeC = "Column" if tnspC == "n" else "Row"
    benchfile.write(bench_template.format(
        sm_arch=sm_arch,
        add_op=add_op,
        mult_op=mult_op,
        precision_char=precision_char,
        precision_type=precision_type,
        tnspA=tnspA,
        tnspB=tnspB,
        tnspC=tnspC,
        tnsp_typeA=tnsp_typeA,
        tnsp_typeB=tnsp_typeB,
        tnsp_typeC=tnsp_typeC
    ))


def main(args):
    num_benches = 0
    benchfile_name = "sm{}_defaults.cu".format(args.sm_arch)
    print(benchfile_name)
    filePath = os.path.join(args.output_dir, benchfile_name)

    # open file and gen all default tests
    with open(filePath, "w") as benchfile:
        write_benchmark_file_header(benchfile)

        # for all semirings
        for add_op, mult_op in semiring_operators:
            # for all precisions
            for precision in precisions:
                precision_char = precision[0]
                precision_type = precision[1]

                # tnspposes
                for tnsppose in tnspposes:
                    # get tnsppose char
                    column_major_A = tnsppose[0]
                    column_major_B = tnsppose[1]
                    column_major_C = tnsppose[2]
                    tnspA = "n" if column_major_A else "t"
                    tnspB = "n" if column_major_B else "t"
                    tnspC = "n" if column_major_C else "t"

                    # write to file
                    write_benchmark_to_file(
                        benchfile,
                        args.sm_arch,
                        add_op,
                        mult_op,
                        precision_char,
                        precision_type,
                        tnspA,
                        tnspB,
                        tnspC)
                    num_benches += 1
    print("Total bench count per semi-ring = {}".format(
        num_benches // len(semiring_operators)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=False, default=".",
                        help="Path to the output dir.")
    parser.add_argument("-sm", "--sm-arch", type=int, required=False, default=50, choices=[50, 80],
                        help="SM architecture version number,")
    args = parser.parse_args(sys.argv[1:])
    main(args)
