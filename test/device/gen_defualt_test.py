import os
import sys
import argparse

################################################################################
# This file generates teset cases for all defualt SRGEMM configurations.
################################################################################

precisions = [
    ["d", "double"],
    ["s", "float"],
]

transposes = [
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


testfile_header = """\
/***************************************************************************************************
* Copyright (c) 2021, Vijay Thakkar (thakkarv@gatech.edu).
**************************************************************************************************/
/////////////////////////////////////////////////////////////////
//  THIS TEST FILE IS GENERATED AUTOMATICALLY : DO NOT MODIFY  //
/////////////////////////////////////////////////////////////////

#include "gtest/gtest.h"

/// from upstream cutlass
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

/// from cuasr lib
#include "cuasr/gemm/device/default_srgemm_configuration.h"
#include "cuasr/gemm/device/srgemm.h"
#include "cuasr/functional.h"

/// from cuasr tools
#include "cuasr/reference/srgemm/host_srgemm.h"

/// from local test dir
#include "testbed.h"

"""


test_template = """\
///////////////////////////////////////////////////////////////////////////////

TEST(SM{sm_arch}_device_{add_op}_{mult_op}_{precision_char}srgemm_{transA}{transB}_{transC}, default_configs) {{
  using precision = {precision_type};
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm{sm_arch};
  using RingOp = cuasr::{add_op}_{mult_op}<precision>;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      RingOp,                                                           //
      precision, cutlass::layout::{trans_typeA}Major,                   //
      precision, cutlass::layout::{trans_typeB}Major,                   //
      precision, cutlass::layout::{trans_typeC}Major,
      precision, OpClass, SmArch>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}}

"""


def write_test_file_header(testfile):
    testfile.write(testfile_header)


def write_test_to_file(
        testfile,
        sm_arch,
        add_op,
        mult_op,
        precision_char,
        precision_type,
        transA,
        transB,
        transC):
    trans_typeA = "Column" if transA == "n" else "Row"
    trans_typeB = "Column" if transB == "n" else "Row"
    trans_typeC = "Column" if transC == "n" else "Row"
    testfile.write(test_template.format(
        sm_arch=sm_arch,
        add_op=add_op,
        mult_op=mult_op,
        precision_char=precision_char,
        precision_type=precision_type,
        transA=transA,
        transB=transB,
        transC=transC,
        trans_typeA=trans_typeA,
        trans_typeB=trans_typeB,
        trans_typeC=trans_typeC
    ))


def main(args):
    num_testes = 0
    testfile_name = "sm{}_defaults.cu".format(args.sm_arch)
    print(testfile_name)
    filePath = os.path.join(args.output_dir, testfile_name)

    # open file and gen all default tests
    with open(filePath, "w") as testfile:
        write_test_file_header(testfile)

        # for all semirings
        for add_op, mult_op in semiring_operators:
            # for all precisions
            for precision in precisions:
                precision_char = precision[0]
                precision_type = precision[1]

                # transposes
                for transpose in transposes:
                    # get transpose char
                    column_major_A = transpose[0]
                    column_major_B = transpose[1]
                    column_major_C = transpose[2]
                    transA = "n" if column_major_A else "t"
                    transB = "n" if column_major_B else "t"
                    transC = "n" if column_major_C else "t"

                    # write to file
                    write_test_to_file(
                        testfile,
                        args.sm_arch,
                        add_op,
                        mult_op,
                        precision_char,
                        precision_type,
                        transA,
                        transB,
                        transC)
                    num_testes += 1
    print("Total test count per semi-ring = {}".format(
        num_testes // len(semiring_operators)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output-dir", type=str, required=False, default=".",
                        help="Path to the output dir.")
    parser.add_argument("-sm", "--sm-arch", type=int, required=False, default=50, choices=[50, 80],
                        help="SM architecture version number,")
    args = parser.parse_args(sys.argv[1:])
    main(args)
