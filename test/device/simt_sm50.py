import os

# this file creates the test/unit/gemm/device simt tests and the CMake file to go with it
################################################################################
# parameters
# Edge - for tiles, the edges represent the length of one side
# Ratio - the maximum ratio between 2 edges, limits the skinnyness of tiles
# MaxEdge - maximum length of each edge
# Min/Max - minimum/maximum of the product of edge lengths
################################################################################
THREADS_PER_WARP = 32
WARPS_PER_TB_EDGE = [1, 2, 4, 8, 16]
WARPS_PER_TB_RATIO = 2
WARPS_PER_TB_MAX = 16
# NOTE 1x32 and 2x16 warp tile shapes fail validation for ~10% of cases

WARP_SHAPE_EDGES = [8, 16, 32, 64, 128, 256]
WARP_SHAPE_RATIO = 4
WARP_SHAPE_MAX = 64*64
WARP_SHAPE_MIN = 8*8

THREADBLOCK_MAX_EDGE = 256

UNROLL_MIN = 8

#      char,      type             bits/elem, max tile,    L0 threadblock tiles
precisions = [
    ["d", "double",                    64,   64*64, [[64,  64], [32,  32]]],
    ["s", "float",                     32, 128 *
     128, [[128, 256], [128, 128], [64,  64]]],
    # ["h", "cutlass::half_t",           16, 128*256, [ [256, 128], [ 64, 128], [ 64,  32] ] ],
    # ["i", "int",                       32, 128*128, [[128,  64], [16, 32]]],
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
    ["plus", "multiplies"],      # regular GEMM
    ["minimum", "plus"],         # min-plus (tropical)
    ["maximum", "plus"],         # max-plus
    ["minimum", "maximum"],      # min-max
    ["maximum", "minimum"],      # max-min
    ["minimum", "multiplies"],   # min-multiplies
    ["maximum", "multiplies"],   # max-multiplies
    ["binary_or", "binary_and"]  # or-and
]

testfile_header = """\
/***************************************************************************************************
* Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).
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

test_header_template = """\
////////////////////////////////////////////////////////////////////////////////
// Elements / Thread: {:3.0f} x {:3.0f}
//    Threads / Warp: {:3.0f} x {:3.0f}
//     Warps / Block: {:3.0f} x {:3.0f}
//       Threadblock: {:3.0f} x {:3.0f} x {:3.0f}
"""

test_template = """\
#if defined(CUASR_TEST_LEVEL) and (CUASR_TEST_LEVEL >= {21})
TEST(SM50_device_{0}_{1}_{2}srgemm_{4}{5}_{6}, {10}x{11}x{12}_{13}x{14}x1_{15}x{16}_{17}x{18}_{19}x{20}) {{
  using precision = {3};
  using OpClass   = cutlass::arch::OpClassSimt;
  using SmArch    = cutlass::arch::Sm50;

  using ThreadblockShape = cutlass::gemm::GemmShape<{10}, {11}, {12}>;
  using WarpShape        = cutlass::gemm::GemmShape<{13}, {14}, {12}>;
  using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

  using Config = typename cuasr::gemm::device::DefaultSemiRingConfiguration< //
      precision, precision, precision, precision, OpClass,                   //
      cuasr::{0}<precision>, cuasr::{1}<precision>, SmArch>;

  using AddOp            = Config::AdditionOp;
  using MultOp           = Config::MultiplicationOp;
  using EpilogueOutputOp = Config::EpilogueOutputOp;

  using Srgemm = cuasr::gemm::device::Srgemm<                           //
      AddOp, MultOp,                                                    //
      precision, cutlass::layout::{7}Major,                             //
      precision, cutlass::layout::{8}Major,                             //
      precision, cutlass::layout::{9}Major,                             //
      precision, OpClass, SmArch,                                       //
      ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp,  //
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

  EXPECT_TRUE(cuasr::test::gemm::device::TestAllGemm<Srgemm>());
}}
#endif

"""


def write_test_file_header(testfile):
  testfile.write(testfile_header)


def write_test_to_file(
        testfile,
        add_op,
        mult_op,
        precision_char,
        precision_type,
        transA,
        transB,
        transC,
        threadblock_tile,
        unroll,
        warp_shape,
        thread_tileM,
        thread_tileN,
        warp_threadsM,
        warp_threadsN,
        warps_per_tb,
        test_level):
  print("{:.0f}x{:.0f}x{:.0f}__{:.0f}x{:.0f}_{:.0f}x{:.0f}_{:.0f}x{:.0f}".format(
      threadblock_tile[0], threadblock_tile[1], unroll,
      thread_tileM, thread_tileN,
      warp_threadsM, warp_threadsN,
      warps_per_tb[0], warps_per_tb[1]))

  testfile.write(test_header_template.format(
      thread_tileM, thread_tileN,
      warp_threadsM, warp_threadsN,
      warps_per_tb[0], warps_per_tb[1],
      threadblock_tile[0], threadblock_tile[1], unroll
  ))

  trans_typeA = "Column" if transA == "n" else "Row"
  trans_typeB = "Column" if transB == "n" else "Row"
  trans_typeC = "Column" if transC == "n" else "Row"
  print(precision_type)
  testfile.write(test_template.format(
      add_op,  # 0
      mult_op,  # 1
      precision_char,  # 2
      precision_type,  # 3
      transA,  # 4
      transB,  # 5
      transC,  # 6
      trans_typeA,  # 7
      trans_typeB,  # 8
      trans_typeC,  # 9
      int(threadblock_tile[0]),  # 10
      int(threadblock_tile[1]),  # 11
      int(unroll),  # 12
      int(warp_shape[0]),  # 13
      int(warp_shape[1]),  # 14
      int(thread_tileM),  # 15
      int(thread_tileN),  # 16
      int(warp_threadsM),  # 17
      int(warp_threadsN),  # 18
      int(warps_per_tb[0]),  # 19
      int(warps_per_tb[1]),  # 20
      int(test_level)  # 21
  ))


def main(output_dir: str):
  # warps per threadblock
  warps_per_threadblocks = []
  for warps_per_tb0 in WARPS_PER_TB_EDGE:
    for warps_per_tb1 in WARPS_PER_TB_EDGE:
      if (warps_per_tb0 / warps_per_tb1 <= WARPS_PER_TB_RATIO)      \
              and (warps_per_tb1 / warps_per_tb0 <= WARPS_PER_TB_RATIO) \
              and (warps_per_tb0 * warps_per_tb1 <= WARPS_PER_TB_MAX):
        warps_per_threadblocks.append([warps_per_tb0, warps_per_tb1])
  print("Warps Per Threadblocks", warps_per_threadblocks)

  # warp shapes
  warp_shapes = []
  for warp0 in WARP_SHAPE_EDGES:
    for warp1 in WARP_SHAPE_EDGES:
      if (warp0 / warp1 <= WARP_SHAPE_RATIO)        \
              and (warp1 / warp0 <= WARP_SHAPE_RATIO)   \
              and (warp0 * warp1 <= WARP_SHAPE_MAX)     \
              and (warp0 * warp1 > WARP_SHAPE_MIN):
        warp_shapes.append([warp0, warp1])
  print("Warp Shapes", warp_shapes)

  # create kernels
  # create a file for each precision/transpose
  # each file contains many tile sizes

  # for all semiring add/mul pairs
  num_tests = 0
  testcount_L0 = 0
  testcount_L1 = 0
  testcount_L2 = 0
  for add_op, mult_op in semiring_operators:

    # precisions
    for precision in precisions:
      precision_char = precision[0]
      precision_type = precision[1]
      precision_bits = precision[2]
      tb_max_elements = precision[3]
      tb_tiles_L0 = precision[4]

      # transposes
      for transpose in transposes:
        # get transpose char
        column_major_A = transpose[0]
        column_major_B = transpose[1]
        column_major_C = transpose[2]
        transA = "n" if column_major_A else "t"
        transB = "n" if column_major_B else "t"
        transC = "n" if column_major_C else "t"

        # open file
        testfile_name = "simt_{}_{}_{}srgemm_{}{}_{}_sm50.cu".format(
            add_op, mult_op, precision_char,
            transA, transB, transC)
        print("\n", testfile_name)

        filePath = os.path.join(output_dir, testfile_name)
        with open(filePath, "w") as testfile:
          write_test_file_header(testfile)

          # keeps track of which L0 and L1 test shapes have been seen
          seen_tb_tiles_L0 = {}
          seen_tb_tiles_L1 = {}

          # for each combination of tile sizes
          for warps_per_tb in warps_per_threadblocks:
            for warp_shape in warp_shapes:
              warp_threadsM = 0
              if warp_shape[0] > warp_shape[1]:
                warp_threadsM = 8
              else:
                warp_threadsM = 4
              warp_threadsN = THREADS_PER_WARP / warp_threadsM

              # skip shapes with conflicting rectangularity
              # they are unlikely to be fastest
              blockG = warps_per_tb[0] > warps_per_tb[1]
              blockL = warps_per_tb[0] < warps_per_tb[1]
              warpG = warp_shape[0] > warp_shape[1]
              warpL = warp_shape[0] < warp_shape[1]

              blockG2 = warps_per_tb[0] > warps_per_tb[1]*2
              blockL2 = warps_per_tb[0] * \
                  2 < warps_per_tb[1]
              warpG2 = warp_shape[0] > warp_shape[1]*2
              warpL2 = warp_shape[0]*2 < warp_shape[1]

              if blockG2 and warpL:
                continue
              if blockL2 and warpG:
                continue
              if warpG2 and blockL:
                continue
              if warpL2 and blockG:
                continue

              # check threadblock ratios and max
              threadblock_tile = [warp_shape[0]*warps_per_tb[0],
                                  warp_shape[1]*warps_per_tb[1]]
              if threadblock_tile[0] * threadblock_tile[1] > tb_max_elements:
                continue
              if threadblock_tile[0] > THREADBLOCK_MAX_EDGE:
                continue
              if threadblock_tile[1] > THREADBLOCK_MAX_EDGE:
                continue
              total_threads = THREADS_PER_WARP * \
                  warps_per_tb[0]*warps_per_tb[1]

              # calculate unroll
              # ensure that every iteration at least a full load of A,B are done
              unroll_min0 = total_threads / threadblock_tile[0]
              unroll_min1 = total_threads / threadblock_tile[1]
              unroll = max(UNROLL_MIN, unroll_min0, unroll_min1)

              thread_tileM = warp_shape[0] / warp_threadsM
              thread_tileN = warp_shape[1] / warp_threadsN
              if thread_tileM < 2 or thread_tileN < 2:
                continue
              if thread_tileM * thread_tileN * precision_bits > 8 * 8 * 32:
                continue

              # epilogue currently only supports N < THREADS_PER_WARP
              if threadblock_tile[1] < THREADS_PER_WARP:
                continue

              # limit smem
              shmem_bitsA = threadblock_tile[0] * unroll * 2 * precision_bits
              shmem_bitsB = threadblock_tile[1] * unroll * 2 * precision_bits
              shmem_KiBs = ((shmem_bitsA + shmem_bitsB) / 8) / 1024
              if (shmem_KiBs > 48):
                continue

              test_level = -1
              for tileId in range(0, len(tb_tiles_L0)):
                tbTile = tb_tiles_L0[tileId]
                if tbTile[0] == threadblock_tile[0] and tbTile[1] == threadblock_tile[1]:
                  if tuple(tbTile) not in seen_tb_tiles_L0:
                    test_level = 0
                    testcount_L0 += 1
                    seen_tb_tiles_L0[tuple(tbTile)] = True

              # test level 1
              if test_level < 0:
                if tuple(threadblock_tile) not in seen_tb_tiles_L1:
                  test_level = 1
                  testcount_L1 += 1
                  seen_tb_tiles_L1[tuple(threadblock_tile)] = True

              # test level 2
              if test_level < 0:
                test_level = 2
                testcount_L2 += 1

              # write this tile to file
              write_test_to_file(
                  testfile,
                  add_op,
                  mult_op,
                  precision_char,
                  precision_type,
                  transA,
                  transB,
                  transC,
                  threadblock_tile,
                  unroll,
                  warp_shape,
                  thread_tileM,
                  thread_tileN,
                  warp_threadsM,
                  warp_threadsN,
                  warps_per_tb,
                  test_level)
              num_tests += 1
  print("Total test count per semi-ring = {}".format(num_tests//len(semiring_operators)))


if __name__ == "__main__":
  main(".")
