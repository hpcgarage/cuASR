/***************************************************************************************************
 * Copyright (c) 2021, Vijay Thakkar.
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Defines basic properties needed by CTA-level GEMMs assuming
   expectations about data layout of the global memory fragments, data types,
   and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting TensorOp
   instructions.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"

#include "cutlass/gemm/warp/mma_simt_policy.h"
#include "cutlass/gemm/warp/mma_simt.h"


#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"

#include "cuasr/gemm/threadblock/default_srmma_core.h"

////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Ring operation that performs FMA
    typename RingOp_,
    /// Number of stages
    int Stages,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSrmmaCore<Shape_, WarpShape_, InstructionShape_,
                        ElementA_, cutlass::layout::ColumnMajor,
                        ElementB_, cutlass::layout::ColumnMajor,
                        ElementC_, LayoutC_, cutlass::arch::OpClassSimt,
                        RingOp_, Stages,
                        false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using RingOp = RingOp;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpShape::kM,
                                             Shape::kN / WarpShape::kN,
                                             Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //


  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator B
  using SmemThreadMapB = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      SmemThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4; // TODO need to extract these from template data
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static_assert(!((Shape::kK / 32) % LaneN),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
    WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,         /// Data type of A elements
    SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
    ElementB,         /// Data type of B elements
    SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
    ElementC,         /// Element type of C matrix
    LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
    Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    RingOp
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, Shape::kK / 32>,
    WarpCount::kK
  >;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Ring operation that performs FMA
    typename RingOp_,
    /// Number of stages
    int Stages,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSrmmaCore<Shape_, WarpShape_, InstructionShape_,
                      ElementA_, cutlass::layout::ColumnMajor,
                      ElementB_, cutlass::layout::RowMajor,
                      ElementC_, LayoutC_, cutlass::arch::OpClassSimt,
                      RingOp_, Stages, false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using RingOp = RingOp;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpShape::kM,
                                             Shape::kN / WarpShape::kN,
                                             Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //


  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      IteratorThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4; // TODO need to extract these from template data
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
    WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,         /// Data type of A elements
    SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
    ElementB,         /// Data type of B elements
    SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
    ElementC,         /// Element type of C matrix
    LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
    Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    RingOp
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<0, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/// Partial specialization for SIMGEMMsT  using multistage pipeline.
///
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Ring operation that performs FMA
    typename RingOp_,
    /// Number of stages
    int Stages,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSrmmaCore<Shape_, WarpShape_, InstructionShape_,
                      ElementA_, cutlass::layout::RowMajor,
                      ElementB_, cutlass::layout::ColumnMajor,
                      ElementC_, LayoutC_, cutlass::arch::OpClassSimt,
                      RingOp_, Stages,
                      false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using RingOp = RingOp;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpShape::kM,
                                             Shape::kN / WarpShape::kN,
                                             Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //


  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      SmemThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator B
  using SmemThreadMapB = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapB>;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      SmemThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4; // TODO need to extract these from template data
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static_assert(!((Shape::kK / 32) % LaneM) && !((Shape::kK / 32) % LaneN),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
    WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,         /// Data type of A elements
    SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
    ElementB,         /// Data type of B elements
    SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
    ElementC,         /// Element type of C matrix
    LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
    Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    RingOp
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<Shape::kK / 32, 0>,
    cutlass::MatrixShape<0, Shape::kK / 32>,
    WarpCount::kK
  >;
};

/// Partial specialization for SIMT GEMMs using multistage pipeline.
///
///
/// This uses the default warp-level operator given tile sizes
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Ring operation that performs FMA
    typename RingOp_,
    /// Number of stages
    int Stages,
    /// Cache operation of operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Cache operation of operand B
    cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultSrmmaCore<Shape_, WarpShape_, InstructionShape_,
                        ElementA_, cutlass::layout::RowMajor,
                        ElementB_, cutlass::layout::RowMajor,
                        ElementC_, LayoutC_, cutlass::arch::OpClassSimt,
                        RingOp_, Stages,
                        false, CacheOpA, CacheOpB> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementA = ElementA_;
  using LayoutA = cutlass::layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = cutlass::layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using RingOp = RingOp;
  static int const kStages = Stages;
  static cutlass::arch::CacheOperation::Kind const kCacheOpA = cutlass::arch::CacheOperation::Always;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = cutlass::arch::CacheOperation::Always;

  /// Number of warps present
  using WarpCount = cutlass::gemm::GemmShape<Shape::kM / WarpShape::kM,
                                             Shape::kN / WarpShape::kN,
                                             Shape::kK / WarpShape::kK>;

  // Divisility requirements
  static_assert(
      !(Shape::kM % WarpShape::kM) && !(Shape::kN % WarpShape::kN),
      "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size.");

  /// Number of threads per warp
  static int const kWarpSize = cutlass::gemm::warp::WarpSize<cutlass::arch::OpClassSimt>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  // Warp thread arrangement
  static int const kElementsPerAccess = 1;

  //
  // Shared memory layouts
  //

  using SmemLayoutA = cutlass::layout::ColumnMajor;

  // Shared memory layout
  using SmemLayoutB = cutlass::layout::RowMajor;

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  using IteratorThreadMapA = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    kElementsPerAccess
  >;

  /// Transpose the ThreadMap of iterator A
  using SmemThreadMapA = cutlass::transform::TransposePitchLinearThreadMapSimt<IteratorThreadMapA>;

  /// Shared memory iterator to A operand
  using SmemIteratorA = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kM, Shape::kK>, ElementA, SmemLayoutA, 0,
      SmemThreadMapA>;

  /// Policy of iterator B
  using IteratorThreadMapB = cutlass::transform::PitchLinearStripminedThreadMap<
    cutlass::layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    kElementsPerAccess
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = cutlass::transform::threadblock::RegularTileAccessIterator<
      cutlass::MatrixShape<Shape::kK, Shape::kN>, ElementB, SmemLayoutB, 1,
      IteratorThreadMapB>;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level op
  static const int WarpNumThreadsM = 4; // TODO need to extract these from template data
  static const int WarpNumThreadsN = 8;
  static_assert(!(WarpShape::kM % WarpNumThreadsM) && !(WarpShape::kN % WarpNumThreadsN),
      "WarpShape must be divisible by ThreadTile shape.");
  static const int ThreadTileM = WarpShape::kM / WarpNumThreadsM;
  static const int ThreadTileN = WarpShape::kN / WarpNumThreadsN;
  static const int LaneLayout = ThreadTileM > 4 && ThreadTileN > 4 ? 2 : 1;
  static const int numElementsA = 128 / cutlass::sizeof_bits<ElementA>::value;
  static const int numElementsB = 128 / cutlass::sizeof_bits<ElementB>::value;
  static const int LaneM = cutlass::const_min(numElementsA, ThreadTileM);
  static const int LaneN = cutlass::const_min(numElementsB, ThreadTileN);

  static_assert(!((Shape::kK / 32) % LaneM),
                "Padding must be divisible by Lane");

  // these should have max of thread tile also
  using LaneMmaShape = cutlass::gemm::GemmShape<
      LaneM,
      LaneN,
      1>;

  using Policy = cutlass::gemm::warp::MmaSimtPolicy<
      cutlass::MatrixShape<WarpNumThreadsM, WarpNumThreadsN>,   // WarpShape
      cutlass::layout::RowMajorInterleaved<LaneLayout>,         // LaneLayout
      LaneMmaShape
  >;

  using MmaWarpSimt = cuasr::gemm::warp::SrmmaSimt<
    WarpShape,        /// Size of the Gemm problem - concept: gemm::GemmShape<> 128, 128, 8
    ElementA,         /// Data type of A elements
    SmemLayoutA,      /// Layout of A matrix (concept: MatrixLayout)
    ElementB,         /// Data type of B elements
    SmemLayoutB,      /// Layout of B matrix (concept: MatrixLayout)
    ElementC,         /// Element type of C matrix
    LayoutC,          /// Layout of C matrix (concept: MatrixLayout)
    Policy,           /// Policy describing warp-level MmaSimtOp (concept: MmaSimtOp policy)
    RingOp
  >;

  /// Policy used to define MmaPipelined
  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
    MmaWarpSimt,
    cutlass::MatrixShape<Shape::kK / 32, 0>,
    cutlass::MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cuasr
