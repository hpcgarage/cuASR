/***************************************************************************************************

 **************************************************************************************************/
/*! \file
    \brief Template for 3D SRGEMM performing a reduction over K partitions in parallel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Srmma_,                ///! Threadblock-scoped matrix multiply-accumulate
  typename AdditionOp_,           ///! Addition operator of the semi-ring
  typename MultiplicationOp_,     ///! Multiplication operator of the semi-ring
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_    ///! Threadblock swizzling function
>
struct SrgemmSplitKParallel {

  using Srmma = Srmma_;
  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Srmma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;
  static int const kAlignmentK = Srmma::Operator::Shape::kK;

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    typename Srmma::IteratorA::Params params_A;
    typename Srmma::IteratorA::TensorRef ref_A;
    typename Srmma::IteratorB::Params params_B;
    typename Srmma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    int64_t splitk_slice_stride;
    int gemm_k_size;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Srmma::IteratorA::TensorRef ref_A,
      typename Srmma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op,
      int64_t splitk_slice_stride
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      params_A(ref_A.layout()),
      ref_A(ref_A),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      output_op(output_op),
      splitk_slice_stride(splitk_slice_stride) {

      int full_gemm_k_iterations = problem_size.k() / Srmma::Shape::kK;
      int gemm_k_iterations = full_gemm_k_iterations / grid_tiled_shape.k();

      gemm_k_size = gemm_k_iterations * Srmma::Shape::kK;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Srmma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  SrgemmSplitKParallel() { }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    constexpr typename OutputOp::ElementCompute kAdditiveIdentity = AdditionOp::Identity;

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Srmma::Shape::kM,
      threadblock_tile_offset.k() * params.gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_B{
      threadblock_tile_offset.k() * params.gemm_k_size,
      threadblock_tile_offset.n() * Srmma::Shape::kN
    };

    // Problem size is a function of threadblock index in the K dimension
    int problem_size_k;
    if (threadblock_tile_offset.k() + 1 == params.grid_tiled_shape.k()) {
      problem_size_k = params.problem_size.k();
    }
    else {
      problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
    }

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A.column() + Srmma::Shape::kK - 1) / Srmma::Shape::kK;

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Srmma::IteratorA iterator_A(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A);

    typename Srmma::IteratorB iterator_B(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B);

    int warp_idx = threadIdx.x / 32;
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Srmma srmma_thrblock_op(shared_storage.main_loop, thread_idx, warp_idx, lane_idx, kAdditiveIdentity);

    // need to clear accumulators to additive identity for SemiRing Gemm
    typename Srmma::FragmentC accumulators;
    accumulators.fill(kAdditiveIdentity);

    srmma_thrblock_op(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.grid_tiled_shape);

    // assume identity swizzle
    cutlass::MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Srmma::Shape::kM,
      threadblock_tile_offset.n() * Srmma::Shape::kN
    );

    // Tile iterator writing to output tile
    typename Epilogue::OutputTileIterator iterator_D(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset
    );

    iterator_D.add_pointer_offset(params.splitk_slice_stride * threadblock_tile_offset.k());

    // Execute the epilogue
    Epilogue epilogue(
      shared_storage.epilogue,
      thread_idx,
      warp_idx,
      lane_idx);

    // Run efficient epilogue
    epilogue(output_op, iterator_D, accumulators, iterator_D);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cuasr
