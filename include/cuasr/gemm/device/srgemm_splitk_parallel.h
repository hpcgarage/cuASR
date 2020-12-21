/***************************************************************************************************
 * Copyright (c) 2020, Vijay Thakkar (thakkarv@gatech.edu).  All rights reserved.
 **************************************************************************************************/
/*! \file
    \brief Template for GEMM performing a reduction over K partitions in parallel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/arch/arch.h"
#include "cutlass/device_kernel.h"
#include "cutlass/numeric_types.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cuasr/gemm/kernel/srgemm.h"

#include "cuasr/gemm/kernel/default_srgemm_splitk_parallel.h"
#include "cuasr/gemm/device/default_srgemm_configuration.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cuasr/reduction/kernel/reduce_split_k.h"
#include "cuasr/reduction/thread/reduction_operators.h"

////////////////////////////////////////////////////////////////////////////////

namespace cuasr {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

/*!
  Gemm device-level operator performing parallel reduction over the K partition.
*/
template <
  /// Addition operator of the semi-ring
  typename AdditionOp_,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp_,
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Layout type for C and D matrix operands
  typename LayoutC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_ = ElementC_,
  /// Operator class tag
  typename OperatorClass_ = cutlass::arch::OpClassSimt,
  /// Tag indicating architecture to tune for
  typename ArchTag_ = cutlass::arch::Sm50,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::WarpShape,
  /// Instruction-level tile size (concept: GemmShape)
  typename InstructionShape_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::InstructionShape,
  /// Epilogue output operator
  typename EpilogueOutputOp_ = typename DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::EpilogueOutputOp,
  /// Epilogue conversion operator
  typename ConvertScaledOp_ = cutlass::epilogue::thread::Convert<
      ElementAccumulator_, DefaultSemiRingConfiguration<
        ElementA_, ElementB_, ElementC_, ElementAccumulator_,
        OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::EpilogueOutputOp::kCount,
      ElementAccumulator_>,
  /// Reduction operator
  typename ReductionOp_ = cuasr::reduction::thread::SemiringReduce<
      AdditionOp_, ElementAccumulator_,
      typename EpilogueOutputOp_::ElementAccumulator, EpilogueOutputOp_::kCount>,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle_ =
      cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
  /// Number of stages used in the pipelined mainloop
  int Stages = DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::kStages,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA = DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::kAlignmentA,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB = DefaultSemiRingConfiguration<
      ElementA_, ElementB_, ElementC_, ElementAccumulator_,
      OperatorClass_, AdditionOp_, MultiplicationOp_, ArchTag_>::kAlignmentB
>
class SrgemmSplitKParallel {
 public:

  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ConvertScaledOp = ConvertScaledOp_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ReductionOp = ReductionOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;

  /// GEMM kernel
  using SrgemmKernel = typename cuasr::gemm::kernel::DefaultSrgemmSplitKParallel<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    AdditionOp,
    MultiplicationOp,
    ConvertScaledOp,
    ThreadblockSwizzle,
    kStages
  >::SrgemmKernel;

  /// Reduction kernel
  using ReductionKernel = cuasr::reduction::kernel::ReduceSplitK<
    cutlass::MatrixShape<4, 32 * EpilogueOutputOp::kCount>,
    EpilogueOutputOp,
    ReductionOp
  >;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    typename ConvertScaledOp::Params convert;
    typename ReductionOp::Params reduction;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      typename ConvertScaledOp::Params convert_ =
        typename ConvertScaledOp::Params(),
      typename ReductionOp::Params reduction_ =
        typename ReductionOp::Params()
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      convert(convert_),
      reduction(reduction_) { }
  };

private:

  /// Kernel parameters object
  typename SrgemmKernel::Params srgemm_params_;

  /// Reduction kernel parameters object
  typename ReductionKernel::Params reduction_params_;

public:

  /// Constructs the GEMM.
  SrgemmSplitKParallel() { }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {
    // TODO
    return cutlass::Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    return sizeof(ElementAccumulator_) * size_t(args.problem_size.m()) * size_t(args.problem_size.n()) * grid_shape.k();
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace) {
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;
    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    // Define a reference to the workspace - this is an aligned region in device memory.
    if (!workspace) {
      return cutlass::Status::kErrorWorkspaceNull;
    }

    cutlass::TensorRef<ElementAccumulator_, cutlass::layout::RowMajor> ref_workspace(
      static_cast<ElementAccumulator_ *>(workspace),
      args.problem_size.n());

    int64_t partition_stride = int64_t(args.problem_size.m()) * int64_t(args.problem_size.n());

    // Initialize the Params structure
    srgemm_params_ = typename SrgemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      ref_workspace,
      args.convert,
      partition_stride
    };

    reduction_params_ = typename ReductionKernel::Params(
      args.problem_size.mn(),
      grid_shape.k(),
      partition_stride,
      ref_workspace,
      args.ref_D,
      args.ref_C.non_const_ref(),
      args.epilogue
    );

    return cutlass::Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    if (!workspace) {
      return cutlass::Status::kErrorWorkspaceNull;
    }

    srgemm_params_.ref_A.reset(args.ref_A.data());
    srgemm_params_.ref_B.reset(args.ref_B.data());
    srgemm_params_.ref_D.reset(workspace);

    reduction_params_.ref_D.reset(args.ref_D.data());
    reduction_params_.ref_C.reset(args.ref_C.data());

    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(cudaStream_t stream = nullptr) {

    //
    // Launch GEMM kernel
    //

    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(srgemm_params_.grid_tiled_shape);
    dim3 block(SrgemmKernel::kThreadCount, 1, 1);

    cudaError_t result;

    int smem_size = int(sizeof(typename SrgemmKernel::SharedStorage));
    if (smem_size >= (48 << 10)) {

      result = cudaFuncSetAttribute(
        cutlass::Kernel<SrgemmKernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

      if (result != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }

      result = cudaFuncSetAttribute(
        cutlass::Kernel<SrgemmKernel>,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100);

      if (result != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    cutlass::Kernel<SrgemmKernel><<<grid, block, smem_size, stream>>>(srgemm_params_);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    //
    // Launch reduction kernel
    //

    block = ReductionKernel::block_shape();
    grid = ReductionKernel::grid_shape(srgemm_params_.problem_size.mn());

    cutlass::Kernel<ReductionKernel><<<grid, block, 0, stream>>>(reduction_params_);

    result = cudaGetLastError();

    if (result != cudaSuccess) {
      return cutlass::Status::kErrorInternal;
    }

    return result == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for column-major output
template <
  /// Addition operator of the semi-ring
  typename AdditionOp_,
  /// Multiplication operator of the semi-ring
  typename MultiplicationOp_,
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  typename LayoutA_,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  typename LayoutB_,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Element type for internal accumulation
  typename ElementAccumulator_,
  /// Operator class tag
  typename OperatorClass_,
  /// Tag indicating architecture to tune for
  typename ArchTag_,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape_,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape_,
  /// Instruction-level tile size (concept: GemmShape)
  typename InstructionShape_,
  /// Epilogue output operator
  typename EpilogueOutputOp_,
  /// Epilogue output operator
  typename ConvertScaledOp_,
  /// Reduction operator
  typename ReductionOp_,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle_,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB>
class SrgemmSplitKParallel<
    AdditionOp_, MultiplicationOp_, ElementA_, LayoutA_, ElementB_,
    LayoutB_, ElementC_, cutlass::layout::ColumnMajor, // partially specialized on LayoutC
    ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
    WarpShape_, InstructionShape_, EpilogueOutputOp_, ConvertScaledOp_,
    ReductionOp_, ThreadblockSwizzle_, Stages, kAlignmentA, kAlignmentB
    > {
 public:

  using AdditionOp = AdditionOp_;
  using MultiplicationOp = MultiplicationOp_;
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = cutlass::layout::ColumnMajor;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ConvertScaledOp = ConvertScaledOp_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ReductionOp = ReductionOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static int const kStages = Stages;

  using UnderlyingOperator = SrgemmSplitKParallel<
    AdditionOp,
    MultiplicationOp,
    ElementB,
    typename cutlass::layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename cutlass::layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ConvertScaledOp,
    ReductionOp,
    ThreadblockSwizzle,
    Stages,
    kAlignmentA,
    kAlignmentB
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using SrgemmKernel = typename UnderlyingOperator::SrgemmKernel;
  using ReductionKernel = typename UnderlyingOperator::ReductionKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    cutlass::gemm::GemmCoord problem_size;
    cutlass::TensorRef<ElementA const, LayoutA> ref_A;
    cutlass::TensorRef<ElementB const, LayoutB> ref_B;
    cutlass::TensorRef<ElementC const, LayoutC> ref_C;
    cutlass::TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    typename ConvertScaledOp::Params convert;
    typename ReductionOp::Params reduction;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      cutlass::gemm::GemmCoord problem_size_,
      cutlass::TensorRef<ElementA const, LayoutA> ref_A_,
      cutlass::TensorRef<ElementB const, LayoutB> ref_B_,
      cutlass::TensorRef<ElementC const, LayoutC> ref_C_,
      cutlass::TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      typename ConvertScaledOp::Params convert_ =
        typename ConvertScaledOp::Params(),
      typename ReductionOp::Params reduction_ =
        typename ReductionOp::Params()
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      convert(convert_),
      reduction(reduction_) { }
  };

private:

  /// Kernel parameters object
  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  SrgemmSplitKParallel() { }

  /// Helper to construct a transposed equivalent for the underlying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {
    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      {args.ref_A.data(), args.ref_A.stride(0)},
      {args.ref_C.data(), args.ref_C.stride(0)},
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.epilogue,
      args.split_k_slices,
      args.convert,
      args.reduction
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(Arguments const &args, void *workspace) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace);
  }

  /// Lightweight update given a subset of arguments
  cutlass::Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(cudaStream_t stream = nullptr) {

    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    cutlass::Status status = initialize(args, workspace);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cuasr

////////////////////////////////////////////////////////////////////////////////
