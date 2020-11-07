#include <fstream>
#include <iostream>
#include <sstream>

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/tensor_view_io.h"

#include "cuasr/reference/srgemm/host_srgemm.h"

#include <cxxabi.h>

namespace cuasr {
namespace test {
namespace gemm {
namespace device {

namespace {
inline char const *to_string(cutlass::Status status) {
  switch (status) {
    case cutlass::Status::kSuccess:
      return "kSuccess";
    case cutlass::Status::kErrorMisalignedOperand:
      return "kErrorMisalignedOperand";
    case cutlass::Status::kErrorInvalidLayout:
      return "kErrorInvalidLayout";
    case cutlass::Status::kErrorInvalidProblem:
      return "kErrorInvalidProblem";
    case cutlass::Status::kErrorNotSupported:
      return "kErrorNotSupported";
    case cutlass::Status::kErrorWorkspaceNull:
      return "kErrorWorkspaceNull";
    case cutlass::Status::kErrorInternal:
      return "kErrorInternal";
    case cutlass::Status::kInvalid:
      return "kInvalid";
    default:
      break;
  }
  return "invalid";
}
}

// Given a SIMT SRGEMM, runs test cases against it
template <typename Srgemm>
struct Testbed {
  using ElementAccumulator = typename Srgemm::ElementAccumulator;
  using ElementCompute =
      typename Srgemm::SrgemmKernel::Epilogue::OutputOp::ElementCompute;

  /// Initialization
  cutlass::Distribution::Kind init_A;
  cutlass::Distribution::Kind init_B;
  cutlass::Distribution::Kind init_C;
  uint64_t seed;

  cutlass::HostTensor<typename Srgemm::ElementA, typename Srgemm::LayoutA> tensor_A;
  cutlass::HostTensor<typename Srgemm::ElementB, typename Srgemm::LayoutB> tensor_B;
  cutlass::HostTensor<typename Srgemm::ElementC, typename Srgemm::LayoutC> tensor_C;
  cutlass::HostTensor<typename Srgemm::ElementC, typename Srgemm::LayoutC> tensor_D;
  cutlass::HostTensor<typename Srgemm::ElementC, typename Srgemm::LayoutC> reference_D;

  // Methods
  Testbed(
      cutlass::Distribution::Kind init_A_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_B_ = cutlass::Distribution::Uniform,
      cutlass::Distribution::Kind init_C_ = cutlass::Distribution::Uniform,
      uint64_t seed_                      = 2080)
      : init_A(init_A_)
      , init_B(init_B_)
      , init_C(init_C_)
      , seed(seed_) { }

  /// Helper to initialize a tensor view
  template <typename Element, typename Layout>
  bool initialize_tensor(
      cutlass::TensorView<Element, Layout> view,
      cutlass::Distribution::Kind dist_kind,
      uint64_t seed) {
    if (dist_kind == cutlass::Distribution::Uniform) {
      double scope_max, scope_min;
      int bits_input  = cutlass::sizeof_bits<Element>::value;
      int bits_output = cutlass::sizeof_bits<typename Srgemm::ElementC>::value;

      if (bits_input == 1) {
        scope_max = 2;
        scope_min = 0;
      }
      else if (bits_input <= 8) {
        scope_max = 2;
        scope_min = -2;
      }
      else if (bits_output == 16) {
        scope_max = 5;
        scope_min = -5;
      }
      else {
        scope_max = 8;
        scope_min = -8;
      }

      cutlass::reference::host::TensorFillRandomUniform(
          view, seed, scope_max, scope_min, 0);
    }
    else if (dist_kind == cutlass::Distribution::Identity) {
      cutlass::reference::host::TensorFillIdentity(view);
    }
    else if (dist_kind == cutlass::Distribution::Gaussian) {
      cutlass::reference::host::TensorFillRandomGaussian(view, seed, 0, 0.5);
    }
    else if (dist_kind == cutlass::Distribution::Sequential) {
      cutlass::reference::host::BlockFillSequential(view.data(), view.capacity());
    }
    else {
      EXPECT_TRUE(false) << "Not implemented";
      return false;
    }

    return true;
  }

  /// Initializes data structures
  void initialize(cutlass::gemm::GemmCoord problem_size) {
    // Allocate the GEMM workspace
    tensor_A.resize(problem_size.mk());
    tensor_B.resize(problem_size.kn());
    tensor_C.resize(problem_size.mn());
    tensor_D.resize(problem_size.mn());
    reference_D.resize(problem_size.mn(), false);

    EXPECT_TRUE(initialize_tensor(tensor_A.host_view(), init_A, seed + 2019));
    EXPECT_TRUE(initialize_tensor(tensor_B.host_view(), init_B, seed + 2018));
    EXPECT_TRUE(initialize_tensor(tensor_C.host_view(), init_C, seed + 2017));

    // It is possible to randomly initialize to all zeros, so override this with non-zeros
    // in the upper left corner of each operand.
    tensor_A.host_view().at({ 0, 0 }) = typename Srgemm::ElementA(1);
    tensor_B.host_view().at({ 0, 0 }) = typename Srgemm::ElementB(1);
    tensor_C.host_view().at({ 0, 0 }) = typename Srgemm::ElementC(1);

    cutlass::reference::host::TensorCopy(reference_D.host_view(), tensor_C.host_view());

    tensor_A.sync_device();
    tensor_B.sync_device();
    tensor_C.sync_device();
    tensor_D.sync_device();
  }

  /// Compares computed reference with device reference and outputs to a file if incorrect
  bool compare_reference(
      cutlass::gemm::GemmCoord problem_size, ElementCompute alpha, ElementCompute beta) {
    tensor_D.sync_host();

    bool passed = cutlass::reference::host::TensorEquals(
        reference_D.host_view(), tensor_D.host_view());
    EXPECT_TRUE(passed);

    if (!passed) {
      // record failed test cases to a file for debug records
      std::string add_op_name_full(abi::__cxa_demangle(
          typeid(typename Srgemm::AdditionOp).name(), //
          nullptr, nullptr, nullptr));

      std::string mult_op_name_full(abi::__cxa_demangle(
          typeid(typename Srgemm::MultiplicationOp).name(), //
          nullptr, nullptr, nullptr));

      std::string add_op_name(
          add_op_name_full.substr(0, add_op_name_full.find_first_of('<')));
      std::string mult_op_name(
          mult_op_name_full.substr(0, mult_op_name_full.find_first_of('<')));

      std::stringstream fname;
      fname << "error_Srgemm_device_" << problem_size.m() << 'x' << problem_size.n()
            << 'x' << problem_size.k() << '_' << add_op_name << '_' << mult_op_name << '_'
            << Srgemm::ThreadblockShape::kM << 'x' << Srgemm::ThreadblockShape::kN << 'x'
            << Srgemm::ThreadblockShape::kK << '_' << Srgemm::WarpShape::kM << 'x'
            << Srgemm::WarpShape::kN << 'x' << Srgemm::WarpShape::kK << ".txt";

      std::ofstream file(fname.str());
      file << "problem: " << problem_size << ", alpha: " << alpha << ", beta: " << beta
           << "\n\n";

      file << "Addition operator: " << add_op_name_full << '\n';
      file << "Multiplication operator: " << mult_op_name_full << '\n';

      file << "A =\n"
           << tensor_A.host_view() << "\nB =\n"
           << tensor_B.host_view() << "\nC =\n"
           << tensor_C.host_view() << "\n\nReference =\n"
           << reference_D.host_view() << "\nComputed =\n"
           << tensor_D.host_view();
    }

    return passed;
  }

  /// Verifies the result is a GEMM
  bool verify(
      cutlass::gemm::GemmCoord problem_size, ElementCompute alpha, ElementCompute beta) {
    cuasr::reference::host::Srgemm<
        typename Srgemm::AdditionOp,                           //
        typename Srgemm::MultiplicationOp,                     //
        typename Srgemm::ElementA, typename Srgemm::LayoutA,   //
        typename Srgemm::ElementB, typename Srgemm::LayoutB,   //
        typename Srgemm::ElementC, typename Srgemm::LayoutC,   //
        typename Srgemm::EpilogueOutputOp::ElementCompute,     //
        typename Srgemm::EpilogueOutputOp::ElementAccumulator, //
        typename Srgemm::EpilogueOutputOp>
        reference_srgemm;

    reference_srgemm(
        problem_size, alpha, tensor_A.host_ref(), tensor_B.host_ref(), //
        beta, tensor_C.host_ref(), reference_D.host_ref(),             //
        Srgemm::AdditionOp::Identity);

    return compare_reference(problem_size, alpha, beta);
  }

  // Executes one test
  bool
  run(cutlass::gemm::GemmCoord problem_size,
      int split_k_slices   = 1,
      ElementCompute alpha = ElementCompute(Srgemm::MultiplicationOp::Identity),
      ElementCompute beta  = ElementCompute(Srgemm::MultiplicationOp::Identity)) {
    this->initialize(problem_size);

    // Initialize the GEMM operator
    typename Srgemm::Arguments arguments {
      problem_size,                 //
      tensor_A.device_ref(),        //
      tensor_B.device_ref(),        //
      tensor_C.device_ref(),        //
      tensor_D.device_ref(),        //
      { alpha, beta },              //
      Srgemm::AdditionOp::Identity, //
      split_k_slices                //
    };

    Srgemm gemm_op;
    size_t workspace_size = Srgemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    cutlass::Status status = gemm_op.initialize(arguments, workspace.get());
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    // Run the GEMM
    status = gemm_op();
    EXPECT_TRUE(status == cutlass::Status::kSuccess) << to_string(status);

    // Verify
    bool passed = this->verify(problem_size, alpha, beta);
    if (!passed) {
      std::cout << "Error with split_k_slices = " << split_k_slices
                << ", alpha: " << alpha << " and beta: " << beta << std::endl;
    }

    return passed;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Wrapper class to run different problem sizes and input combinations
template <typename Srgemm>
bool TestAllGemm() {
  bool passed = true;

  int const kMinimumOperandElementSize = std::min(
      int(cutlass::sizeof_bits<typename Srgemm::ElementA>::value),
      int(cutlass::sizeof_bits<typename Srgemm::ElementB>::value));

  int const kAlignment
      = cutlass::platform::is_same<
            typename Srgemm::OperatorClass, cutlass::arch::OpClassSimt>::value
      ? 1
      : 128 / kMinimumOperandElementSize;

  // int8_t gemm alignment constraints
  int const kAlignmentM
      = cutlass::platform::is_same<
            typename Srgemm::OperatorClass, cutlass::arch::OpClassSimt>::value
          && cutlass::platform::is_same<typename Srgemm::ElementA, int8_t>::value
          && cutlass::platform::is_same<
              typename Srgemm::LayoutA, cutlass::layout::ColumnMajor>::value
      ? 4
      : kAlignment;

  int const kAlignmentN
      = cutlass::platform::is_same<
            typename Srgemm::OperatorClass, cutlass::arch::OpClassSimt>::value
          && cutlass::platform::is_same<typename Srgemm::ElementB, int8_t>::value
          && cutlass::platform::is_same<
              typename Srgemm::LayoutB, cutlass::layout::RowMajor>::value
      ? 4
      : kAlignment;

  int const kAlignmentK
      = cutlass::platform::is_same<
            typename Srgemm::OperatorClass, cutlass::arch::OpClassSimt>::value
          && cutlass::platform::is_same<typename Srgemm::ElementA, int8_t>::value
          && cutlass::platform::is_same<typename Srgemm::ElementB, int8_t>::value
          && (cutlass::platform::is_same<
                  typename Srgemm::LayoutA, cutlass::layout::RowMajor>::value
              || cutlass::platform::is_same<
                  typename Srgemm::LayoutB, cutlass::layout::ColumnMajor>::value)
      ? 4
      : kAlignment;

  int problem_size_m[] = { kAlignmentM, 512 - 3 * kAlignmentM };

  int problem_size_n[] = { kAlignmentN, 512 - 2 * kAlignmentN };

  int problem_size_k[]
      = { kAlignmentK,
          Srgemm::ThreadblockShape::kK * (Srgemm::kStages + 1) - kAlignmentK };

  // TODO: add split-K SRGEMM
  int split_k_slices[] = { 1, 2, 3, 8 };

  double problem_alpha[] = { Srgemm::MultiplicationOp::Identity };
  double problem_beta[]  = { Srgemm::MultiplicationOp::Annihilator };

  Testbed<Srgemm> testbed;
  using ElementCompute = typename Srgemm::EpilogueOutputOp::ElementCompute;

  for (int m : problem_size_m) {
    for (int n : problem_size_n) {
      for (int k : problem_size_k) {
        for (int split_k : split_k_slices) {
          if (!Srgemm::kSplitKSerial && split_k > 1) {
            continue;
          }

          if (split_k > 1 && k / Srgemm::ThreadblockShape::kK < split_k) {
            continue;
          }

          for (auto alpha : problem_alpha) {
            for (auto beta : problem_beta) {
              cutlass::gemm::GemmCoord problem_size(m, n, k);

              passed = testbed.run(
                  problem_size, split_k, cutlass::from_real<ElementCompute>(alpha),
                  cutlass::from_real<ElementCompute>(beta));

              if (!passed) {
                return false;
              }
            }
          }
        }
      }
    }
  }

  return passed;
}

} // namespace device
} // namespace gemm
} // namespace test
} // namespace cuasr
