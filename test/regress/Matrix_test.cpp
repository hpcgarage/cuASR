#include "gtest/gtest.h"

#include "fwgpu/Matrix.hpp"

TEST(cuASR_Matrix, BasicConstructorCorrect) {
  auto x = fwgpu::Matrix<float>(6, 2);
  for (auto i = 0u; i < 12; ++i) {
    x(i) = (float)i;
  }

  EXPECT_EQ(size_t { 12 }, x.size());
  EXPECT_EQ(size_t { 12 * sizeof(float) }, x.bytesize());
  EXPECT_EQ(size_t { 6 }, x.num_rows());
  EXPECT_EQ(size_t { 2 }, x.num_cols());
  EXPECT_FLOAT_EQ(10.0f, x(10));
  EXPECT_FLOAT_EQ(0.0f, x(0, 0));
  EXPECT_FLOAT_EQ(8.0f, x(2, 1));
  EXPECT_FLOAT_EQ(11.0f, x(5, 1));
}

TEST(cuASR_Matrix, InitializerListConstructorCorrect) {
  // [8.0   3.0   0.0   1.0]
  // [2.0   5.0   4.0   9.0]
  // [7.0   6.0   10.   13.]
  auto x = fwgpu::Matrix<float>(
      3, 4, { 8.0, 2.0, 7.0, 3.0, 5.0, 6.0, 0.0, 4.0, 10.0, 1.0, 9.0, 13.0 });

  EXPECT_EQ(size_t { 12 }, x.size());
  EXPECT_EQ(size_t { 12 * sizeof(float) }, x.bytesize());
  EXPECT_EQ(size_t { 3 }, x.num_rows());
  EXPECT_EQ(size_t { 4 }, x.num_cols());
  EXPECT_FLOAT_EQ(8.0f, x(0, 0));
  EXPECT_FLOAT_EQ(1.0f, x(0, 3));
  EXPECT_FLOAT_EQ(10.0f, x(2, 2));
}

TEST(cuASR_Matrix, RandomFloatMatrixConstructorCorrect) {
  size_t const seed  = 8;
  auto const minimum = 1.0545;
  auto const maximum = 28.1;
  auto x             = fwgpu::Matrix<double>(9, 8, seed, minimum, maximum);

  EXPECT_EQ(size_t { 9 * 8 }, x.size());
  EXPECT_EQ(size_t { 9 * 8 * sizeof(double) }, x.bytesize());
  EXPECT_EQ(size_t { 9 }, x.num_rows());
  EXPECT_EQ(size_t { 8 }, x.num_cols());

  for (auto i = 0u; i < x.size(); ++i) {
    double const val = x(i);
    EXPECT_TRUE((val >= minimum && val <= maximum));
  }
}

TEST(cuASR_Matrix, RandomIntMatrixConstructorCorrect) {
  size_t const seed  = 8;
  auto const minimum = 1;
  auto const maximum = 128;
  auto x             = fwgpu::Matrix<int>(7, 5, seed, minimum, maximum);

  EXPECT_EQ(size_t { 7 * 5 }, x.size());
  EXPECT_EQ(size_t { 7 * 5 * sizeof(int) }, x.bytesize());
  EXPECT_EQ(size_t { 7 }, x.num_rows());
  EXPECT_EQ(size_t { 5 }, x.num_cols());

  for (auto i = 0u; i < x.size(); ++i) {
    int const val = x(i);
    EXPECT_TRUE((val >= minimum && val <= maximum));
  }
}

TEST(cuASR_Matrix, CopyConstructorCorrect) {
  auto from = fwgpu::Matrix<float>(5, 7, 0.0f);
  auto to   = from;
  EXPECT_TRUE(from == to);
}

TEST(cuASR_Matrix, MoveConstructorCorrect) {
  auto from = fwgpu::Matrix<float>(5, 7, 0.0f);
  EXPECT_EQ(from.num_rows(), 5);
  EXPECT_EQ(from.num_cols(), 7);
  EXPECT_TRUE(from.get_buf() != nullptr);

  auto to = fwgpu::Matrix<float, fwgpu::ColumnMajor>(std::move(from));
  EXPECT_EQ(to.num_rows(), 5);
  EXPECT_EQ(to.num_cols(), 7);
  EXPECT_TRUE(to.get_buf() != nullptr);

  EXPECT_EQ(from.num_rows(), 0);
  EXPECT_EQ(from.num_cols(), 0);
  EXPECT_TRUE(from.get_buf() == nullptr);
}

TEST(cuASR_Matrix, ConstantConstructorCorrect) {
  auto mat = fwgpu::Matrix<int>(6, 2, 42);
  EXPECT_EQ(mat.num_rows(), 6);
  EXPECT_EQ(mat.num_cols(), 2);
  EXPECT_EQ(mat(0, 0), 42);
  EXPECT_EQ(mat(3, 0), 42);
  EXPECT_EQ(mat(3, 1), 42);
  EXPECT_EQ(mat(5, 1), 42);
}

TEST(cuASR_Matrix, BufferConstructorCorrect) {
  std::vector<int> matvals(12, 42);
  auto mat = fwgpu::Matrix<int>(6, 2, matvals.data());

  EXPECT_EQ(mat.num_rows(), 6);
  EXPECT_EQ(mat.num_cols(), 2);
  EXPECT_EQ(mat(0, 0), 42);
  EXPECT_EQ(mat(3, 0), 42);
  EXPECT_EQ(mat(3, 1), 42);
  EXPECT_EQ(mat(5, 1), 42);
}

TEST(cuASR_Matrix, CopyAssignmentCorrect) {
  auto from = fwgpu::Matrix<float>(5, 7, 0.0f);
  fwgpu::Matrix<float> to(1, 1);
  to = from;
  EXPECT_TRUE(from == to);
}

TEST(cuASR_Matrix, ColumnMajorLayoutCorrect) {
  auto mat = fwgpu::Matrix<float, fwgpu::ColumnMajor>(
      4, 4,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.197551370, 0.553969979, 0.513400912, //
          0.729605675, 0.335222751, 0.477397054, 0.952229738, //
          0.798440039, 0.768229604, 0.628870904, 0.916195095  //
      });

  // corners
  EXPECT_FLOAT_EQ(mat(0, 0), 0.840187728);
  EXPECT_FLOAT_EQ(mat(0, 3), 0.798440039);
  EXPECT_FLOAT_EQ(mat(3, 0), 0.364784479);
  EXPECT_FLOAT_EQ(mat(3, 3), 0.916195095);

  // middle 2x2
  EXPECT_FLOAT_EQ(mat(1, 1), 0.197551370);
  EXPECT_FLOAT_EQ(mat(1, 2), 0.335222751);
  EXPECT_FLOAT_EQ(mat(2, 1), 0.553969979);
  EXPECT_FLOAT_EQ(mat(2, 2), 0.477397054);
}

TEST(cuASR_Matrix, RowMajorLayoutCorrect) {
  auto mat = fwgpu::Matrix<float, fwgpu::RowMajor>(
      4, 4,
      {
          0.840187728, 0.911647379, 0.277774721, 0.364784479, //
          0.394382924, 0.197551370, 0.553969979, 0.513400912, //
          0.729605675, 0.335222751, 0.477397054, 0.952229738, //
          0.798440039, 0.768229604, 0.628870904, 0.916195095  //
      });

  // corners
  EXPECT_FLOAT_EQ(mat(0, 0), 0.840187728);
  EXPECT_FLOAT_EQ(mat(0, 3), 0.364784479);
  EXPECT_FLOAT_EQ(mat(3, 0), 0.798440039);
  EXPECT_FLOAT_EQ(mat(3, 3), 0.916195095);

  // middle 2x2
  EXPECT_FLOAT_EQ(mat(1, 1), 0.197551370);
  EXPECT_FLOAT_EQ(mat(1, 2), 0.553969979);
  EXPECT_FLOAT_EQ(mat(2, 1), 0.335222751);
  EXPECT_FLOAT_EQ(mat(2, 2), 0.477397054);
}
