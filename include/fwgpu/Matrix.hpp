#ifndef FWGPU_MATRIX_HPP
#define FWGPU_MATRIX_HPP

#include <cstring>
#include <initializer_list>
#include <ostream>
#include <random>
#include <vector>

namespace fwgpu {

/*
 * Matrix datastructure: Wrapper around a buffer of ElementT.
 * ElementT = float and column major by default.
 **/
template <typename ElementT = float>
class Matrix {
private:
  std::vector<ElementT> m_host_buf;
  size_t m_rows;
  size_t m_cols;

public:
  /*
   * No default contructor.
   **/
  Matrix() = delete;

  /*
   * Default distructor.
   **/
  ~Matrix() = default;

  /*
   * De-facto default constructor: allocate ElementT buffer of size rows*cols
   **/
  Matrix(size_t rows, size_t cols)
      : m_rows(rows)
      , m_cols(cols) {
    m_host_buf.reserve(rows * cols);
  }

  /*
   * Assign buf from external source.
   * TODO: not sure we should allow this?
   **/
  Matrix(size_t rows, size_t cols, ElementT *buf)
      : m_host_buf(buf, buf + (rows * cols))
      , m_rows(rows)
      , m_cols(cols) { }

  /*
   * Allocate and initialize buf with input value.
   **/
  Matrix(size_t rows, size_t cols, ElementT val)
      : m_host_buf(rows * cols)
      , m_rows(rows)
      , m_cols(cols) {
    for (auto i = 0ull; i < (rows * cols); ++i) {
      m_host_buf[i] = val;
    }
  }

  /*
   * Allocate and initialize of a buffer of size rows*cols.
   * Assign random numbers in the input range.
   **/
  Matrix(
      size_t rows,
      size_t cols,
      size_t seed,
      ElementT min = ElementT(0.0),
      ElementT max = ElementT(1.0))
      : m_host_buf(rows * cols)
      , m_rows(rows)
      , m_cols(cols) {
    auto rng  = std::mt19937_64(seed);
    auto dist = std::uniform_real_distribution<ElementT>(min, max);
    m_host_buf.reserve(cols * rows);
    for (auto i = 0ull; i < (rows * cols); ++i) {
      m_host_buf[i] = dist(rng);
    }
  }

  /*
   * Allocate and initialize buffer from an initializer list.
   * This mainly makes testing easier.
   **/
  Matrix(size_t rows, size_t cols, const std::initializer_list<ElementT> &elements)
      : m_host_buf(rows * cols)
      , m_rows(rows)
      , m_cols(cols) {
    auto i = 0ull;
    for (auto val : elements) {
      m_host_buf[i++] = val;
    }
  }

  /*
   * Copy constructor: allocate new buffer and memcpy from other
   **/
  Matrix(const Matrix &other)
      : m_rows(other.m_rows)
      , m_cols(other.m_cols) {
    m_host_buf = other.m_host_buf;
  }

  Matrix(Matrix &&other)
      : m_rows(other.m_rows)
      , m_cols(other.m_cols) {
    m_host_buf = std::move(other.m_host_buf);
  }

  /*
   * Copy assignment operator.
   **/
  auto operator=(const Matrix &other) -> Matrix & {
    m_rows     = other.num_rows();
    m_cols     = other.num_cols();
    m_host_buf = other.m_host_buf;
    return *this;
  }

  /*
   * Returns a non-owning, const pointer to the backing buffer of type ElementT[].
   **/
  auto get_buf() const -> const ElementT * { return m_host_buf.data(); }

  /*
   * Returns a non-owning pointer to the backing buffer of type ElementT[].
   **/
  auto get_buf() noexcept -> ElementT * { return m_host_buf.data(); }

  /*
   * Returns total number of elements stored in the matrix.
   **/
  auto size() const noexcept -> size_t { return m_rows * m_cols; }

  /*
   * Returns total number of bytes occupied by the backing store ElementT[].
   **/
  auto bytesize() const noexcept -> size_t { return size() * sizeof(ElementT); }

  /*
   * Returns true if matrix has (0, 0) dimentions. False otherwise.
   **/
  auto is_empty() const noexcept -> size_t { return (m_rows == 0) || (m_cols == 0); }

  /*
   * Returns numbers of rows in the matrix.
   **/
  auto num_rows() const noexcept -> size_t { return m_rows; }

  /*
   * Returns numbers of columns in the matrix.
   **/
  auto num_cols() const noexcept -> size_t { return m_cols; }

  /*
   * Linear index into the flat buffer.
   **/
  auto operator[](size_t idx) -> ElementT & { return m_host_buf[idx]; }
  auto operator[](size_t idx) const -> ElementT const & { return m_host_buf[idx]; }

  /*
   * Linear index into the flat buffer.
   **/
  auto operator()(size_t idx) -> ElementT & { return m_host_buf[idx]; }
  auto operator()(size_t idx) const -> ElementT const & { return m_host_buf[idx]; }

  /*
   * Matrix index with major dimention offset.
   * Column major for now, but we can add support for changing to row major later
   * with some template magic.
   */
  auto operator()(size_t row_idx, size_t col_idx) -> ElementT & {
    return m_host_buf[row_idx + (col_idx * m_rows)];
  }

  auto operator()(size_t row_idx, size_t col_idx) const -> ElementT const & {
    return m_host_buf[row_idx + (col_idx * m_rows)];
  }
};

// Element-wise equality test for two matrices of the same template type.
template <typename ElementT>
inline auto operator==(const Matrix<ElementT> &lhs, const Matrix<ElementT> &rhs) -> bool {
  // both dims much match first
  if ((lhs.num_rows() != rhs.num_rows()) || (lhs.num_cols() != rhs.num_cols())) {
    return false;
  }

  for (auto i = 0ull; i < lhs.size(); ++i) {
    if (lhs[i] < rhs[i]) {
      return false;
    }
  }

  return true;
}

// Element-wise inequality test for two matrices of the same template type.
template <typename ElementT>
inline auto operator!=(Matrix<ElementT> &lhs, Matrix<ElementT> &rhs) -> bool {
  return !(lhs == rhs);
}

// Prints matrix to stdout; prefer using this only for small matrices.
template <typename ElementT>
inline auto operator<<(std::ostream &os, const Matrix<ElementT> &mat) -> std::ostream & {
  for (auto row_idx = 0ull; row_idx < mat.num_rows(); ++row_idx) {
    os << '[' << mat(row_idx, 0);

    for (auto col_idx = 1ull; col_idx < mat.num_cols() - 1; ++col_idx) {
      os << ", " << mat(row_idx, col_idx);
    }

    os << "]\n";
  }

  return os;
}

} // namespace fwgpu

#endif // FWGPU_MATRIX_HPP
