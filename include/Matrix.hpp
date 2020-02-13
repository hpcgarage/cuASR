#ifndef FWGPU_MATRIX_HPP
#define FWGPU_MATRIX_HPP

#include <cstring>
#include <initializer_list>
#include <memory>
#include <ostream>
#include <random>

namespace fwgpu {

/*
 * Matrix datastructure: Wrapper around a buffer of T.
 * T = float and column major by default.
 **/
template <typename T = float>
class Matrix {
  // sizeof(Matrix<T>) == 32 bytes for any T that has a default deleter
  // NOTE: m_buf_raw is used here only for internal indexing purposes
private:
  std::unique_ptr<T> m_buf;
  size_t m_rows;
  size_t m_cols;
  T *m_buf_raw;

public:
  /*
   * No default contructor.
   **/
  Matrix() = delete;

  /*
   * Default distructor. Unique pointer will take care of buffer lifetime.
   **/
  ~Matrix() = default;

  /*
   * De-facto default constructor: allocate T buffer of size rows*cols
   **/
  Matrix(size_t rows, size_t cols)
      : m_buf(new T[rows * cols])
      , m_rows(rows)
      , m_cols(cols) {
    m_buf_raw = m_buf.get();
  }

  /*
   * Assign buf from external source.
   * TODO: not sure we should allow this?
   **/
  Matrix(size_t rows, size_t cols, T *buf)
      : m_rows(rows)
      , m_cols(cols) {
    m_buf.reset(buf);
    m_buf_raw = m_buf.get();
  }

  /*
   * Allocate and initialize buf with input value.
   **/
  Matrix(size_t rows, size_t cols, T val)
      : m_buf(new T[rows * cols])
      , m_rows(rows)
      , m_cols(cols) {
    m_buf_raw = m_buf.get();
    for (auto i = 0ull; i < (rows * cols); ++i) {
      m_buf_raw[i] = val;
    }
  }

  /*
   * Allocate and initialize of a buffer of size rows*cols.
   * Assign random numbers in the input range.
   **/
  Matrix(size_t rows, size_t cols, size_t seed, T min = T(0.0), T max = T(1.0))
      : m_buf(new T[rows * cols])
      , m_rows(rows)
      , m_cols(cols) {
    auto rng  = std::mt19937_64(seed);
    auto dist = std::uniform_real_distribution<T>(min, max);
    m_buf_raw = m_buf.get();
    for (auto i = 0ull; i < (rows * cols); ++i) {
      m_buf_raw[i] = dist(rng);
    }
  }

  /*
   * Allocate and initialize buffer from an initializer list.
   * This mainly makes testing easier.
   **/
  Matrix(size_t rows, size_t cols, const std::initializer_list<T> &elements)
      : m_buf(new T[rows * cols])
      , m_rows(rows)
      , m_cols(cols) {
    m_buf_raw = m_buf.get();
    for (auto val : elements) {
      *(m_buf_raw++) = val;
    }
    m_buf_raw = get_buf();
  }

  /*
   * Copy constructor: allocate new buffer and memcpy from other
   **/
  Matrix(const Matrix &other)
      : m_rows(other.m_rows)
      , m_cols(other.m_cols) {
    m_buf.reset(new T[other.size()]);
    std::memcpy(m_buf.get(), other.m_buf.get(), other.bytesize());
    m_buf_raw = get_buf();
  }

  Matrix(Matrix &&other)
      : m_rows(other.m_rows)
      , m_cols(other.m_cols) {
    m_buf.reset(other.m_buf.release());
    m_buf_raw = get_buf();
  }

  /*
   * Copy assignment operator.
   **/
  auto operator=(const Matrix &other) -> Matrix & {
    m_rows = other.num_rows();
    m_cols = other.num_cols();
    m_buf.reset(new T[other.size()]);
    std::memcpy(m_buf.get(), other.m_buf.get(), other.bytesize());
    m_buf_raw = get_buf();
    return *this;
  }

  /*
   * Returns a non-owning, const pointer to the backing buffer of type T[].
   **/
  auto get_buf() const -> const T * { return const_cast<const T *>(m_buf.get()); }

  /*
   * Returns a non-owning pointer to the backing buffer of type T[].
   **/
  auto get_buf() noexcept -> T * { return m_buf.get(); }

  /*
   * Returns total number of elements stored in the matrix.
   **/
  auto size() const noexcept -> size_t { return m_rows * m_cols; }

  /*
   * Returns total number of bytes occupied by the backing store T[].
   **/
  auto bytesize() const noexcept -> size_t { return size() * sizeof(T); }

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
  auto operator[](size_t idx) const -> T & { return m_buf_raw[idx]; }

  /*
   * Linear index into the flat buffer.
   **/
  auto operator()(size_t idx) const -> T & { return m_buf_raw[idx]; }

  /*
   * Matrix index with major dimention offset.
   * Column major for now, but we can add support for changing to row major later
   * with some template magic.
   */
  auto operator()(size_t row_idx, size_t col_idx) const -> T & {
    return m_buf_raw[row_idx + (col_idx * m_rows)];
  }
};

// Element-wise equality test for two matrices of the same template type.
template <typename T>
inline auto operator==(const Matrix<T> &lhs, const Matrix<T> &rhs) -> bool {
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
template <typename T>
inline auto operator!=(Matrix<T> &lhs, Matrix<T> &rhs) -> bool {
  return !(lhs == rhs);
}

// Prints matrix to stdout for sanity-checks etc.
template <typename T>
inline auto operator<<(std::ostream &os, const Matrix<T> &mat) -> std::ostream & {
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
