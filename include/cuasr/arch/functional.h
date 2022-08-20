/***************************************************************************************************
 * Copyright (c) 2022, Vijay Thakkar (thakkarv@gatech.edu).
 **************************************************************************************************/
#pragma once

namespace cuasr {
namespace arch {

///////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline
int min(int lhs, int rhs) {
#if defined(__CUDA_ARCH__)
  int ret;
  asm("min.s32 %0, %1, %2;\n" : "=r"(ret) : "r"(lhs), "r"(rhs));
  return ret;
#else
  return (lhs < rhs) ? lhs : rhs;
#endif
}

__host__ __device__ inline
float min(float lhs, float rhs) {
#if defined(__CUDA_ARCH__)
  float ret;
  asm("min.f32 %0, %1, %2;\n" : "=f"(ret) : "f"(lhs), "f"(rhs));
  return ret;
#else
  return (lhs < rhs) ? lhs : rhs;
#endif
}

__host__ __device__ inline
double min(double lhs, double rhs) {
#if defined(__CUDA_ARCH__)
  double ret;
  asm("min.f64 %0, %1, %2;\n" : "=d"(ret) : "d"(lhs), "d"(rhs));
  return ret;
#else
  return (lhs < rhs) ? lhs : rhs;
#endif
}

///////////////////////////////////////////////////////////////////////////////

__host__ __device__ inline
int max(int lhs, int rhs) {
#if defined(__CUDA_ARCH__)
  int ret;
  asm("max.s32 %0, %1, %2;\n" : "=r"(ret) : "r"(lhs), "r"(rhs));
  return ret;
#else
  return (lhs > rhs) ? lhs : rhs;
#endif
}

__host__ __device__ inline
float max(float lhs, float rhs) {
#if defined(__CUDA_ARCH__)
  float ret;
  asm("max.f32 %0, %1, %2;\n" : "=f"(ret) : "f"(lhs), "f"(rhs));
  return ret;
#else
  return (lhs > rhs) ? lhs : rhs;
#endif
}

__host__ __device__ inline
double max(double lhs, double rhs) {
#if defined(__CUDA_ARCH__)
  double ret;
  asm("max.f64 %0, %1, %2;\n" : "=d"(ret) : "d"(lhs), "d"(rhs));
  return ret;
#else
  return (lhs > rhs) ? lhs : rhs;
#endif
}

///////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cuasr
