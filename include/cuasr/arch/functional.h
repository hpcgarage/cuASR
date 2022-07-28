#pragma once

namespace cuasr {
namespace arch {

///////////////////////////////////////////////////////////////////////////////

template<class T>
__device__
T plus(T lhs, T rhs) {
  return lhs + rhs;
}

__device__
float min(float lhs, float rhs) {
  float ret;
  asm("min.f32 %0, %1, %2;\n" : "=f"(ret) : "f"(lhs), "f"(rhs));
  return ret;
}

///////////////////////////////////////////////////////////////////////////////

} // namespace arch
} // namespace cuasr
