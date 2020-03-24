#ifndef FWGPU_UTILS
#define FWGPU_UTILS

namespace fwgpu {


auto malloc_device(void **dptr, size_t size) -> int;
auto free_device(void *dptr) -> int;

auto memcpy_d2h(void *dest, const void *src, size_t size) -> int;
auto memcpy_h2d(void *dest, const void *src, size_t size) -> int;

auto memcpy_2d_h2d(
    void *deset,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int;
auto memcpy_2d_d2h(
    void *deset,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int;

} // namespace fwgpu

#endif // FWGPU_UTILS
