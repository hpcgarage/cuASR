#pragma once

namespace fwgpu {

// device memory allocation of size bytes
auto malloc_device(void **dptr, size_t size) -> int;

// unified managed memory allocation of size bytes
auto malloc_unified(void **dptr, size_t size) -> int;

// free cuda allocated memory, managed or unmanaged
auto free_device(void *dptr) -> int;

// MEMCPY API
// memory copy: device -> host
auto memcpy_d2h(void *dest, const void *src, size_t size) -> int;

// memory copy: host -> device
auto memcpy_h2d(void *dest, const void *src, size_t size) -> int;

// memory copy: host -> host
auto memcpy_h2h(void *dest, const void *src, size_t size) -> int;

// memory copy: device -> device
auto memcpy_d2d(void *dest, const void *src, size_t size) -> int;

// memory copy: direction inferred based on src and dest. Requires unified memory.
auto memcpy_inferred(void *dest, const void *src, size_t size) -> int;

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
auto memcpy_2d_d2d(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int;
auto memcpy_2d_inferred(
    void *dest,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height) -> int;
} // namespace fwgpu
