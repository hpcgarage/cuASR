# SemiRing GEMM

## Compiling:
```sh
$ git clone --recurse-submodules https://github.com/hpcgarage/semiring-gemm /path/to/repo
$ cd /path/to/repo
$ mkdir build && cd build
$ cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RELEASE
$ ninja
```

## Benchmarks:
This project uses the [`google/benchmark`](https://github.com/google/benchmark) micro-benchmark support library for measuring the performance of each kernel. An additional custom counter has been added to all the kernels to calculate flop rate of each kernel.

Benchmarks can be run using: `./build/bench/${bench_suite} --benchmark_counters_tabular=true`

Available `benchmark_suite` are: `cpu_bench` and `gpu_bench`

## Tests:
This project uses the [`google/googletest`](https://github.com/google/googletest) xUnit test framework for testing the correctness of all the kernels.

All the tests can be run using a single CMake target as such: `cd build && ninja test`
Individual tests can also be run with `./build/test/${test_name}`. Available tests are:
```
Matrix_tests
Gemm_tests
Srgemm_tests
```
