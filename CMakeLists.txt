cmake_minimum_required(VERSION 3.22)
project(neon_spmv)

set(CMAKE_CXX_STANDARD 20)

include_directories(/usr/lib/gcc/aarch64-linux-gnu/10.3.1/include)

add_executable(neon_spmv main.cpp
        src/kernel/spmv_kernel_FP32_sy.h)
