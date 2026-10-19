/* One dimensional convolution (Metal) with compile-time kernel constants */

// Metal constants and compatibility macros for translated OpenCL kernels
#include <metal_stdlib>
using namespace metal;

// Storage-class mappings
#define __kernel kernel
#define __global device
#define __local threadgroup
#define READONLY device
#define READWRITE device

// Math helpers missing in MSL but present in OpenCL
#ifndef M_PI_F
  #define M_PI_F 3.14159265358979323846f
#endif

#define atan2pi(y,x) (atan2((y),(x)) / M_PI_F)

// Vector constants (use MSL constructors)
#define ZERO 0.0f
#define ZERO2 float2(ZERO, ZERO)
#define ZERO3 float3(ZERO, ZERO, ZERO)
#define ZERO4 float4(ZERO, ZERO, ZERO, ZERO)
#define ONE 1.0f
#define ONE2 float2(ONE, ONE)
#define ONE3 float3(ONE, ONE, ONE)
#define ONE4 float4(ONE, ONE, ONE, ONE)
#define FLOAT float
#define FLOAT2 float2
#define FLOAT3 float3
#define FLOAT4 float4

// Workgroup barrier compatibility
#define barrier(flags) threadgroup_barrier(mem_flags::mem_device)

// Atomic increment compatibility for device memory
static inline void atomic_inc(device atomic_int* p) {
    (void)atomic_fetch_add_explicit(p, 1, memory_order_relaxed);
}




#define FAST 1000

kernel void binomial5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > (size_t)2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.0625;
        sum = sum + a[i+(-1)] * 0.25;
        sum = sum + a[i+(0)] * 0.375;
        sum = sum + a[i+(1)] * 0.25;
        sum = sum + a[i+(2)] * 0.0625;
    }
    ret[i] = sum;
}

kernel void binomial5_fast(constant int &width [[ buffer(0) ]],
                         device float* a [[ buffer(1) ]],
                         device float* ret [[ buffer(2) ]],
                         constant int &chunk [[ buffer(3) ]],
                         uint i [[ thread_position_in_grid ]])
{
    size_t idx = i * (size_t)chunk;
    float sum = 0.0f;
    size_t right = width-2;
    threadgroup float buffer[FAST];
    for (int j = 0; j < 5; j++) {
        buffer[j] = idx+j >= (size_t)width ? 0.0f : a[idx+j];
    }
    for (int j = 0; j < chunk-5; j++) {
        sum = 0.0f;
        size_t ipixel = idx+j >= right ? (right-1) : (idx+j);
        if ((idx > (size_t)2) &&  (idx < right)) {
            sum = sum + buffer[0] * 0.0625;
            sum = sum + buffer[1] * 0.25;
            sum = sum + buffer[2] * 0.375;
            sum = sum + buffer[3] * 0.25;
            sum = sum + buffer[4] * 0.0625;
        }
        ret[ipixel] = sum;
    }
}
