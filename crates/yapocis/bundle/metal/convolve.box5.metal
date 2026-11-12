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





kernel void box5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > (size_t)2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.2;
        sum = sum + a[i+(-1)] * 0.2;
        sum = sum + a[i+(0)] * 0.2;
        sum = sum + a[i+(1)] * 0.2;
        sum = sum + a[i+(2)] * 0.2;
    }
    ret[i] = sum;
}

/* Thread g computes ret[g*chunk .. g*chunk+chunk-1], each pixel exactly as box5
   computes it, and stops at width. Needs a global size of at least ceil(width/chunk);
   a larger one (the default, one thread per element) is correct but idles. */
kernel void box5_fast(constant int &width [[ buffer(0) ]],
                         device float* a [[ buffer(1) ]],
                         device float* ret [[ buffer(2) ]],
                         constant int &chunk [[ buffer(3) ]],
                         uint g [[ thread_position_in_grid ]])
{
    int n = chunk > 0 ? chunk : 1;
    size_t right = width-2;
    size_t start = (size_t)g * (size_t)n;
    for (int j = 0; j < n; j++) {
        size_t i = start + (size_t)j;
        if (i >= (size_t)width) return;
        float sum = 0.0f;
        if ((i > (size_t)2) &&  (i < right)) {
            sum = sum + a[i+(-2)] * 0.2;
            sum = sum + a[i+(-1)] * 0.2;
            sum = sum + a[i+(0)] * 0.2;
            sum = sum + a[i+(1)] * 0.2;
            sum = sum + a[i+(2)] * 0.2;
        }
        ret[i] = sum;
    }
}
