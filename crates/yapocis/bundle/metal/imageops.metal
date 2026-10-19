// Metal translations of imageops kernels
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



kernel void clamp_pixels(const device float *input [[ buffer(0) ]],
                         constant float &low [[ buffer(1) ]],
                         constant float &high [[ buffer(2) ]],
                         device float *output [[ buffer(3) ]],
                         uint i [[ thread_position_in_grid ]]) {
    float value = input[i];
    value = value < low ? low:  value;
    value = value > high ? high : value;
    output[i] = value;
}

kernel void stretch(const device float *input [[ buffer(0) ]],
                    constant float &input_min [[ buffer(1) ]],
                    constant float &input_max [[ buffer(2) ]],
                    constant float &low [[ buffer(3) ]],
                    constant float &high [[ buffer(4) ]],
                    device float *output [[ buffer(5) ]],
                    uint i [[ thread_position_in_grid ]]) {
    float value = input[i];
    value = value < input_min ? input_min : value;
    value = value > input_max ? input_max : value;
    value = (value - input_min)/(input_max-input_min);
    value = low + value * (high-low);
    output[i] = value;
}

kernel void binarize(const device float *input [[ buffer(0) ]],
                     constant float &threshold [[ buffer(1) ]],
                     device float *output [[ buffer(2) ]],
                     uint i [[ thread_position_in_grid ]]) {
    float value = input[i];
    value = value >= threshold ? 1.0 : 0.0;
    output[i] = value;
}

kernel void histogram(const device float *input [[ buffer(0) ]],
                      constant int &size_of_input [[ buffer(1) ]],
                      constant float &low [[ buffer(2) ]],
                      constant float &high [[ buffer(3) ]],
                      device int *hst [[ buffer(4) ]],
                      constant int &size_of_hst [[ buffer(5) ]],
                      uint i [[ thread_position_in_grid ]]) {
    (void)size_of_input; (void)size_of_hst; // global size determines coverage
    int n_bins = size_of_hst;
    float value = input[i];
    value = value < low ? low : value;
    value = value > high ? high : value;
    uint bucket = (uint)((n_bins*value)/(high-low));
    bucket = min((uint)(n_bins-1), bucket);
    // Atomic increment on device memory
    atomic_inc((device atomic_int*)(hst + bucket));
}
