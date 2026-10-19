/* Multiple one-dimensional convolutions (Metal) */

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




kernel void hamming5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.035714287;
        sum = sum + a[i+(-1)] * 0.24107143;
        sum = sum + a[i+(0)] * 0.44642857;
        sum = sum + a[i+(1)] * 0.24107143;
        sum = sum + a[i+(2)] * 0.035714287;
    }
    ret[i] = sum;
}

kernel void hamming9(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-4;
    if ((i > 4) &&  (i < right)) {
        sum = sum + a[i+(-4)] * 0.018181818;
        sum = sum + a[i+(-3)] * 0.048802473;
        sum = sum + a[i+(-2)] * 0.122727275;
        sum = sum + a[i+(-1)] * 0.19665207;
        sum = sum + a[i+(0)] * 0.22727273;
        sum = sum + a[i+(1)] * 0.19665207;
        sum = sum + a[i+(2)] * 0.122727275;
        sum = sum + a[i+(3)] * 0.048802473;
        sum = sum + a[i+(4)] * 0.018181818;
    }
    ret[i] = sum;
}

kernel void hamming13(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-6;
    if ((i > 6) &&  (i < right)) {
        sum = sum + a[i+(-6)] * 0.0121951215;
        sum = sum + a[i+(-5)] * 0.021589682;
        sum = sum + a[i+(-4)] * 0.047256097;
        sum = sum + a[i+(-3)] * 0.08231708;
        sum = sum + a[i+(-2)] * 0.11737805;
        sum = sum + a[i+(-1)] * 0.14304446;
        sum = sum + a[i+(0)] * 0.15243903;
        sum = sum + a[i+(1)] * 0.14304446;
        sum = sum + a[i+(2)] * 0.11737805;
        sum = sum + a[i+(3)] * 0.08231708;
        sum = sum + a[i+(4)] * 0.047256097;
        sum = sum + a[i+(5)] * 0.021589682;
        sum = sum + a[i+(6)] * 0.0121951215;
    }
    ret[i] = sum;
}

kernel void hamming17(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-8;
    if ((i > 8) &&  (i < right)) {
        sum = sum + a[i+(-8)] * 0.0091743115;
        sum = sum + a[i+(-7)] * 0.013189841;
        sum = sum + a[i+(-6)] * 0.0246251;
        sum = sum + a[i+(-5)] * 0.041739177;
        sum = sum + a[i+(-4)] * 0.061926607;
        sum = sum + a[i+(-3)] * 0.08211403;
        sum = sum + a[i+(-2)] * 0.09922811;
        sum = sum + a[i+(-1)] * 0.11066337;
        sum = sum + a[i+(0)] * 0.1146789;
        sum = sum + a[i+(1)] * 0.11066337;
        sum = sum + a[i+(2)] * 0.09922811;
        sum = sum + a[i+(3)] * 0.08211403;
        sum = sum + a[i+(4)] * 0.061926607;
        sum = sum + a[i+(5)] * 0.041739177;
        sum = sum + a[i+(6)] * 0.0246251;
        sum = sum + a[i+(7)] * 0.013189841;
        sum = sum + a[i+(8)] * 0.0091743115;
    }
    ret[i] = sum;
}

kernel void hamming25(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-12;
    if ((i > 12) &&  (i < right)) {
        sum = sum + a[i+(-12)] * 0.006134969;
        sum = sum + a[i+(-11)] * 0.0073369723;
        sum = sum + a[i+(-10)] * 0.010861067;
        sum = sum + a[i+(-9)] * 0.016467093;
        sum = sum + a[i+(-8)] * 0.023773005;
        sum = sum + a[i+(-7)] * 0.032280922;
        sum = sum + a[i+(-6)] * 0.041411042;
        sum = sum + a[i+(-5)] * 0.050541162;
        sum = sum + a[i+(-4)] * 0.05904908;
        sum = sum + a[i+(-3)] * 0.066355;
        sum = sum + a[i+(-2)] * 0.071961015;
        sum = sum + a[i+(-1)] * 0.07548511;
        sum = sum + a[i+(0)] * 0.07668712;
        sum = sum + a[i+(1)] * 0.07548511;
        sum = sum + a[i+(2)] * 0.071961015;
        sum = sum + a[i+(3)] * 0.066355;
        sum = sum + a[i+(4)] * 0.05904908;
        sum = sum + a[i+(5)] * 0.050541162;
        sum = sum + a[i+(6)] * 0.041411042;
        sum = sum + a[i+(7)] * 0.032280922;
        sum = sum + a[i+(8)] * 0.023773005;
        sum = sum + a[i+(9)] * 0.016467093;
        sum = sum + a[i+(10)] * 0.010861067;
        sum = sum + a[i+(11)] * 0.0073369723;
        sum = sum + a[i+(12)] * 0.006134969;
    }
    ret[i] = sum;
}

kernel void hann5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.0;
        sum = sum + a[i+(-1)] * 0.25;
        sum = sum + a[i+(0)] * 0.5;
        sum = sum + a[i+(1)] * 0.25;
        sum = sum + a[i+(2)] * 0.0;
    }
    ret[i] = sum;
}

kernel void hann9(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-4;
    if ((i > 4) &&  (i < right)) {
        sum = sum + a[i+(-4)] * 0.0;
        sum = sum + a[i+(-3)] * 0.036611654;
        sum = sum + a[i+(-2)] * 0.125;
        sum = sum + a[i+(-1)] * 0.21338835;
        sum = sum + a[i+(0)] * 0.25;
        sum = sum + a[i+(1)] * 0.21338835;
        sum = sum + a[i+(2)] * 0.125;
        sum = sum + a[i+(3)] * 0.036611654;
        sum = sum + a[i+(4)] * 0.0;
    }
    ret[i] = sum;
}

kernel void hann13(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-6;
    if ((i > 6) &&  (i < right)) {
        sum = sum + a[i+(-6)] * 0.0;
        sum = sum + a[i+(-5)] * 0.01116455;
        sum = sum + a[i+(-4)] * 0.041666668;
        sum = sum + a[i+(-3)] * 0.083333336;
        sum = sum + a[i+(-2)] * 0.125;
        sum = sum + a[i+(-1)] * 0.15550211;
        sum = sum + a[i+(0)] * 0.16666667;
        sum = sum + a[i+(1)] * 0.15550211;
        sum = sum + a[i+(2)] * 0.125;
        sum = sum + a[i+(3)] * 0.083333336;
        sum = sum + a[i+(4)] * 0.041666668;
        sum = sum + a[i+(5)] * 0.01116455;
        sum = sum + a[i+(6)] * 0.0;
    }
    ret[i] = sum;
}

kernel void hann17(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-8;
    if ((i > 8) &&  (i < right)) {
        sum = sum + a[i+(-8)] * 0.0;
        sum = sum + a[i+(-7)] * 0.004757529;
        sum = sum + a[i+(-6)] * 0.018305827;
        sum = sum + a[i+(-5)] * 0.038582284;
        sum = sum + a[i+(-4)] * 0.0625;
        sum = sum + a[i+(-3)] * 0.08641771;
        sum = sum + a[i+(-2)] * 0.10669418;
        sum = sum + a[i+(-1)] * 0.12024247;
        sum = sum + a[i+(0)] * 0.125;
        sum = sum + a[i+(1)] * 0.12024247;
        sum = sum + a[i+(2)] * 0.10669418;
        sum = sum + a[i+(3)] * 0.08641771;
        sum = sum + a[i+(4)] * 0.0625;
        sum = sum + a[i+(5)] * 0.038582284;
        sum = sum + a[i+(6)] * 0.018305827;
        sum = sum + a[i+(7)] * 0.004757529;
        sum = sum + a[i+(8)] * 0.0;
    }
    ret[i] = sum;
}

kernel void hann25(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-12;
    if ((i > 12) &&  (i < right)) {
        sum = sum + a[i+(-12)] * 0.0;
        sum = sum + a[i+(-11)] * 0.0014197573;
        sum = sum + a[i+(-10)] * 0.005582275;
        sum = sum + a[i+(-9)] * 0.012203884;
        sum = sum + a[i+(-8)] * 0.020833334;
        sum = sum + a[i+(-7)] * 0.03088254;
        sum = sum + a[i+(-6)] * 0.041666668;
        sum = sum + a[i+(-5)] * 0.052450795;
        sum = sum + a[i+(-4)] * 0.0625;
        sum = sum + a[i+(-3)] * 0.07112945;
        sum = sum + a[i+(-2)] * 0.077751055;
        sum = sum + a[i+(-1)] * 0.081913576;
        sum = sum + a[i+(0)] * 0.083333336;
        sum = sum + a[i+(1)] * 0.081913576;
        sum = sum + a[i+(2)] * 0.077751055;
        sum = sum + a[i+(3)] * 0.07112945;
        sum = sum + a[i+(4)] * 0.0625;
        sum = sum + a[i+(5)] * 0.052450795;
        sum = sum + a[i+(6)] * 0.041666668;
        sum = sum + a[i+(7)] * 0.03088254;
        sum = sum + a[i+(8)] * 0.020833334;
        sum = sum + a[i+(9)] * 0.012203884;
        sum = sum + a[i+(10)] * 0.005582275;
        sum = sum + a[i+(11)] * 0.0014197573;
        sum = sum + a[i+(12)] * 0.0;
    }
    ret[i] = sum;
}

kernel void blackman7(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * -5.5070587e-18;
        sum = sum + a[i+(-2)] * 0.051587302;
        sum = sum + a[i+(-1)] * 0.25;
        sum = sum + a[i+(0)] * 0.3968254;
        sum = sum + a[i+(1)] * 0.25;
        sum = sum + a[i+(2)] * 0.051587302;
        sum = sum + a[i+(3)] * -5.5070587e-18;
    }
    ret[i] = sum;
}

kernel void blackman11(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-5;
    if ((i > 5) &&  (i < right)) {
        sum = sum + a[i+(-5)] * -3.3042352e-18;
        sum = sum + a[i+(-4)] * 0.009574491;
        sum = sum + a[i+(-3)] * 0.047802415;
        sum = sum + a[i+(-2)] * 0.12137789;
        sum = sum + a[i+(-1)] * 0.20219758;
        sum = sum + a[i+(0)] * 0.23809524;
        sum = sum + a[i+(1)] * 0.20219758;
        sum = sum + a[i+(2)] * 0.12137789;
        sum = sum + a[i+(3)] * 0.047802415;
        sum = sum + a[i+(4)] * 0.009574491;
        sum = sum + a[i+(5)] * -3.3042352e-18;
    }
    ret[i] = sum;
}

kernel void blackman15(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-7;
    if ((i > 7) &&  (i < right)) {
        sum = sum + a[i+(-7)] * -2.360168e-18;
        sum = sum + a[i+(-6)] * 0.003298427;
        sum = sum + a[i+(-5)] * 0.015383235;
        sum = sum + a[i+(-4)] * 0.040248644;
        sum = sum + a[i+(-3)] * 0.07809234;
        sum = sum + a[i+(-2)] * 0.121418916;
        sum = sum + a[i+(-1)] * 0.15652442;
        sum = sum + a[i+(0)] * 0.17006803;
        sum = sum + a[i+(1)] * 0.15652442;
        sum = sum + a[i+(2)] * 0.121418916;
        sum = sum + a[i+(3)] * 0.07809234;
        sum = sum + a[i+(4)] * 0.040248644;
        sum = sum + a[i+(5)] * 0.015383235;
        sum = sum + a[i+(6)] * 0.003298427;
        sum = sum + a[i+(7)] * -2.360168e-18;
    }
    ret[i] = sum;
}

kernel void blackman21(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-10;
    if ((i > 10) &&  (i < right)) {
        sum = sum + a[i+(-10)] * -1.6521176e-18;
        sum = sum + a[i+(-9)] * 0.0010944168;
        sum = sum + a[i+(-8)] * 0.0047872453;
        sum = sum + a[i+(-7)] * 0.012069764;
        sum = sum + a[i+(-6)] * 0.023901207;
        sum = sum + a[i+(-5)] * 0.04047619;
        sum = sum + a[i+(-4)] * 0.060688946;
        sum = sum + a[i+(-3)] * 0.0820442;
        sum = sum + a[i+(-2)] * 0.10109879;
        sum = sum + a[i+(-1)] * 0.11431543;
        sum = sum + a[i+(0)] * 0.11904762;
        sum = sum + a[i+(1)] * 0.11431543;
        sum = sum + a[i+(2)] * 0.10109879;
        sum = sum + a[i+(3)] * 0.0820442;
        sum = sum + a[i+(4)] * 0.060688946;
        sum = sum + a[i+(5)] * 0.04047619;
        sum = sum + a[i+(6)] * 0.023901207;
        sum = sum + a[i+(7)] * 0.012069764;
        sum = sum + a[i+(8)] * 0.0047872453;
        sum = sum + a[i+(9)] * 0.0010944168;
        sum = sum + a[i+(10)] * -1.6521176e-18;
    }
    ret[i] = sum;
}

kernel void bartlett5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.0;
        sum = sum + a[i+(-1)] * 0.25;
        sum = sum + a[i+(0)] * 0.5;
        sum = sum + a[i+(1)] * 0.25;
        sum = sum + a[i+(2)] * 0.0;
    }
    ret[i] = sum;
}

kernel void bartlett9(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-4;
    if ((i > 4) &&  (i < right)) {
        sum = sum + a[i+(-4)] * 0.0;
        sum = sum + a[i+(-3)] * 0.0625;
        sum = sum + a[i+(-2)] * 0.125;
        sum = sum + a[i+(-1)] * 0.1875;
        sum = sum + a[i+(0)] * 0.25;
        sum = sum + a[i+(1)] * 0.1875;
        sum = sum + a[i+(2)] * 0.125;
        sum = sum + a[i+(3)] * 0.0625;
        sum = sum + a[i+(4)] * 0.0;
    }
    ret[i] = sum;
}

kernel void bartlett13(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-6;
    if ((i > 6) &&  (i < right)) {
        sum = sum + a[i+(-6)] * 0.0;
        sum = sum + a[i+(-5)] * 0.027777778;
        sum = sum + a[i+(-4)] * 0.055555556;
        sum = sum + a[i+(-3)] * 0.083333336;
        sum = sum + a[i+(-2)] * 0.11111111;
        sum = sum + a[i+(-1)] * 0.1388889;
        sum = sum + a[i+(0)] * 0.16666667;
        sum = sum + a[i+(1)] * 0.1388889;
        sum = sum + a[i+(2)] * 0.11111111;
        sum = sum + a[i+(3)] * 0.083333336;
        sum = sum + a[i+(4)] * 0.055555556;
        sum = sum + a[i+(5)] * 0.027777778;
        sum = sum + a[i+(6)] * 0.0;
    }
    ret[i] = sum;
}

kernel void bartlett17(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-8;
    if ((i > 8) &&  (i < right)) {
        sum = sum + a[i+(-8)] * 0.0;
        sum = sum + a[i+(-7)] * 0.015625;
        sum = sum + a[i+(-6)] * 0.03125;
        sum = sum + a[i+(-5)] * 0.046875;
        sum = sum + a[i+(-4)] * 0.0625;
        sum = sum + a[i+(-3)] * 0.078125;
        sum = sum + a[i+(-2)] * 0.09375;
        sum = sum + a[i+(-1)] * 0.109375;
        sum = sum + a[i+(0)] * 0.125;
        sum = sum + a[i+(1)] * 0.109375;
        sum = sum + a[i+(2)] * 0.09375;
        sum = sum + a[i+(3)] * 0.078125;
        sum = sum + a[i+(4)] * 0.0625;
        sum = sum + a[i+(5)] * 0.046875;
        sum = sum + a[i+(6)] * 0.03125;
        sum = sum + a[i+(7)] * 0.015625;
        sum = sum + a[i+(8)] * 0.0;
    }
    ret[i] = sum;
}

kernel void box3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * 0.33333334;
        sum = sum + a[i+(0)] * 0.33333334;
        sum = sum + a[i+(1)] * 0.33333334;
    }
    ret[i] = sum;
}

kernel void box5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.2;
        sum = sum + a[i+(-1)] * 0.2;
        sum = sum + a[i+(0)] * 0.2;
        sum = sum + a[i+(1)] * 0.2;
        sum = sum + a[i+(2)] * 0.2;
    }
    ret[i] = sum;
}

kernel void box7(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * 0.14285715;
        sum = sum + a[i+(-2)] * 0.14285715;
        sum = sum + a[i+(-1)] * 0.14285715;
        sum = sum + a[i+(0)] * 0.14285715;
        sum = sum + a[i+(1)] * 0.14285715;
        sum = sum + a[i+(2)] * 0.14285715;
        sum = sum + a[i+(3)] * 0.14285715;
    }
    ret[i] = sum;
}

kernel void box9(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-4;
    if ((i > 4) &&  (i < right)) {
        sum = sum + a[i+(-4)] * 0.11111111;
        sum = sum + a[i+(-3)] * 0.11111111;
        sum = sum + a[i+(-2)] * 0.11111111;
        sum = sum + a[i+(-1)] * 0.11111111;
        sum = sum + a[i+(0)] * 0.11111111;
        sum = sum + a[i+(1)] * 0.11111111;
        sum = sum + a[i+(2)] * 0.11111111;
        sum = sum + a[i+(3)] * 0.11111111;
        sum = sum + a[i+(4)] * 0.11111111;
    }
    ret[i] = sum;
}

kernel void box11(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-5;
    if ((i > 5) &&  (i < right)) {
        sum = sum + a[i+(-5)] * 0.09090909;
        sum = sum + a[i+(-4)] * 0.09090909;
        sum = sum + a[i+(-3)] * 0.09090909;
        sum = sum + a[i+(-2)] * 0.09090909;
        sum = sum + a[i+(-1)] * 0.09090909;
        sum = sum + a[i+(0)] * 0.09090909;
        sum = sum + a[i+(1)] * 0.09090909;
        sum = sum + a[i+(2)] * 0.09090909;
        sum = sum + a[i+(3)] * 0.09090909;
        sum = sum + a[i+(4)] * 0.09090909;
        sum = sum + a[i+(5)] * 0.09090909;
    }
    ret[i] = sum;
}

kernel void box15(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-7;
    if ((i > 7) &&  (i < right)) {
        sum = sum + a[i+(-7)] * 0.06666667;
        sum = sum + a[i+(-6)] * 0.06666667;
        sum = sum + a[i+(-5)] * 0.06666667;
        sum = sum + a[i+(-4)] * 0.06666667;
        sum = sum + a[i+(-3)] * 0.06666667;
        sum = sum + a[i+(-2)] * 0.06666667;
        sum = sum + a[i+(-1)] * 0.06666667;
        sum = sum + a[i+(0)] * 0.06666667;
        sum = sum + a[i+(1)] * 0.06666667;
        sum = sum + a[i+(2)] * 0.06666667;
        sum = sum + a[i+(3)] * 0.06666667;
        sum = sum + a[i+(4)] * 0.06666667;
        sum = sum + a[i+(5)] * 0.06666667;
        sum = sum + a[i+(6)] * 0.06666667;
        sum = sum + a[i+(7)] * 0.06666667;
    }
    ret[i] = sum;
}

kernel void box21(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-10;
    if ((i > 10) &&  (i < right)) {
        sum = sum + a[i+(-10)] * 0.04761905;
        sum = sum + a[i+(-9)] * 0.04761905;
        sum = sum + a[i+(-8)] * 0.04761905;
        sum = sum + a[i+(-7)] * 0.04761905;
        sum = sum + a[i+(-6)] * 0.04761905;
        sum = sum + a[i+(-5)] * 0.04761905;
        sum = sum + a[i+(-4)] * 0.04761905;
        sum = sum + a[i+(-3)] * 0.04761905;
        sum = sum + a[i+(-2)] * 0.04761905;
        sum = sum + a[i+(-1)] * 0.04761905;
        sum = sum + a[i+(0)] * 0.04761905;
        sum = sum + a[i+(1)] * 0.04761905;
        sum = sum + a[i+(2)] * 0.04761905;
        sum = sum + a[i+(3)] * 0.04761905;
        sum = sum + a[i+(4)] * 0.04761905;
        sum = sum + a[i+(5)] * 0.04761905;
        sum = sum + a[i+(6)] * 0.04761905;
        sum = sum + a[i+(7)] * 0.04761905;
        sum = sum + a[i+(8)] * 0.04761905;
        sum = sum + a[i+(9)] * 0.04761905;
        sum = sum + a[i+(10)] * 0.04761905;
    }
    ret[i] = sum;
}

kernel void triangle3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * 0.0;
        sum = sum + a[i+(0)] * 1.0;
        sum = sum + a[i+(1)] * 0.0;
    }
    ret[i] = sum;
}

kernel void triangle5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.0;
        sum = sum + a[i+(-1)] * 0.25;
        sum = sum + a[i+(0)] * 0.5;
        sum = sum + a[i+(1)] * 0.25;
        sum = sum + a[i+(2)] * 0.0;
    }
    ret[i] = sum;
}

kernel void triangle7(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * 0.0;
        sum = sum + a[i+(-2)] * 0.11111111;
        sum = sum + a[i+(-1)] * 0.22222222;
        sum = sum + a[i+(0)] * 0.33333334;
        sum = sum + a[i+(1)] * 0.22222222;
        sum = sum + a[i+(2)] * 0.11111111;
        sum = sum + a[i+(3)] * 0.0;
    }
    ret[i] = sum;
}

kernel void triangle9(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-4;
    if ((i > 4) &&  (i < right)) {
        sum = sum + a[i+(-4)] * 0.0;
        sum = sum + a[i+(-3)] * 0.0625;
        sum = sum + a[i+(-2)] * 0.125;
        sum = sum + a[i+(-1)] * 0.1875;
        sum = sum + a[i+(0)] * 0.25;
        sum = sum + a[i+(1)] * 0.1875;
        sum = sum + a[i+(2)] * 0.125;
        sum = sum + a[i+(3)] * 0.0625;
        sum = sum + a[i+(4)] * 0.0;
    }
    ret[i] = sum;
}

kernel void triangle11(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-5;
    if ((i > 5) &&  (i < right)) {
        sum = sum + a[i+(-5)] * 0.0;
        sum = sum + a[i+(-4)] * 0.04;
        sum = sum + a[i+(-3)] * 0.08;
        sum = sum + a[i+(-2)] * 0.12;
        sum = sum + a[i+(-1)] * 0.16;
        sum = sum + a[i+(0)] * 0.2;
        sum = sum + a[i+(1)] * 0.16;
        sum = sum + a[i+(2)] * 0.12;
        sum = sum + a[i+(3)] * 0.08;
        sum = sum + a[i+(4)] * 0.04;
        sum = sum + a[i+(5)] * 0.0;
    }
    ret[i] = sum;
}

kernel void binomial3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * 0.25;
        sum = sum + a[i+(0)] * 0.5;
        sum = sum + a[i+(1)] * 0.25;
    }
    ret[i] = sum;
}

kernel void binomial5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.0625;
        sum = sum + a[i+(-1)] * 0.25;
        sum = sum + a[i+(0)] * 0.375;
        sum = sum + a[i+(1)] * 0.25;
        sum = sum + a[i+(2)] * 0.0625;
    }
    ret[i] = sum;
}

kernel void binomial7(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * 0.015625;
        sum = sum + a[i+(-2)] * 0.09375;
        sum = sum + a[i+(-1)] * 0.234375;
        sum = sum + a[i+(0)] * 0.3125;
        sum = sum + a[i+(1)] * 0.234375;
        sum = sum + a[i+(2)] * 0.09375;
        sum = sum + a[i+(3)] * 0.015625;
    }
    ret[i] = sum;
}

kernel void binomial9(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-4;
    if ((i > 4) &&  (i < right)) {
        sum = sum + a[i+(-4)] * 0.00390625;
        sum = sum + a[i+(-3)] * 0.03125;
        sum = sum + a[i+(-2)] * 0.109375;
        sum = sum + a[i+(-1)] * 0.21875;
        sum = sum + a[i+(0)] * 0.2734375;
        sum = sum + a[i+(1)] * 0.21875;
        sum = sum + a[i+(2)] * 0.109375;
        sum = sum + a[i+(3)] * 0.03125;
        sum = sum + a[i+(4)] * 0.00390625;
    }
    ret[i] = sum;
}

kernel void deriv13(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * -0.5;
        sum = sum + a[i+(0)] * 0.0;
        sum = sum + a[i+(1)] * 0.5;
    }
    ret[i] = sum;
}

kernel void deriv15(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * 0.083333336;
        sum = sum + a[i+(-1)] * -0.6666667;
        sum = sum + a[i+(0)] * 0.0;
        sum = sum + a[i+(1)] * 0.6666667;
        sum = sum + a[i+(2)] * -0.083333336;
    }
    ret[i] = sum;
}

kernel void deriv17(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * -0.016666668;
        sum = sum + a[i+(-2)] * 0.15;
        sum = sum + a[i+(-1)] * -0.75;
        sum = sum + a[i+(0)] * 0.0;
        sum = sum + a[i+(1)] * 0.75;
        sum = sum + a[i+(2)] * -0.15;
        sum = sum + a[i+(3)] * 0.016666668;
    }
    ret[i] = sum;
}

kernel void deriv23(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * 1.0;
        sum = sum + a[i+(0)] * -2.0;
        sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
}

kernel void deriv25(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * -0.083333336;
        sum = sum + a[i+(-1)] * 1.3333334;
        sum = sum + a[i+(0)] * -2.5;
        sum = sum + a[i+(1)] * 1.3333334;
        sum = sum + a[i+(2)] * -0.083333336;
    }
    ret[i] = sum;
}

kernel void deriv27(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * 0.011111111;
        sum = sum + a[i+(-2)] * -0.15;
        sum = sum + a[i+(-1)] * 1.5;
        sum = sum + a[i+(0)] * -2.7222223;
        sum = sum + a[i+(1)] * 1.5;
        sum = sum + a[i+(2)] * -0.15;
        sum = sum + a[i+(3)] * 0.011111111;
    }
    ret[i] = sum;
}

kernel void sobel3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * -1.0;
        sum = sum + a[i+(0)] * 0.0;
        sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
}

kernel void prewitt3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * -1.0;
        sum = sum + a[i+(0)] * 0.0;
        sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
}

kernel void sharpen3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * -1.0;
        sum = sum + a[i+(0)] * 3.0;
        sum = sum + a[i+(1)] * -1.0;
    }
    ret[i] = sum;
}

kernel void sharpen5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * -1.0;
        sum = sum + a[i+(-1)] * -1.0;
        sum = sum + a[i+(0)] * 5.0;
        sum = sum + a[i+(1)] * -1.0;
        sum = sum + a[i+(2)] * -1.0;
    }
    ret[i] = sum;
}

kernel void unsharp3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * -0.33333334;
        sum = sum + a[i+(0)] * 1.6666666;
        sum = sum + a[i+(1)] * -0.33333334;
    }
    ret[i] = sum;
}

kernel void unsharp5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * -0.2;
        sum = sum + a[i+(-1)] * -0.2;
        sum = sum + a[i+(0)] * 1.8;
        sum = sum + a[i+(1)] * -0.2;
        sum = sum + a[i+(2)] * -0.2;
    }
    ret[i] = sum;
}

kernel void unsharp7(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * -0.14285715;
        sum = sum + a[i+(-2)] * -0.14285715;
        sum = sum + a[i+(-1)] * -0.14285715;
        sum = sum + a[i+(0)] * 1.8571428;
        sum = sum + a[i+(1)] * -0.14285715;
        sum = sum + a[i+(2)] * -0.14285715;
        sum = sum + a[i+(3)] * -0.14285715;
    }
    ret[i] = sum;
}

kernel void dog5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * -0.25;
        sum = sum + a[i+(-1)] * -0.25;
        sum = sum + a[i+(0)] * 1.0;
        sum = sum + a[i+(1)] * -0.25;
        sum = sum + a[i+(2)] * -0.25;
    }
    ret[i] = sum;
}

kernel void dog7(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * -0.16666667;
        sum = sum + a[i+(-2)] * -0.16666667;
        sum = sum + a[i+(-1)] * -0.16666667;
        sum = sum + a[i+(0)] * 1.0;
        sum = sum + a[i+(1)] * -0.16666667;
        sum = sum + a[i+(2)] * -0.16666667;
        sum = sum + a[i+(3)] * -0.16666667;
    }
    ret[i] = sum;
}

kernel void dog9(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-4;
    if ((i > 4) &&  (i < right)) {
        sum = sum + a[i+(-4)] * -0.125;
        sum = sum + a[i+(-3)] * -0.125;
        sum = sum + a[i+(-2)] * -0.125;
        sum = sum + a[i+(-1)] * -0.125;
        sum = sum + a[i+(0)] * 1.0;
        sum = sum + a[i+(1)] * -0.125;
        sum = sum + a[i+(2)] * -0.125;
        sum = sum + a[i+(3)] * -0.125;
        sum = sum + a[i+(4)] * -0.125;
    }
    ret[i] = sum;
}

kernel void dog11(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-5;
    if ((i > 5) &&  (i < right)) {
        sum = sum + a[i+(-5)] * -0.1;
        sum = sum + a[i+(-4)] * -0.1;
        sum = sum + a[i+(-3)] * -0.1;
        sum = sum + a[i+(-2)] * -0.1;
        sum = sum + a[i+(-1)] * -0.1;
        sum = sum + a[i+(0)] * 1.0;
        sum = sum + a[i+(1)] * -0.1;
        sum = sum + a[i+(2)] * -0.1;
        sum = sum + a[i+(3)] * -0.1;
        sum = sum + a[i+(4)] * -0.1;
        sum = sum + a[i+(5)] * -0.1;
    }
    ret[i] = sum;
}

kernel void edge3(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-1;
    if ((i > 1) &&  (i < right)) {
        sum = sum + a[i+(-1)] * -1.0;
        sum = sum + a[i+(0)] * 0.0;
        sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
}

kernel void edge5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * -1.0;
        sum = sum + a[i+(-1)] * -0.5;
        sum = sum + a[i+(0)] * 0.0;
        sum = sum + a[i+(1)] * 0.5;
        sum = sum + a[i+(2)] * 1.0;
    }
    ret[i] = sum;
}

kernel void savgol5(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-2;
    if ((i > 2) &&  (i < right)) {
        sum = sum + a[i+(-2)] * -0.06382979;
        sum = sum + a[i+(-1)] * 0.25531915;
        sum = sum + a[i+(0)] * 0.3617021;
        sum = sum + a[i+(1)] * 0.25531915;
        sum = sum + a[i+(2)] * -0.06382979;
    }
    ret[i] = sum;
}

kernel void savgol7(constant int &width [[ buffer(0) ]],
                    device float* a [[ buffer(1) ]],
                    device float* ret [[ buffer(2) ]],
                    uint i [[ thread_position_in_grid ]])
{
    float sum = 0.0f;
    size_t right = width-3;
    if ((i > 3) &&  (i < right)) {
        sum = sum + a[i+(-3)] * -0.06896552;
        sum = sum + a[i+(-2)] * 0.10344828;
        sum = sum + a[i+(-1)] * 0.20689656;
        sum = sum + a[i+(0)] * 0.2413793;
        sum = sum + a[i+(1)] * 0.20689656;
        sum = sum + a[i+(2)] * 0.10344828;
        sum = sum + a[i+(3)] * -0.06896552;
    }
    ret[i] = sum;
}
