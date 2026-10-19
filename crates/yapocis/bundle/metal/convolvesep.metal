/* Separable convolution kernels (Metal), names and constants determined at compile time */

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


/* Helpful macros and in-line functions (MSL-safe) */

// Address macros (column-major, matching OCL): index = x*height + y
#define ADDRESS(xx,yy) (((xx)*height)+(yy))

#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)

#define SWAPINT(xxx,yyy) {int tmp = xxx; xxx=yyy; yyy=tmp;}
#define SWAPFLOAT(xxx,yyy) {float tmp = xxx; xxx=yyy; yyy=tmp;}

#define ABS(x) (((x) < 0) ? -(x) : (x))

#ifdef DXDY2
static inline float dxdy2(float2 a, float2 b) {
    float2 dxdy =  (a-b)*(a-b);
    return dxdy.x + dxdy.y;
}
#endif

#ifdef DOT3
static inline float dot3(float3 a, float3 b) {
   return (float) (a.x*b.x + a.y*b.y + a.z*b.z);
}
#endif

#ifdef VXM3
static inline float3 vxm3(float3 v, float3 a, float3 b, float3 c) {
    float3 result = float3(dot3(v,a), dot3(v,b), dot3(v,c));
    return result;
}
#endif

#define ANYISLESS(a,b) (a.x < b.x || a.y < b.y)
#define ANYISGTE(a,b) (a.x >= b.x || a.y >= b.y)



kernel void gauss9(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(4*height) < i) && (i < (n-4*height))){
       // Horizontal convolution
        sum = sum + a[i+(-4*height)] * 0.00013383062461474178;
        sum = sum + a[i+(-3*height)] * 0.004431861620031266;
        sum = sum + a[i+(-2*height)] * 0.053991127420704416;
        sum = sum + a[i+(-1*height)] * 0.24197144565660075;
        sum = sum + a[i+(0*height)] * 0.39894346935609776;
        sum = sum + a[i+(1*height)] * 0.24197144565660075;
        sum = sum + a[i+(2*height)] * 0.053991127420704416;
        sum = sum + a[i+(3*height)] * 0.004431861620031266;
        sum = sum + a[i+(4*height)] * 0.00013383062461474178;
    }
    if (vertical && (4 < i) && (i < (n-4))) {
       // Vertical convolution
        sum = sum + a[i+(-4)] * 0.00013383062461474178;
        sum = sum + a[i+(-3)] * 0.004431861620031266;
        sum = sum + a[i+(-2)] * 0.053991127420704416;
        sum = sum + a[i+(-1)] * 0.24197144565660075;
        sum = sum + a[i+(0)] * 0.39894346935609776;
        sum = sum + a[i+(1)] * 0.24197144565660075;
        sum = sum + a[i+(2)] * 0.053991127420704416;
        sum = sum + a[i+(3)] * 0.004431861620031266;
        sum = sum + a[i+(4)] * 0.00013383062461474178;
    }
    ret[i] = sum;
}

kernel void gauss9_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 4) && (x<width-4) && (y >= 4) && (y < height-4);
    //int inside = (((size_t)(4*height) < i) && (i < (n-4*height)));
    sum += inside ? a[i+(-4*height)] * 0.00013383062461474178 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.004431861620031266 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.053991127420704416 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.24197144565660075 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.39894346935609776 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.24197144565660075 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.053991127420704416 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.004431861620031266 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.00013383062461474178 : 0.0f;
    ret[i] = sum;
}

kernel void gauss9_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 4) && (x<width-4) && (y >= 4) && (y < height-4);
    //int inside = (4 < i) && (i < (n-4));
    sum +=  inside ?  a[i+(-4)] * 0.00013383062461474178 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.004431861620031266 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.053991127420704416 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.24197144565660075 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.39894346935609776 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.24197144565660075 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.053991127420704416 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.004431861620031266 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.00013383062461474178 : 0.0f;

    ret[i] = sum;
}

kernel void gauss13(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(6*height) < i) && (i < (n-6*height))){
       // Horizontal convolution
        sum = sum + a[i+(-6*height)] * 0.00022038039379869594;
        sum = sum + a[i+(-5*height)] * 0.0018889806349479833;
        sum = sum + a[i+(-4*height)] * 0.010955590095311843;
        sum = sum + a[i+(-3*height)] * 0.04299300108511958;
        sum = sum + a[i+(-2*height)] * 0.11415987029318388;
        sum = sum + a[i+(-1*height)] * 0.20510813686914908;
        sum = sum + a[i+(0*height)] * 0.24934808125697794;
        sum = sum + a[i+(1*height)] * 0.20510813686914908;
        sum = sum + a[i+(2*height)] * 0.11415987029318388;
        sum = sum + a[i+(3*height)] * 0.04299300108511958;
        sum = sum + a[i+(4*height)] * 0.010955590095311843;
        sum = sum + a[i+(5*height)] * 0.0018889806349479833;
        sum = sum + a[i+(6*height)] * 0.00022038039379869594;
    }
    if (vertical && (6 < i) && (i < (n-6))) {
       // Vertical convolution
        sum = sum + a[i+(-6)] * 0.00022038039379869594;
        sum = sum + a[i+(-5)] * 0.0018889806349479833;
        sum = sum + a[i+(-4)] * 0.010955590095311843;
        sum = sum + a[i+(-3)] * 0.04299300108511958;
        sum = sum + a[i+(-2)] * 0.11415987029318388;
        sum = sum + a[i+(-1)] * 0.20510813686914908;
        sum = sum + a[i+(0)] * 0.24934808125697794;
        sum = sum + a[i+(1)] * 0.20510813686914908;
        sum = sum + a[i+(2)] * 0.11415987029318388;
        sum = sum + a[i+(3)] * 0.04299300108511958;
        sum = sum + a[i+(4)] * 0.010955590095311843;
        sum = sum + a[i+(5)] * 0.0018889806349479833;
        sum = sum + a[i+(6)] * 0.00022038039379869594;
    }
    ret[i] = sum;
}

kernel void gauss13_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 6) && (x<width-6) && (y >= 6) && (y < height-6);
    //int inside = (((size_t)(6*height) < i) && (i < (n-6*height)));
    sum += inside ? a[i+(-6*height)] * 0.00022038039379869594 : 0.0f;
    sum += inside ? a[i+(-5*height)] * 0.0018889806349479833 : 0.0f;
    sum += inside ? a[i+(-4*height)] * 0.010955590095311843 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.04299300108511958 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.11415987029318388 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.20510813686914908 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.24934808125697794 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.20510813686914908 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.11415987029318388 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.04299300108511958 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.010955590095311843 : 0.0f;
    sum += inside ? a[i+(5*height)] * 0.0018889806349479833 : 0.0f;
    sum += inside ? a[i+(6*height)] * 0.00022038039379869594 : 0.0f;
    ret[i] = sum;
}

kernel void gauss13_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 6) && (x<width-6) && (y >= 6) && (y < height-6);
    //int inside = (6 < i) && (i < (n-6));
    sum +=  inside ?  a[i+(-6)] * 0.00022038039379869594 : 0.0f;
    sum +=  inside ?  a[i+(-5)] * 0.0018889806349479833 : 0.0f;
    sum +=  inside ?  a[i+(-4)] * 0.010955590095311843 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.04299300108511958 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.11415987029318388 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.20510813686914908 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.24934808125697794 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.20510813686914908 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.11415987029318388 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.04299300108511958 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.010955590095311843 : 0.0f;
    sum +=  inside ?  a[i+(5)] * 0.0018889806349479833 : 0.0f;
    sum +=  inside ?  a[i+(6)] * 0.00022038039379869594 : 0.0f;

    ret[i] = sum;
}

kernel void gauss19(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(9*height) < i) && (i < (n-9*height))){
       // Horizontal convolution
        sum = sum + a[i+(-9*height)] * 0.0003227710854991425;
        sum = sum + a[i+(-8*height)] * 0.0011807917046821662;
        sum = sum + a[i+(-7*height)] * 0.0037083770547489477;
        sum = sum + a[i+(-6*height)] * 0.009998305397878104;
        sum = sum + a[i+(-5*height)] * 0.02314199695463347;
        sum = sum + a[i+(-4*height)] * 0.045984046663448144;
        sum = sum + a[i+(-3*height)] * 0.0784414229561743;
        sum = sum + a[i+(-2*height)] * 0.1148723909594504;
        sum = sum + a[i+(-1*height)] * 0.14441682038843295;
        sum = sum + a[i+(0*height)] * 0.15586615367010456;
        sum = sum + a[i+(1*height)] * 0.14441682038843295;
        sum = sum + a[i+(2*height)] * 0.1148723909594504;
        sum = sum + a[i+(3*height)] * 0.0784414229561743;
        sum = sum + a[i+(4*height)] * 0.045984046663448144;
        sum = sum + a[i+(5*height)] * 0.02314199695463347;
        sum = sum + a[i+(6*height)] * 0.009998305397878104;
        sum = sum + a[i+(7*height)] * 0.0037083770547489477;
        sum = sum + a[i+(8*height)] * 0.0011807917046821662;
        sum = sum + a[i+(9*height)] * 0.0003227710854991425;
    }
    if (vertical && (9 < i) && (i < (n-9))) {
       // Vertical convolution
        sum = sum + a[i+(-9)] * 0.0003227710854991425;
        sum = sum + a[i+(-8)] * 0.0011807917046821662;
        sum = sum + a[i+(-7)] * 0.0037083770547489477;
        sum = sum + a[i+(-6)] * 0.009998305397878104;
        sum = sum + a[i+(-5)] * 0.02314199695463347;
        sum = sum + a[i+(-4)] * 0.045984046663448144;
        sum = sum + a[i+(-3)] * 0.0784414229561743;
        sum = sum + a[i+(-2)] * 0.1148723909594504;
        sum = sum + a[i+(-1)] * 0.14441682038843295;
        sum = sum + a[i+(0)] * 0.15586615367010456;
        sum = sum + a[i+(1)] * 0.14441682038843295;
        sum = sum + a[i+(2)] * 0.1148723909594504;
        sum = sum + a[i+(3)] * 0.0784414229561743;
        sum = sum + a[i+(4)] * 0.045984046663448144;
        sum = sum + a[i+(5)] * 0.02314199695463347;
        sum = sum + a[i+(6)] * 0.009998305397878104;
        sum = sum + a[i+(7)] * 0.0037083770547489477;
        sum = sum + a[i+(8)] * 0.0011807917046821662;
        sum = sum + a[i+(9)] * 0.0003227710854991425;
    }
    ret[i] = sum;
}

kernel void gauss19_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 9) && (x<width-9) && (y >= 9) && (y < height-9);
    //int inside = (((size_t)(9*height) < i) && (i < (n-9*height)));
    sum += inside ? a[i+(-9*height)] * 0.0003227710854991425 : 0.0f;
    sum += inside ? a[i+(-8*height)] * 0.0011807917046821662 : 0.0f;
    sum += inside ? a[i+(-7*height)] * 0.0037083770547489477 : 0.0f;
    sum += inside ? a[i+(-6*height)] * 0.009998305397878104 : 0.0f;
    sum += inside ? a[i+(-5*height)] * 0.02314199695463347 : 0.0f;
    sum += inside ? a[i+(-4*height)] * 0.045984046663448144 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.0784414229561743 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.1148723909594504 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.14441682038843295 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.15586615367010456 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.14441682038843295 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.1148723909594504 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.0784414229561743 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.045984046663448144 : 0.0f;
    sum += inside ? a[i+(5*height)] * 0.02314199695463347 : 0.0f;
    sum += inside ? a[i+(6*height)] * 0.009998305397878104 : 0.0f;
    sum += inside ? a[i+(7*height)] * 0.0037083770547489477 : 0.0f;
    sum += inside ? a[i+(8*height)] * 0.0011807917046821662 : 0.0f;
    sum += inside ? a[i+(9*height)] * 0.0003227710854991425 : 0.0f;
    ret[i] = sum;
}

kernel void gauss19_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 9) && (x<width-9) && (y >= 9) && (y < height-9);
    //int inside = (9 < i) && (i < (n-9));
    sum +=  inside ?  a[i+(-9)] * 0.0003227710854991425 : 0.0f;
    sum +=  inside ?  a[i+(-8)] * 0.0011807917046821662 : 0.0f;
    sum +=  inside ?  a[i+(-7)] * 0.0037083770547489477 : 0.0f;
    sum +=  inside ?  a[i+(-6)] * 0.009998305397878104 : 0.0f;
    sum +=  inside ?  a[i+(-5)] * 0.02314199695463347 : 0.0f;
    sum +=  inside ?  a[i+(-4)] * 0.045984046663448144 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.0784414229561743 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.1148723909594504 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.14441682038843295 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.15586615367010456 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.14441682038843295 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.1148723909594504 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.0784414229561743 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.045984046663448144 : 0.0f;
    sum +=  inside ?  a[i+(5)] * 0.02314199695463347 : 0.0f;
    sum +=  inside ?  a[i+(6)] * 0.009998305397878104 : 0.0f;
    sum +=  inside ?  a[i+(7)] * 0.0037083770547489477 : 0.0f;
    sum +=  inside ?  a[i+(8)] * 0.0011807917046821662 : 0.0f;
    sum +=  inside ?  a[i+(9)] * 0.0003227710854991425 : 0.0f;

    ret[i] = sum;
}

kernel void gauss27(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(13*height) < i) && (i < (n-13*height))){
       // Horizontal convolution
        sum = sum + a[i+(-13*height)] * 0.0006332859052814378;
        sum = sum + a[i+(-12*height)] * 0.0013340571237718665;
        sum = sum + a[i+(-11*height)] * 0.0026476651678995567;
        sum = sum + a[i+(-10*height)] * 0.004950689961621483;
        sum = sum + a[i+(-9*height)] * 0.008721324149676133;
        sum = sum + a[i+(-8*height)] * 0.01447481957823061;
        sum = sum + a[i+(-7*height)] * 0.022633823326070935;
        sum = sum + a[i+(-6*height)] * 0.0333439246338091;
        sum = sum + a[i+(-5*height)] * 0.04627959411715879;
        sum = sum + a[i+(-4*height)] * 0.06051686179859002;
        sum = sum + a[i+(-3*height)] * 0.0745550881457432;
        sum = sum + a[i+(-2*height)] * 0.0865350818530791;
        sum = sum + a[i+(-1*height)] * 0.09462832685372734;
        sum = sum + a[i+(0*height)] * 0.09749091477068102;
        sum = sum + a[i+(1*height)] * 0.09462832685372734;
        sum = sum + a[i+(2*height)] * 0.0865350818530791;
        sum = sum + a[i+(3*height)] * 0.0745550881457432;
        sum = sum + a[i+(4*height)] * 0.06051686179859002;
        sum = sum + a[i+(5*height)] * 0.04627959411715879;
        sum = sum + a[i+(6*height)] * 0.0333439246338091;
        sum = sum + a[i+(7*height)] * 0.022633823326070935;
        sum = sum + a[i+(8*height)] * 0.01447481957823061;
        sum = sum + a[i+(9*height)] * 0.008721324149676133;
        sum = sum + a[i+(10*height)] * 0.004950689961621483;
        sum = sum + a[i+(11*height)] * 0.0026476651678995567;
        sum = sum + a[i+(12*height)] * 0.0013340571237718665;
        sum = sum + a[i+(13*height)] * 0.0006332859052814378;
    }
    if (vertical && (13 < i) && (i < (n-13))) {
       // Vertical convolution
        sum = sum + a[i+(-13)] * 0.0006332859052814378;
        sum = sum + a[i+(-12)] * 0.0013340571237718665;
        sum = sum + a[i+(-11)] * 0.0026476651678995567;
        sum = sum + a[i+(-10)] * 0.004950689961621483;
        sum = sum + a[i+(-9)] * 0.008721324149676133;
        sum = sum + a[i+(-8)] * 0.01447481957823061;
        sum = sum + a[i+(-7)] * 0.022633823326070935;
        sum = sum + a[i+(-6)] * 0.0333439246338091;
        sum = sum + a[i+(-5)] * 0.04627959411715879;
        sum = sum + a[i+(-4)] * 0.06051686179859002;
        sum = sum + a[i+(-3)] * 0.0745550881457432;
        sum = sum + a[i+(-2)] * 0.0865350818530791;
        sum = sum + a[i+(-1)] * 0.09462832685372734;
        sum = sum + a[i+(0)] * 0.09749091477068102;
        sum = sum + a[i+(1)] * 0.09462832685372734;
        sum = sum + a[i+(2)] * 0.0865350818530791;
        sum = sum + a[i+(3)] * 0.0745550881457432;
        sum = sum + a[i+(4)] * 0.06051686179859002;
        sum = sum + a[i+(5)] * 0.04627959411715879;
        sum = sum + a[i+(6)] * 0.0333439246338091;
        sum = sum + a[i+(7)] * 0.022633823326070935;
        sum = sum + a[i+(8)] * 0.01447481957823061;
        sum = sum + a[i+(9)] * 0.008721324149676133;
        sum = sum + a[i+(10)] * 0.004950689961621483;
        sum = sum + a[i+(11)] * 0.0026476651678995567;
        sum = sum + a[i+(12)] * 0.0013340571237718665;
        sum = sum + a[i+(13)] * 0.0006332859052814378;
    }
    ret[i] = sum;
}

kernel void gauss27_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 13) && (x<width-13) && (y >= 13) && (y < height-13);
    //int inside = (((size_t)(13*height) < i) && (i < (n-13*height)));
    sum += inside ? a[i+(-13*height)] * 0.0006332859052814378 : 0.0f;
    sum += inside ? a[i+(-12*height)] * 0.0013340571237718665 : 0.0f;
    sum += inside ? a[i+(-11*height)] * 0.0026476651678995567 : 0.0f;
    sum += inside ? a[i+(-10*height)] * 0.004950689961621483 : 0.0f;
    sum += inside ? a[i+(-9*height)] * 0.008721324149676133 : 0.0f;
    sum += inside ? a[i+(-8*height)] * 0.01447481957823061 : 0.0f;
    sum += inside ? a[i+(-7*height)] * 0.022633823326070935 : 0.0f;
    sum += inside ? a[i+(-6*height)] * 0.0333439246338091 : 0.0f;
    sum += inside ? a[i+(-5*height)] * 0.04627959411715879 : 0.0f;
    sum += inside ? a[i+(-4*height)] * 0.06051686179859002 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.0745550881457432 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.0865350818530791 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.09462832685372734 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.09749091477068102 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.09462832685372734 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.0865350818530791 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.0745550881457432 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.06051686179859002 : 0.0f;
    sum += inside ? a[i+(5*height)] * 0.04627959411715879 : 0.0f;
    sum += inside ? a[i+(6*height)] * 0.0333439246338091 : 0.0f;
    sum += inside ? a[i+(7*height)] * 0.022633823326070935 : 0.0f;
    sum += inside ? a[i+(8*height)] * 0.01447481957823061 : 0.0f;
    sum += inside ? a[i+(9*height)] * 0.008721324149676133 : 0.0f;
    sum += inside ? a[i+(10*height)] * 0.004950689961621483 : 0.0f;
    sum += inside ? a[i+(11*height)] * 0.0026476651678995567 : 0.0f;
    sum += inside ? a[i+(12*height)] * 0.0013340571237718665 : 0.0f;
    sum += inside ? a[i+(13*height)] * 0.0006332859052814378 : 0.0f;
    ret[i] = sum;
}

kernel void gauss27_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 13) && (x<width-13) && (y >= 13) && (y < height-13);
    //int inside = (13 < i) && (i < (n-13));
    sum +=  inside ?  a[i+(-13)] * 0.0006332859052814378 : 0.0f;
    sum +=  inside ?  a[i+(-12)] * 0.0013340571237718665 : 0.0f;
    sum +=  inside ?  a[i+(-11)] * 0.0026476651678995567 : 0.0f;
    sum +=  inside ?  a[i+(-10)] * 0.004950689961621483 : 0.0f;
    sum +=  inside ?  a[i+(-9)] * 0.008721324149676133 : 0.0f;
    sum +=  inside ?  a[i+(-8)] * 0.01447481957823061 : 0.0f;
    sum +=  inside ?  a[i+(-7)] * 0.022633823326070935 : 0.0f;
    sum +=  inside ?  a[i+(-6)] * 0.0333439246338091 : 0.0f;
    sum +=  inside ?  a[i+(-5)] * 0.04627959411715879 : 0.0f;
    sum +=  inside ?  a[i+(-4)] * 0.06051686179859002 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.0745550881457432 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.0865350818530791 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.09462832685372734 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.09749091477068102 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.09462832685372734 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.0865350818530791 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.0745550881457432 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.06051686179859002 : 0.0f;
    sum +=  inside ?  a[i+(5)] * 0.04627959411715879 : 0.0f;
    sum +=  inside ?  a[i+(6)] * 0.0333439246338091 : 0.0f;
    sum +=  inside ?  a[i+(7)] * 0.022633823326070935 : 0.0f;
    sum +=  inside ?  a[i+(8)] * 0.01447481957823061 : 0.0f;
    sum +=  inside ?  a[i+(9)] * 0.008721324149676133 : 0.0f;
    sum +=  inside ?  a[i+(10)] * 0.004950689961621483 : 0.0f;
    sum +=  inside ?  a[i+(11)] * 0.0026476651678995567 : 0.0f;
    sum +=  inside ?  a[i+(12)] * 0.0013340571237718665 : 0.0f;
    sum +=  inside ?  a[i+(13)] * 0.0006332859052814378 : 0.0f;

    ret[i] = sum;
}

kernel void gauss39(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(19*height) < i) && (i < (n-19*height))){
       // Horizontal convolution
        sum = sum + a[i+(-19*height)] * 0.000913119384660958;
        sum = sum + a[i+(-18*height)] * 0.0014047341175216514;
        sum = sum + a[i+(-17*height)] * 0.002111295354219901;
        sum = sum + a[i+(-16*height)] * 0.003100217384617136;
        sum = sum + a[i+(-15*height)] * 0.004447578233139086;
        sum = sum + a[i+(-14*height)] * 0.00623366326536526;
        sum = sum + a[i+(-13*height)] * 0.008535940410370202;
        sum = sum + a[i+(-12*height)] * 0.011419516182950293;
        sum = sum + a[i+(-11*height)] * 0.014925618183728262;
        sum = sum + a[i+(-10*height)] * 0.019059225329222967;
        sum = sum + a[i+(-9*height)] * 0.023777514497134777;
        sum = sum + a[i+(-8*height)] * 0.028981172997560947;
        sum = sum + a[i+(-7*height)] * 0.03451069910411127;
        sum = sum + a[i+(-6*height)] * 0.04014947295048908;
        sum = sum + a[i+(-5*height)] * 0.045634598474731086;
        sum = sum + a[i+(-4*height)] * 0.05067536813549639;
        sum = sum + a[i+(-3*height)] * 0.054977866789386806;
        sum = sum + a[i+(-2*height)] * 0.05827296963061248;
        sum = sum + a[i+(-1*height)] * 0.06034408551584046;
        sum = sum + a[i+(0*height)] * 0.06105068811768207;
        sum = sum + a[i+(1*height)] * 0.06034408551584046;
        sum = sum + a[i+(2*height)] * 0.05827296963061248;
        sum = sum + a[i+(3*height)] * 0.054977866789386806;
        sum = sum + a[i+(4*height)] * 0.05067536813549639;
        sum = sum + a[i+(5*height)] * 0.045634598474731086;
        sum = sum + a[i+(6*height)] * 0.04014947295048908;
        sum = sum + a[i+(7*height)] * 0.03451069910411127;
        sum = sum + a[i+(8*height)] * 0.028981172997560947;
        sum = sum + a[i+(9*height)] * 0.023777514497134777;
        sum = sum + a[i+(10*height)] * 0.019059225329222967;
        sum = sum + a[i+(11*height)] * 0.014925618183728262;
        sum = sum + a[i+(12*height)] * 0.011419516182950293;
        sum = sum + a[i+(13*height)] * 0.008535940410370202;
        sum = sum + a[i+(14*height)] * 0.00623366326536526;
        sum = sum + a[i+(15*height)] * 0.004447578233139086;
        sum = sum + a[i+(16*height)] * 0.003100217384617136;
        sum = sum + a[i+(17*height)] * 0.002111295354219901;
        sum = sum + a[i+(18*height)] * 0.0014047341175216514;
        sum = sum + a[i+(19*height)] * 0.000913119384660958;
    }
    if (vertical && (19 < i) && (i < (n-19))) {
       // Vertical convolution
        sum = sum + a[i+(-19)] * 0.000913119384660958;
        sum = sum + a[i+(-18)] * 0.0014047341175216514;
        sum = sum + a[i+(-17)] * 0.002111295354219901;
        sum = sum + a[i+(-16)] * 0.003100217384617136;
        sum = sum + a[i+(-15)] * 0.004447578233139086;
        sum = sum + a[i+(-14)] * 0.00623366326536526;
        sum = sum + a[i+(-13)] * 0.008535940410370202;
        sum = sum + a[i+(-12)] * 0.011419516182950293;
        sum = sum + a[i+(-11)] * 0.014925618183728262;
        sum = sum + a[i+(-10)] * 0.019059225329222967;
        sum = sum + a[i+(-9)] * 0.023777514497134777;
        sum = sum + a[i+(-8)] * 0.028981172997560947;
        sum = sum + a[i+(-7)] * 0.03451069910411127;
        sum = sum + a[i+(-6)] * 0.04014947295048908;
        sum = sum + a[i+(-5)] * 0.045634598474731086;
        sum = sum + a[i+(-4)] * 0.05067536813549639;
        sum = sum + a[i+(-3)] * 0.054977866789386806;
        sum = sum + a[i+(-2)] * 0.05827296963061248;
        sum = sum + a[i+(-1)] * 0.06034408551584046;
        sum = sum + a[i+(0)] * 0.06105068811768207;
        sum = sum + a[i+(1)] * 0.06034408551584046;
        sum = sum + a[i+(2)] * 0.05827296963061248;
        sum = sum + a[i+(3)] * 0.054977866789386806;
        sum = sum + a[i+(4)] * 0.05067536813549639;
        sum = sum + a[i+(5)] * 0.045634598474731086;
        sum = sum + a[i+(6)] * 0.04014947295048908;
        sum = sum + a[i+(7)] * 0.03451069910411127;
        sum = sum + a[i+(8)] * 0.028981172997560947;
        sum = sum + a[i+(9)] * 0.023777514497134777;
        sum = sum + a[i+(10)] * 0.019059225329222967;
        sum = sum + a[i+(11)] * 0.014925618183728262;
        sum = sum + a[i+(12)] * 0.011419516182950293;
        sum = sum + a[i+(13)] * 0.008535940410370202;
        sum = sum + a[i+(14)] * 0.00623366326536526;
        sum = sum + a[i+(15)] * 0.004447578233139086;
        sum = sum + a[i+(16)] * 0.003100217384617136;
        sum = sum + a[i+(17)] * 0.002111295354219901;
        sum = sum + a[i+(18)] * 0.0014047341175216514;
        sum = sum + a[i+(19)] * 0.000913119384660958;
    }
    ret[i] = sum;
}

kernel void gauss39_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 19) && (x<width-19) && (y >= 19) && (y < height-19);
    //int inside = (((size_t)(19*height) < i) && (i < (n-19*height)));
    sum += inside ? a[i+(-19*height)] * 0.000913119384660958 : 0.0f;
    sum += inside ? a[i+(-18*height)] * 0.0014047341175216514 : 0.0f;
    sum += inside ? a[i+(-17*height)] * 0.002111295354219901 : 0.0f;
    sum += inside ? a[i+(-16*height)] * 0.003100217384617136 : 0.0f;
    sum += inside ? a[i+(-15*height)] * 0.004447578233139086 : 0.0f;
    sum += inside ? a[i+(-14*height)] * 0.00623366326536526 : 0.0f;
    sum += inside ? a[i+(-13*height)] * 0.008535940410370202 : 0.0f;
    sum += inside ? a[i+(-12*height)] * 0.011419516182950293 : 0.0f;
    sum += inside ? a[i+(-11*height)] * 0.014925618183728262 : 0.0f;
    sum += inside ? a[i+(-10*height)] * 0.019059225329222967 : 0.0f;
    sum += inside ? a[i+(-9*height)] * 0.023777514497134777 : 0.0f;
    sum += inside ? a[i+(-8*height)] * 0.028981172997560947 : 0.0f;
    sum += inside ? a[i+(-7*height)] * 0.03451069910411127 : 0.0f;
    sum += inside ? a[i+(-6*height)] * 0.04014947295048908 : 0.0f;
    sum += inside ? a[i+(-5*height)] * 0.045634598474731086 : 0.0f;
    sum += inside ? a[i+(-4*height)] * 0.05067536813549639 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.054977866789386806 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.05827296963061248 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.06034408551584046 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.06105068811768207 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.06034408551584046 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.05827296963061248 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.054977866789386806 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.05067536813549639 : 0.0f;
    sum += inside ? a[i+(5*height)] * 0.045634598474731086 : 0.0f;
    sum += inside ? a[i+(6*height)] * 0.04014947295048908 : 0.0f;
    sum += inside ? a[i+(7*height)] * 0.03451069910411127 : 0.0f;
    sum += inside ? a[i+(8*height)] * 0.028981172997560947 : 0.0f;
    sum += inside ? a[i+(9*height)] * 0.023777514497134777 : 0.0f;
    sum += inside ? a[i+(10*height)] * 0.019059225329222967 : 0.0f;
    sum += inside ? a[i+(11*height)] * 0.014925618183728262 : 0.0f;
    sum += inside ? a[i+(12*height)] * 0.011419516182950293 : 0.0f;
    sum += inside ? a[i+(13*height)] * 0.008535940410370202 : 0.0f;
    sum += inside ? a[i+(14*height)] * 0.00623366326536526 : 0.0f;
    sum += inside ? a[i+(15*height)] * 0.004447578233139086 : 0.0f;
    sum += inside ? a[i+(16*height)] * 0.003100217384617136 : 0.0f;
    sum += inside ? a[i+(17*height)] * 0.002111295354219901 : 0.0f;
    sum += inside ? a[i+(18*height)] * 0.0014047341175216514 : 0.0f;
    sum += inside ? a[i+(19*height)] * 0.000913119384660958 : 0.0f;
    ret[i] = sum;
}

kernel void gauss39_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 19) && (x<width-19) && (y >= 19) && (y < height-19);
    //int inside = (19 < i) && (i < (n-19));
    sum +=  inside ?  a[i+(-19)] * 0.000913119384660958 : 0.0f;
    sum +=  inside ?  a[i+(-18)] * 0.0014047341175216514 : 0.0f;
    sum +=  inside ?  a[i+(-17)] * 0.002111295354219901 : 0.0f;
    sum +=  inside ?  a[i+(-16)] * 0.003100217384617136 : 0.0f;
    sum +=  inside ?  a[i+(-15)] * 0.004447578233139086 : 0.0f;
    sum +=  inside ?  a[i+(-14)] * 0.00623366326536526 : 0.0f;
    sum +=  inside ?  a[i+(-13)] * 0.008535940410370202 : 0.0f;
    sum +=  inside ?  a[i+(-12)] * 0.011419516182950293 : 0.0f;
    sum +=  inside ?  a[i+(-11)] * 0.014925618183728262 : 0.0f;
    sum +=  inside ?  a[i+(-10)] * 0.019059225329222967 : 0.0f;
    sum +=  inside ?  a[i+(-9)] * 0.023777514497134777 : 0.0f;
    sum +=  inside ?  a[i+(-8)] * 0.028981172997560947 : 0.0f;
    sum +=  inside ?  a[i+(-7)] * 0.03451069910411127 : 0.0f;
    sum +=  inside ?  a[i+(-6)] * 0.04014947295048908 : 0.0f;
    sum +=  inside ?  a[i+(-5)] * 0.045634598474731086 : 0.0f;
    sum +=  inside ?  a[i+(-4)] * 0.05067536813549639 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.054977866789386806 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.05827296963061248 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.06034408551584046 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.06105068811768207 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.06034408551584046 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.05827296963061248 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.054977866789386806 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.05067536813549639 : 0.0f;
    sum +=  inside ?  a[i+(5)] * 0.045634598474731086 : 0.0f;
    sum +=  inside ?  a[i+(6)] * 0.04014947295048908 : 0.0f;
    sum +=  inside ?  a[i+(7)] * 0.03451069910411127 : 0.0f;
    sum +=  inside ?  a[i+(8)] * 0.028981172997560947 : 0.0f;
    sum +=  inside ?  a[i+(9)] * 0.023777514497134777 : 0.0f;
    sum +=  inside ?  a[i+(10)] * 0.019059225329222967 : 0.0f;
    sum +=  inside ?  a[i+(11)] * 0.014925618183728262 : 0.0f;
    sum +=  inside ?  a[i+(12)] * 0.011419516182950293 : 0.0f;
    sum +=  inside ?  a[i+(13)] * 0.008535940410370202 : 0.0f;
    sum +=  inside ?  a[i+(14)] * 0.00623366326536526 : 0.0f;
    sum +=  inside ?  a[i+(15)] * 0.004447578233139086 : 0.0f;
    sum +=  inside ?  a[i+(16)] * 0.003100217384617136 : 0.0f;
    sum +=  inside ?  a[i+(17)] * 0.002111295354219901 : 0.0f;
    sum +=  inside ?  a[i+(18)] * 0.0014047341175216514 : 0.0f;
    sum +=  inside ?  a[i+(19)] * 0.000913119384660958 : 0.0f;

    ret[i] = sum;
}

kernel void gauss59(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(29*height) < i) && (i < (n-29*height))){
       // Horizontal convolution
        sum = sum + a[i+(-29*height)] * 0.0008346624486312882;
        sum = sum + a[i+(-28*height)] * 0.0010816396680817678;
        sum = sum + a[i+(-27*height)] * 0.0013890070564446745;
        sum = sum + a[i+(-26*height)] * 0.0017675691480281148;
        sum = sum + a[i+(-25*height)] * 0.0022289405634180093;
        sum = sum + a[i+(-24*height)] * 0.002785291653424543;
        sum = sum + a[i+(-23*height)] * 0.0034489983162374906;
        sum = sum + a[i+(-22*height)] * 0.004232192387321655;
        sum = sum + a[i+(-21*height)] * 0.005146215110828653;
        sum = sum + a[i+(-20*height)] * 0.006200983623439666;
        sum = sum + a[i+(-19*height)] * 0.007404288765658636;
        sum = sum + a[i+(-18*height)] * 0.00876105135186714;
        sum = sum + a[i+(-17*height)] * 0.010272572586954246;
        sum = sum + a[i+(-16*height)] * 0.011935821799531288;
        sum = sum + a[i+(-15*height)] * 0.013742810195200031;
        sum = sum + a[i+(-14*height)] * 0.015680102066087814;
        sum = sum + a[i+(-13*height)] * 0.01772851409095386;
        sum = sum + a[i+(-12*height)] * 0.019863048506856777;
        sum = sum + a[i+(-11*height)] * 0.022053096819998785;
        sum = sum + a[i+(-10*height)] * 0.024262937519266066;
        sum = sum + a[i+(-9*height)] * 0.026452534545430947;
        sum = sum + a[i+(-8*height)] * 0.028578624045773795;
        sum = sum + a[i+(-7*height)] * 0.030596056558636356;
        sum = sum + a[i+(-6*height)] * 0.032459341833099106;
        sum = sum + a[i+(-5*height)] * 0.03412432572152615;
        sum = sum + a[i+(-4*height)] * 0.03554991466461278;
        sum = sum + a[i+(-3*height)] * 0.0366997546728828;
        sum = sum + a[i+(-2*height)] * 0.03754376945906495;
        sum = sum + a[i+(-1*height)] * 0.038059467033509466;
        sum = sum + a[i+(0*height)] * 0.03823293557446616;
        sum = sum + a[i+(1*height)] * 0.038059467033509466;
        sum = sum + a[i+(2*height)] * 0.03754376945906495;
        sum = sum + a[i+(3*height)] * 0.0366997546728828;
        sum = sum + a[i+(4*height)] * 0.03554991466461278;
        sum = sum + a[i+(5*height)] * 0.03412432572152615;
        sum = sum + a[i+(6*height)] * 0.032459341833099106;
        sum = sum + a[i+(7*height)] * 0.030596056558636356;
        sum = sum + a[i+(8*height)] * 0.028578624045773795;
        sum = sum + a[i+(9*height)] * 0.026452534545430947;
        sum = sum + a[i+(10*height)] * 0.024262937519266066;
        sum = sum + a[i+(11*height)] * 0.022053096819998785;
        sum = sum + a[i+(12*height)] * 0.019863048506856777;
        sum = sum + a[i+(13*height)] * 0.01772851409095386;
        sum = sum + a[i+(14*height)] * 0.015680102066087814;
        sum = sum + a[i+(15*height)] * 0.013742810195200031;
        sum = sum + a[i+(16*height)] * 0.011935821799531288;
        sum = sum + a[i+(17*height)] * 0.010272572586954246;
        sum = sum + a[i+(18*height)] * 0.00876105135186714;
        sum = sum + a[i+(19*height)] * 0.007404288765658636;
        sum = sum + a[i+(20*height)] * 0.006200983623439666;
        sum = sum + a[i+(21*height)] * 0.005146215110828653;
        sum = sum + a[i+(22*height)] * 0.004232192387321655;
        sum = sum + a[i+(23*height)] * 0.0034489983162374906;
        sum = sum + a[i+(24*height)] * 0.002785291653424543;
        sum = sum + a[i+(25*height)] * 0.0022289405634180093;
        sum = sum + a[i+(26*height)] * 0.0017675691480281148;
        sum = sum + a[i+(27*height)] * 0.0013890070564446745;
        sum = sum + a[i+(28*height)] * 0.0010816396680817678;
        sum = sum + a[i+(29*height)] * 0.0008346624486312882;
    }
    if (vertical && (29 < i) && (i < (n-29))) {
       // Vertical convolution
        sum = sum + a[i+(-29)] * 0.0008346624486312882;
        sum = sum + a[i+(-28)] * 0.0010816396680817678;
        sum = sum + a[i+(-27)] * 0.0013890070564446745;
        sum = sum + a[i+(-26)] * 0.0017675691480281148;
        sum = sum + a[i+(-25)] * 0.0022289405634180093;
        sum = sum + a[i+(-24)] * 0.002785291653424543;
        sum = sum + a[i+(-23)] * 0.0034489983162374906;
        sum = sum + a[i+(-22)] * 0.004232192387321655;
        sum = sum + a[i+(-21)] * 0.005146215110828653;
        sum = sum + a[i+(-20)] * 0.006200983623439666;
        sum = sum + a[i+(-19)] * 0.007404288765658636;
        sum = sum + a[i+(-18)] * 0.00876105135186714;
        sum = sum + a[i+(-17)] * 0.010272572586954246;
        sum = sum + a[i+(-16)] * 0.011935821799531288;
        sum = sum + a[i+(-15)] * 0.013742810195200031;
        sum = sum + a[i+(-14)] * 0.015680102066087814;
        sum = sum + a[i+(-13)] * 0.01772851409095386;
        sum = sum + a[i+(-12)] * 0.019863048506856777;
        sum = sum + a[i+(-11)] * 0.022053096819998785;
        sum = sum + a[i+(-10)] * 0.024262937519266066;
        sum = sum + a[i+(-9)] * 0.026452534545430947;
        sum = sum + a[i+(-8)] * 0.028578624045773795;
        sum = sum + a[i+(-7)] * 0.030596056558636356;
        sum = sum + a[i+(-6)] * 0.032459341833099106;
        sum = sum + a[i+(-5)] * 0.03412432572152615;
        sum = sum + a[i+(-4)] * 0.03554991466461278;
        sum = sum + a[i+(-3)] * 0.0366997546728828;
        sum = sum + a[i+(-2)] * 0.03754376945906495;
        sum = sum + a[i+(-1)] * 0.038059467033509466;
        sum = sum + a[i+(0)] * 0.03823293557446616;
        sum = sum + a[i+(1)] * 0.038059467033509466;
        sum = sum + a[i+(2)] * 0.03754376945906495;
        sum = sum + a[i+(3)] * 0.0366997546728828;
        sum = sum + a[i+(4)] * 0.03554991466461278;
        sum = sum + a[i+(5)] * 0.03412432572152615;
        sum = sum + a[i+(6)] * 0.032459341833099106;
        sum = sum + a[i+(7)] * 0.030596056558636356;
        sum = sum + a[i+(8)] * 0.028578624045773795;
        sum = sum + a[i+(9)] * 0.026452534545430947;
        sum = sum + a[i+(10)] * 0.024262937519266066;
        sum = sum + a[i+(11)] * 0.022053096819998785;
        sum = sum + a[i+(12)] * 0.019863048506856777;
        sum = sum + a[i+(13)] * 0.01772851409095386;
        sum = sum + a[i+(14)] * 0.015680102066087814;
        sum = sum + a[i+(15)] * 0.013742810195200031;
        sum = sum + a[i+(16)] * 0.011935821799531288;
        sum = sum + a[i+(17)] * 0.010272572586954246;
        sum = sum + a[i+(18)] * 0.00876105135186714;
        sum = sum + a[i+(19)] * 0.007404288765658636;
        sum = sum + a[i+(20)] * 0.006200983623439666;
        sum = sum + a[i+(21)] * 0.005146215110828653;
        sum = sum + a[i+(22)] * 0.004232192387321655;
        sum = sum + a[i+(23)] * 0.0034489983162374906;
        sum = sum + a[i+(24)] * 0.002785291653424543;
        sum = sum + a[i+(25)] * 0.0022289405634180093;
        sum = sum + a[i+(26)] * 0.0017675691480281148;
        sum = sum + a[i+(27)] * 0.0013890070564446745;
        sum = sum + a[i+(28)] * 0.0010816396680817678;
        sum = sum + a[i+(29)] * 0.0008346624486312882;
    }
    ret[i] = sum;
}

kernel void gauss59_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 29) && (x<width-29) && (y >= 29) && (y < height-29);
    //int inside = (((size_t)(29*height) < i) && (i < (n-29*height)));
    sum += inside ? a[i+(-29*height)] * 0.0008346624486312882 : 0.0f;
    sum += inside ? a[i+(-28*height)] * 0.0010816396680817678 : 0.0f;
    sum += inside ? a[i+(-27*height)] * 0.0013890070564446745 : 0.0f;
    sum += inside ? a[i+(-26*height)] * 0.0017675691480281148 : 0.0f;
    sum += inside ? a[i+(-25*height)] * 0.0022289405634180093 : 0.0f;
    sum += inside ? a[i+(-24*height)] * 0.002785291653424543 : 0.0f;
    sum += inside ? a[i+(-23*height)] * 0.0034489983162374906 : 0.0f;
    sum += inside ? a[i+(-22*height)] * 0.004232192387321655 : 0.0f;
    sum += inside ? a[i+(-21*height)] * 0.005146215110828653 : 0.0f;
    sum += inside ? a[i+(-20*height)] * 0.006200983623439666 : 0.0f;
    sum += inside ? a[i+(-19*height)] * 0.007404288765658636 : 0.0f;
    sum += inside ? a[i+(-18*height)] * 0.00876105135186714 : 0.0f;
    sum += inside ? a[i+(-17*height)] * 0.010272572586954246 : 0.0f;
    sum += inside ? a[i+(-16*height)] * 0.011935821799531288 : 0.0f;
    sum += inside ? a[i+(-15*height)] * 0.013742810195200031 : 0.0f;
    sum += inside ? a[i+(-14*height)] * 0.015680102066087814 : 0.0f;
    sum += inside ? a[i+(-13*height)] * 0.01772851409095386 : 0.0f;
    sum += inside ? a[i+(-12*height)] * 0.019863048506856777 : 0.0f;
    sum += inside ? a[i+(-11*height)] * 0.022053096819998785 : 0.0f;
    sum += inside ? a[i+(-10*height)] * 0.024262937519266066 : 0.0f;
    sum += inside ? a[i+(-9*height)] * 0.026452534545430947 : 0.0f;
    sum += inside ? a[i+(-8*height)] * 0.028578624045773795 : 0.0f;
    sum += inside ? a[i+(-7*height)] * 0.030596056558636356 : 0.0f;
    sum += inside ? a[i+(-6*height)] * 0.032459341833099106 : 0.0f;
    sum += inside ? a[i+(-5*height)] * 0.03412432572152615 : 0.0f;
    sum += inside ? a[i+(-4*height)] * 0.03554991466461278 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.0366997546728828 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.03754376945906495 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.038059467033509466 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.03823293557446616 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.038059467033509466 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.03754376945906495 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.0366997546728828 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.03554991466461278 : 0.0f;
    sum += inside ? a[i+(5*height)] * 0.03412432572152615 : 0.0f;
    sum += inside ? a[i+(6*height)] * 0.032459341833099106 : 0.0f;
    sum += inside ? a[i+(7*height)] * 0.030596056558636356 : 0.0f;
    sum += inside ? a[i+(8*height)] * 0.028578624045773795 : 0.0f;
    sum += inside ? a[i+(9*height)] * 0.026452534545430947 : 0.0f;
    sum += inside ? a[i+(10*height)] * 0.024262937519266066 : 0.0f;
    sum += inside ? a[i+(11*height)] * 0.022053096819998785 : 0.0f;
    sum += inside ? a[i+(12*height)] * 0.019863048506856777 : 0.0f;
    sum += inside ? a[i+(13*height)] * 0.01772851409095386 : 0.0f;
    sum += inside ? a[i+(14*height)] * 0.015680102066087814 : 0.0f;
    sum += inside ? a[i+(15*height)] * 0.013742810195200031 : 0.0f;
    sum += inside ? a[i+(16*height)] * 0.011935821799531288 : 0.0f;
    sum += inside ? a[i+(17*height)] * 0.010272572586954246 : 0.0f;
    sum += inside ? a[i+(18*height)] * 0.00876105135186714 : 0.0f;
    sum += inside ? a[i+(19*height)] * 0.007404288765658636 : 0.0f;
    sum += inside ? a[i+(20*height)] * 0.006200983623439666 : 0.0f;
    sum += inside ? a[i+(21*height)] * 0.005146215110828653 : 0.0f;
    sum += inside ? a[i+(22*height)] * 0.004232192387321655 : 0.0f;
    sum += inside ? a[i+(23*height)] * 0.0034489983162374906 : 0.0f;
    sum += inside ? a[i+(24*height)] * 0.002785291653424543 : 0.0f;
    sum += inside ? a[i+(25*height)] * 0.0022289405634180093 : 0.0f;
    sum += inside ? a[i+(26*height)] * 0.0017675691480281148 : 0.0f;
    sum += inside ? a[i+(27*height)] * 0.0013890070564446745 : 0.0f;
    sum += inside ? a[i+(28*height)] * 0.0010816396680817678 : 0.0f;
    sum += inside ? a[i+(29*height)] * 0.0008346624486312882 : 0.0f;
    ret[i] = sum;
}

kernel void gauss59_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 29) && (x<width-29) && (y >= 29) && (y < height-29);
    //int inside = (29 < i) && (i < (n-29));
    sum +=  inside ?  a[i+(-29)] * 0.0008346624486312882 : 0.0f;
    sum +=  inside ?  a[i+(-28)] * 0.0010816396680817678 : 0.0f;
    sum +=  inside ?  a[i+(-27)] * 0.0013890070564446745 : 0.0f;
    sum +=  inside ?  a[i+(-26)] * 0.0017675691480281148 : 0.0f;
    sum +=  inside ?  a[i+(-25)] * 0.0022289405634180093 : 0.0f;
    sum +=  inside ?  a[i+(-24)] * 0.002785291653424543 : 0.0f;
    sum +=  inside ?  a[i+(-23)] * 0.0034489983162374906 : 0.0f;
    sum +=  inside ?  a[i+(-22)] * 0.004232192387321655 : 0.0f;
    sum +=  inside ?  a[i+(-21)] * 0.005146215110828653 : 0.0f;
    sum +=  inside ?  a[i+(-20)] * 0.006200983623439666 : 0.0f;
    sum +=  inside ?  a[i+(-19)] * 0.007404288765658636 : 0.0f;
    sum +=  inside ?  a[i+(-18)] * 0.00876105135186714 : 0.0f;
    sum +=  inside ?  a[i+(-17)] * 0.010272572586954246 : 0.0f;
    sum +=  inside ?  a[i+(-16)] * 0.011935821799531288 : 0.0f;
    sum +=  inside ?  a[i+(-15)] * 0.013742810195200031 : 0.0f;
    sum +=  inside ?  a[i+(-14)] * 0.015680102066087814 : 0.0f;
    sum +=  inside ?  a[i+(-13)] * 0.01772851409095386 : 0.0f;
    sum +=  inside ?  a[i+(-12)] * 0.019863048506856777 : 0.0f;
    sum +=  inside ?  a[i+(-11)] * 0.022053096819998785 : 0.0f;
    sum +=  inside ?  a[i+(-10)] * 0.024262937519266066 : 0.0f;
    sum +=  inside ?  a[i+(-9)] * 0.026452534545430947 : 0.0f;
    sum +=  inside ?  a[i+(-8)] * 0.028578624045773795 : 0.0f;
    sum +=  inside ?  a[i+(-7)] * 0.030596056558636356 : 0.0f;
    sum +=  inside ?  a[i+(-6)] * 0.032459341833099106 : 0.0f;
    sum +=  inside ?  a[i+(-5)] * 0.03412432572152615 : 0.0f;
    sum +=  inside ?  a[i+(-4)] * 0.03554991466461278 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.0366997546728828 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.03754376945906495 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.038059467033509466 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.03823293557446616 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.038059467033509466 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.03754376945906495 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.0366997546728828 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.03554991466461278 : 0.0f;
    sum +=  inside ?  a[i+(5)] * 0.03412432572152615 : 0.0f;
    sum +=  inside ?  a[i+(6)] * 0.032459341833099106 : 0.0f;
    sum +=  inside ?  a[i+(7)] * 0.030596056558636356 : 0.0f;
    sum +=  inside ?  a[i+(8)] * 0.028578624045773795 : 0.0f;
    sum +=  inside ?  a[i+(9)] * 0.026452534545430947 : 0.0f;
    sum +=  inside ?  a[i+(10)] * 0.024262937519266066 : 0.0f;
    sum +=  inside ?  a[i+(11)] * 0.022053096819998785 : 0.0f;
    sum +=  inside ?  a[i+(12)] * 0.019863048506856777 : 0.0f;
    sum +=  inside ?  a[i+(13)] * 0.01772851409095386 : 0.0f;
    sum +=  inside ?  a[i+(14)] * 0.015680102066087814 : 0.0f;
    sum +=  inside ?  a[i+(15)] * 0.013742810195200031 : 0.0f;
    sum +=  inside ?  a[i+(16)] * 0.011935821799531288 : 0.0f;
    sum +=  inside ?  a[i+(17)] * 0.010272572586954246 : 0.0f;
    sum +=  inside ?  a[i+(18)] * 0.00876105135186714 : 0.0f;
    sum +=  inside ?  a[i+(19)] * 0.007404288765658636 : 0.0f;
    sum +=  inside ?  a[i+(20)] * 0.006200983623439666 : 0.0f;
    sum +=  inside ?  a[i+(21)] * 0.005146215110828653 : 0.0f;
    sum +=  inside ?  a[i+(22)] * 0.004232192387321655 : 0.0f;
    sum +=  inside ?  a[i+(23)] * 0.0034489983162374906 : 0.0f;
    sum +=  inside ?  a[i+(24)] * 0.002785291653424543 : 0.0f;
    sum +=  inside ?  a[i+(25)] * 0.0022289405634180093 : 0.0f;
    sum +=  inside ?  a[i+(26)] * 0.0017675691480281148 : 0.0f;
    sum +=  inside ?  a[i+(27)] * 0.0013890070564446745 : 0.0f;
    sum +=  inside ?  a[i+(28)] * 0.0010816396680817678 : 0.0f;
    sum +=  inside ?  a[i+(29)] * 0.0008346624486312882 : 0.0f;

    ret[i] = sum;
}

kernel void gauss87(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(43*height) < i) && (i < (n-43*height))){
       // Horizontal convolution
        sum = sum + a[i+(-43*height)] * 0.0008993039625709468;
        sum = sum + a[i+(-42*height)] * 0.0010458773907974057;
        sum = sum + a[i+(-41*height)] * 0.001212026508650259;
        sum = sum + a[i+(-40*height)] * 0.0013995890575663427;
        sum = sum + a[i+(-39*height)] * 0.0016104455076356643;
        sum = sum + a[i+(-38*height)] * 0.0018464969991795126;
        sum = sum + a[i+(-37*height)] * 0.002109639528087963;
        sum = sum + a[i+(-36*height)] * 0.0024017343985059617;
        sum = sum + a[i+(-35*height)] * 0.0027245750690657517;
        sum = sum + a[i+(-34*height)] * 0.0030798506326501626;
        sum = sum + a[i+(-33*height)] * 0.003469106292374379;
        sum = sum + a[i+(-32*height)] * 0.0038937013251737576;
        sum = sum + a[i+(-31*height)] * 0.004354765155533271;
        sum = sum + a[i+(-30*height)] * 0.00485315229137409;
        sum = sum + a[i+(-29*height)] * 0.0053893969973359;
        sum = sum + a[i+(-28*height)] * 0.00596366869272753;
        sum = sum + a[i+(-27*height)] * 0.0065757291571481155;
        sum = sum + a[i+(-26*height)] * 0.007224892701095954;
        sum = sum + a[i+(-25*height)] * 0.007909990506887891;
        sum = sum + a[i+(-24*height)] * 0.008629340362453617;
        sum = sum + a[i+(-23*height)] * 0.009380722993261873;
        sum = sum + a[i+(-22*height)] * 0.010161366142887922;
        sum = sum + a[i+(-21*height)] * 0.010967937458754493;
        sum = sum + a[i+(-20*height)] * 0.011796547105862186;
        sum = sum + a[i+(-19*height)] * 0.01264276085878075;
        sum = sum + a[i+(-18*height)] * 0.013501624213218474;
        sum = sum + a[i+(-17*height)] * 0.014367697817081183;
        sum = sum + a[i+(-16*height)] * 0.015235104252536422;
        sum = sum + a[i+(-15*height)] * 0.016097585912073973;
        sum = sum + a[i+(-14*height)] * 0.016948573410988158;
        sum = sum + a[i+(-13*height)] * 0.017781263675176226;
        sum = sum + a[i+(-12*height)] * 0.018588706546412734;
        sum = sum + a[i+(-11*height)] * 0.0193638984674273;
        sum = sum + a[i+(-10*height)] * 0.02009988155624177;
        sum = sum + a[i+(-9*height)] * 0.020789846162928012;
        sum = sum + a[i+(-8*height)] * 0.02142723483099038;
        sum = sum + a[i+(-7*height)] * 0.02200584546748023;
        sum = sum + a[i+(-6*height)] * 0.022519931466643595;
        sum = sum + a[i+(-5*height)] * 0.022964296535426684;
        sum = sum + a[i+(-4*height)] * 0.02333438203744017;
        sum = sum + a[i+(-3*height)] * 0.023626344804671157;
        sum = sum + a[i+(-2*height)] * 0.02383712356067809;
        sum = sum + a[i+(-1*height)] * 0.023964492350286193;
        sum = sum + a[i+(0*height)] * 0.024007099671875188;
        sum = sum + a[i+(1*height)] * 0.023964492350286193;
        sum = sum + a[i+(2*height)] * 0.02383712356067809;
        sum = sum + a[i+(3*height)] * 0.023626344804671157;
        sum = sum + a[i+(4*height)] * 0.02333438203744017;
        sum = sum + a[i+(5*height)] * 0.022964296535426684;
        sum = sum + a[i+(6*height)] * 0.022519931466643595;
        sum = sum + a[i+(7*height)] * 0.02200584546748023;
        sum = sum + a[i+(8*height)] * 0.02142723483099038;
        sum = sum + a[i+(9*height)] * 0.020789846162928012;
        sum = sum + a[i+(10*height)] * 0.02009988155624177;
        sum = sum + a[i+(11*height)] * 0.0193638984674273;
        sum = sum + a[i+(12*height)] * 0.018588706546412734;
        sum = sum + a[i+(13*height)] * 0.017781263675176226;
        sum = sum + a[i+(14*height)] * 0.016948573410988158;
        sum = sum + a[i+(15*height)] * 0.016097585912073973;
        sum = sum + a[i+(16*height)] * 0.015235104252536422;
        sum = sum + a[i+(17*height)] * 0.014367697817081183;
        sum = sum + a[i+(18*height)] * 0.013501624213218474;
        sum = sum + a[i+(19*height)] * 0.01264276085878075;
        sum = sum + a[i+(20*height)] * 0.011796547105862186;
        sum = sum + a[i+(21*height)] * 0.010967937458754493;
        sum = sum + a[i+(22*height)] * 0.010161366142887922;
        sum = sum + a[i+(23*height)] * 0.009380722993261873;
        sum = sum + a[i+(24*height)] * 0.008629340362453617;
        sum = sum + a[i+(25*height)] * 0.007909990506887891;
        sum = sum + a[i+(26*height)] * 0.007224892701095954;
        sum = sum + a[i+(27*height)] * 0.0065757291571481155;
        sum = sum + a[i+(28*height)] * 0.00596366869272753;
        sum = sum + a[i+(29*height)] * 0.0053893969973359;
        sum = sum + a[i+(30*height)] * 0.00485315229137409;
        sum = sum + a[i+(31*height)] * 0.004354765155533271;
        sum = sum + a[i+(32*height)] * 0.0038937013251737576;
        sum = sum + a[i+(33*height)] * 0.003469106292374379;
        sum = sum + a[i+(34*height)] * 0.0030798506326501626;
        sum = sum + a[i+(35*height)] * 0.0027245750690657517;
        sum = sum + a[i+(36*height)] * 0.0024017343985059617;
        sum = sum + a[i+(37*height)] * 0.002109639528087963;
        sum = sum + a[i+(38*height)] * 0.0018464969991795126;
        sum = sum + a[i+(39*height)] * 0.0016104455076356643;
        sum = sum + a[i+(40*height)] * 0.0013995890575663427;
        sum = sum + a[i+(41*height)] * 0.001212026508650259;
        sum = sum + a[i+(42*height)] * 0.0010458773907974057;
        sum = sum + a[i+(43*height)] * 0.0008993039625709468;
    }
    if (vertical && (43 < i) && (i < (n-43))) {
       // Vertical convolution
        sum = sum + a[i+(-43)] * 0.0008993039625709468;
        sum = sum + a[i+(-42)] * 0.0010458773907974057;
        sum = sum + a[i+(-41)] * 0.001212026508650259;
        sum = sum + a[i+(-40)] * 0.0013995890575663427;
        sum = sum + a[i+(-39)] * 0.0016104455076356643;
        sum = sum + a[i+(-38)] * 0.0018464969991795126;
        sum = sum + a[i+(-37)] * 0.002109639528087963;
        sum = sum + a[i+(-36)] * 0.0024017343985059617;
        sum = sum + a[i+(-35)] * 0.0027245750690657517;
        sum = sum + a[i+(-34)] * 0.0030798506326501626;
        sum = sum + a[i+(-33)] * 0.003469106292374379;
        sum = sum + a[i+(-32)] * 0.0038937013251737576;
        sum = sum + a[i+(-31)] * 0.004354765155533271;
        sum = sum + a[i+(-30)] * 0.00485315229137409;
        sum = sum + a[i+(-29)] * 0.0053893969973359;
        sum = sum + a[i+(-28)] * 0.00596366869272753;
        sum = sum + a[i+(-27)] * 0.0065757291571481155;
        sum = sum + a[i+(-26)] * 0.007224892701095954;
        sum = sum + a[i+(-25)] * 0.007909990506887891;
        sum = sum + a[i+(-24)] * 0.008629340362453617;
        sum = sum + a[i+(-23)] * 0.009380722993261873;
        sum = sum + a[i+(-22)] * 0.010161366142887922;
        sum = sum + a[i+(-21)] * 0.010967937458754493;
        sum = sum + a[i+(-20)] * 0.011796547105862186;
        sum = sum + a[i+(-19)] * 0.01264276085878075;
        sum = sum + a[i+(-18)] * 0.013501624213218474;
        sum = sum + a[i+(-17)] * 0.014367697817081183;
        sum = sum + a[i+(-16)] * 0.015235104252536422;
        sum = sum + a[i+(-15)] * 0.016097585912073973;
        sum = sum + a[i+(-14)] * 0.016948573410988158;
        sum = sum + a[i+(-13)] * 0.017781263675176226;
        sum = sum + a[i+(-12)] * 0.018588706546412734;
        sum = sum + a[i+(-11)] * 0.0193638984674273;
        sum = sum + a[i+(-10)] * 0.02009988155624177;
        sum = sum + a[i+(-9)] * 0.020789846162928012;
        sum = sum + a[i+(-8)] * 0.02142723483099038;
        sum = sum + a[i+(-7)] * 0.02200584546748023;
        sum = sum + a[i+(-6)] * 0.022519931466643595;
        sum = sum + a[i+(-5)] * 0.022964296535426684;
        sum = sum + a[i+(-4)] * 0.02333438203744017;
        sum = sum + a[i+(-3)] * 0.023626344804671157;
        sum = sum + a[i+(-2)] * 0.02383712356067809;
        sum = sum + a[i+(-1)] * 0.023964492350286193;
        sum = sum + a[i+(0)] * 0.024007099671875188;
        sum = sum + a[i+(1)] * 0.023964492350286193;
        sum = sum + a[i+(2)] * 0.02383712356067809;
        sum = sum + a[i+(3)] * 0.023626344804671157;
        sum = sum + a[i+(4)] * 0.02333438203744017;
        sum = sum + a[i+(5)] * 0.022964296535426684;
        sum = sum + a[i+(6)] * 0.022519931466643595;
        sum = sum + a[i+(7)] * 0.02200584546748023;
        sum = sum + a[i+(8)] * 0.02142723483099038;
        sum = sum + a[i+(9)] * 0.020789846162928012;
        sum = sum + a[i+(10)] * 0.02009988155624177;
        sum = sum + a[i+(11)] * 0.0193638984674273;
        sum = sum + a[i+(12)] * 0.018588706546412734;
        sum = sum + a[i+(13)] * 0.017781263675176226;
        sum = sum + a[i+(14)] * 0.016948573410988158;
        sum = sum + a[i+(15)] * 0.016097585912073973;
        sum = sum + a[i+(16)] * 0.015235104252536422;
        sum = sum + a[i+(17)] * 0.014367697817081183;
        sum = sum + a[i+(18)] * 0.013501624213218474;
        sum = sum + a[i+(19)] * 0.01264276085878075;
        sum = sum + a[i+(20)] * 0.011796547105862186;
        sum = sum + a[i+(21)] * 0.010967937458754493;
        sum = sum + a[i+(22)] * 0.010161366142887922;
        sum = sum + a[i+(23)] * 0.009380722993261873;
        sum = sum + a[i+(24)] * 0.008629340362453617;
        sum = sum + a[i+(25)] * 0.007909990506887891;
        sum = sum + a[i+(26)] * 0.007224892701095954;
        sum = sum + a[i+(27)] * 0.0065757291571481155;
        sum = sum + a[i+(28)] * 0.00596366869272753;
        sum = sum + a[i+(29)] * 0.0053893969973359;
        sum = sum + a[i+(30)] * 0.00485315229137409;
        sum = sum + a[i+(31)] * 0.004354765155533271;
        sum = sum + a[i+(32)] * 0.0038937013251737576;
        sum = sum + a[i+(33)] * 0.003469106292374379;
        sum = sum + a[i+(34)] * 0.0030798506326501626;
        sum = sum + a[i+(35)] * 0.0027245750690657517;
        sum = sum + a[i+(36)] * 0.0024017343985059617;
        sum = sum + a[i+(37)] * 0.002109639528087963;
        sum = sum + a[i+(38)] * 0.0018464969991795126;
        sum = sum + a[i+(39)] * 0.0016104455076356643;
        sum = sum + a[i+(40)] * 0.0013995890575663427;
        sum = sum + a[i+(41)] * 0.001212026508650259;
        sum = sum + a[i+(42)] * 0.0010458773907974057;
        sum = sum + a[i+(43)] * 0.0008993039625709468;
    }
    ret[i] = sum;
}

kernel void gauss87_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 43) && (x<width-43) && (y >= 43) && (y < height-43);
    //int inside = (((size_t)(43*height) < i) && (i < (n-43*height)));
    sum += inside ? a[i+(-43*height)] * 0.0008993039625709468 : 0.0f;
    sum += inside ? a[i+(-42*height)] * 0.0010458773907974057 : 0.0f;
    sum += inside ? a[i+(-41*height)] * 0.001212026508650259 : 0.0f;
    sum += inside ? a[i+(-40*height)] * 0.0013995890575663427 : 0.0f;
    sum += inside ? a[i+(-39*height)] * 0.0016104455076356643 : 0.0f;
    sum += inside ? a[i+(-38*height)] * 0.0018464969991795126 : 0.0f;
    sum += inside ? a[i+(-37*height)] * 0.002109639528087963 : 0.0f;
    sum += inside ? a[i+(-36*height)] * 0.0024017343985059617 : 0.0f;
    sum += inside ? a[i+(-35*height)] * 0.0027245750690657517 : 0.0f;
    sum += inside ? a[i+(-34*height)] * 0.0030798506326501626 : 0.0f;
    sum += inside ? a[i+(-33*height)] * 0.003469106292374379 : 0.0f;
    sum += inside ? a[i+(-32*height)] * 0.0038937013251737576 : 0.0f;
    sum += inside ? a[i+(-31*height)] * 0.004354765155533271 : 0.0f;
    sum += inside ? a[i+(-30*height)] * 0.00485315229137409 : 0.0f;
    sum += inside ? a[i+(-29*height)] * 0.0053893969973359 : 0.0f;
    sum += inside ? a[i+(-28*height)] * 0.00596366869272753 : 0.0f;
    sum += inside ? a[i+(-27*height)] * 0.0065757291571481155 : 0.0f;
    sum += inside ? a[i+(-26*height)] * 0.007224892701095954 : 0.0f;
    sum += inside ? a[i+(-25*height)] * 0.007909990506887891 : 0.0f;
    sum += inside ? a[i+(-24*height)] * 0.008629340362453617 : 0.0f;
    sum += inside ? a[i+(-23*height)] * 0.009380722993261873 : 0.0f;
    sum += inside ? a[i+(-22*height)] * 0.010161366142887922 : 0.0f;
    sum += inside ? a[i+(-21*height)] * 0.010967937458754493 : 0.0f;
    sum += inside ? a[i+(-20*height)] * 0.011796547105862186 : 0.0f;
    sum += inside ? a[i+(-19*height)] * 0.01264276085878075 : 0.0f;
    sum += inside ? a[i+(-18*height)] * 0.013501624213218474 : 0.0f;
    sum += inside ? a[i+(-17*height)] * 0.014367697817081183 : 0.0f;
    sum += inside ? a[i+(-16*height)] * 0.015235104252536422 : 0.0f;
    sum += inside ? a[i+(-15*height)] * 0.016097585912073973 : 0.0f;
    sum += inside ? a[i+(-14*height)] * 0.016948573410988158 : 0.0f;
    sum += inside ? a[i+(-13*height)] * 0.017781263675176226 : 0.0f;
    sum += inside ? a[i+(-12*height)] * 0.018588706546412734 : 0.0f;
    sum += inside ? a[i+(-11*height)] * 0.0193638984674273 : 0.0f;
    sum += inside ? a[i+(-10*height)] * 0.02009988155624177 : 0.0f;
    sum += inside ? a[i+(-9*height)] * 0.020789846162928012 : 0.0f;
    sum += inside ? a[i+(-8*height)] * 0.02142723483099038 : 0.0f;
    sum += inside ? a[i+(-7*height)] * 0.02200584546748023 : 0.0f;
    sum += inside ? a[i+(-6*height)] * 0.022519931466643595 : 0.0f;
    sum += inside ? a[i+(-5*height)] * 0.022964296535426684 : 0.0f;
    sum += inside ? a[i+(-4*height)] * 0.02333438203744017 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.023626344804671157 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.02383712356067809 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.023964492350286193 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.024007099671875188 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.023964492350286193 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.02383712356067809 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.023626344804671157 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.02333438203744017 : 0.0f;
    sum += inside ? a[i+(5*height)] * 0.022964296535426684 : 0.0f;
    sum += inside ? a[i+(6*height)] * 0.022519931466643595 : 0.0f;
    sum += inside ? a[i+(7*height)] * 0.02200584546748023 : 0.0f;
    sum += inside ? a[i+(8*height)] * 0.02142723483099038 : 0.0f;
    sum += inside ? a[i+(9*height)] * 0.020789846162928012 : 0.0f;
    sum += inside ? a[i+(10*height)] * 0.02009988155624177 : 0.0f;
    sum += inside ? a[i+(11*height)] * 0.0193638984674273 : 0.0f;
    sum += inside ? a[i+(12*height)] * 0.018588706546412734 : 0.0f;
    sum += inside ? a[i+(13*height)] * 0.017781263675176226 : 0.0f;
    sum += inside ? a[i+(14*height)] * 0.016948573410988158 : 0.0f;
    sum += inside ? a[i+(15*height)] * 0.016097585912073973 : 0.0f;
    sum += inside ? a[i+(16*height)] * 0.015235104252536422 : 0.0f;
    sum += inside ? a[i+(17*height)] * 0.014367697817081183 : 0.0f;
    sum += inside ? a[i+(18*height)] * 0.013501624213218474 : 0.0f;
    sum += inside ? a[i+(19*height)] * 0.01264276085878075 : 0.0f;
    sum += inside ? a[i+(20*height)] * 0.011796547105862186 : 0.0f;
    sum += inside ? a[i+(21*height)] * 0.010967937458754493 : 0.0f;
    sum += inside ? a[i+(22*height)] * 0.010161366142887922 : 0.0f;
    sum += inside ? a[i+(23*height)] * 0.009380722993261873 : 0.0f;
    sum += inside ? a[i+(24*height)] * 0.008629340362453617 : 0.0f;
    sum += inside ? a[i+(25*height)] * 0.007909990506887891 : 0.0f;
    sum += inside ? a[i+(26*height)] * 0.007224892701095954 : 0.0f;
    sum += inside ? a[i+(27*height)] * 0.0065757291571481155 : 0.0f;
    sum += inside ? a[i+(28*height)] * 0.00596366869272753 : 0.0f;
    sum += inside ? a[i+(29*height)] * 0.0053893969973359 : 0.0f;
    sum += inside ? a[i+(30*height)] * 0.00485315229137409 : 0.0f;
    sum += inside ? a[i+(31*height)] * 0.004354765155533271 : 0.0f;
    sum += inside ? a[i+(32*height)] * 0.0038937013251737576 : 0.0f;
    sum += inside ? a[i+(33*height)] * 0.003469106292374379 : 0.0f;
    sum += inside ? a[i+(34*height)] * 0.0030798506326501626 : 0.0f;
    sum += inside ? a[i+(35*height)] * 0.0027245750690657517 : 0.0f;
    sum += inside ? a[i+(36*height)] * 0.0024017343985059617 : 0.0f;
    sum += inside ? a[i+(37*height)] * 0.002109639528087963 : 0.0f;
    sum += inside ? a[i+(38*height)] * 0.0018464969991795126 : 0.0f;
    sum += inside ? a[i+(39*height)] * 0.0016104455076356643 : 0.0f;
    sum += inside ? a[i+(40*height)] * 0.0013995890575663427 : 0.0f;
    sum += inside ? a[i+(41*height)] * 0.001212026508650259 : 0.0f;
    sum += inside ? a[i+(42*height)] * 0.0010458773907974057 : 0.0f;
    sum += inside ? a[i+(43*height)] * 0.0008993039625709468 : 0.0f;
    ret[i] = sum;
}

kernel void gauss87_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 43) && (x<width-43) && (y >= 43) && (y < height-43);
    //int inside = (43 < i) && (i < (n-43));
    sum +=  inside ?  a[i+(-43)] * 0.0008993039625709468 : 0.0f;
    sum +=  inside ?  a[i+(-42)] * 0.0010458773907974057 : 0.0f;
    sum +=  inside ?  a[i+(-41)] * 0.001212026508650259 : 0.0f;
    sum +=  inside ?  a[i+(-40)] * 0.0013995890575663427 : 0.0f;
    sum +=  inside ?  a[i+(-39)] * 0.0016104455076356643 : 0.0f;
    sum +=  inside ?  a[i+(-38)] * 0.0018464969991795126 : 0.0f;
    sum +=  inside ?  a[i+(-37)] * 0.002109639528087963 : 0.0f;
    sum +=  inside ?  a[i+(-36)] * 0.0024017343985059617 : 0.0f;
    sum +=  inside ?  a[i+(-35)] * 0.0027245750690657517 : 0.0f;
    sum +=  inside ?  a[i+(-34)] * 0.0030798506326501626 : 0.0f;
    sum +=  inside ?  a[i+(-33)] * 0.003469106292374379 : 0.0f;
    sum +=  inside ?  a[i+(-32)] * 0.0038937013251737576 : 0.0f;
    sum +=  inside ?  a[i+(-31)] * 0.004354765155533271 : 0.0f;
    sum +=  inside ?  a[i+(-30)] * 0.00485315229137409 : 0.0f;
    sum +=  inside ?  a[i+(-29)] * 0.0053893969973359 : 0.0f;
    sum +=  inside ?  a[i+(-28)] * 0.00596366869272753 : 0.0f;
    sum +=  inside ?  a[i+(-27)] * 0.0065757291571481155 : 0.0f;
    sum +=  inside ?  a[i+(-26)] * 0.007224892701095954 : 0.0f;
    sum +=  inside ?  a[i+(-25)] * 0.007909990506887891 : 0.0f;
    sum +=  inside ?  a[i+(-24)] * 0.008629340362453617 : 0.0f;
    sum +=  inside ?  a[i+(-23)] * 0.009380722993261873 : 0.0f;
    sum +=  inside ?  a[i+(-22)] * 0.010161366142887922 : 0.0f;
    sum +=  inside ?  a[i+(-21)] * 0.010967937458754493 : 0.0f;
    sum +=  inside ?  a[i+(-20)] * 0.011796547105862186 : 0.0f;
    sum +=  inside ?  a[i+(-19)] * 0.01264276085878075 : 0.0f;
    sum +=  inside ?  a[i+(-18)] * 0.013501624213218474 : 0.0f;
    sum +=  inside ?  a[i+(-17)] * 0.014367697817081183 : 0.0f;
    sum +=  inside ?  a[i+(-16)] * 0.015235104252536422 : 0.0f;
    sum +=  inside ?  a[i+(-15)] * 0.016097585912073973 : 0.0f;
    sum +=  inside ?  a[i+(-14)] * 0.016948573410988158 : 0.0f;
    sum +=  inside ?  a[i+(-13)] * 0.017781263675176226 : 0.0f;
    sum +=  inside ?  a[i+(-12)] * 0.018588706546412734 : 0.0f;
    sum +=  inside ?  a[i+(-11)] * 0.0193638984674273 : 0.0f;
    sum +=  inside ?  a[i+(-10)] * 0.02009988155624177 : 0.0f;
    sum +=  inside ?  a[i+(-9)] * 0.020789846162928012 : 0.0f;
    sum +=  inside ?  a[i+(-8)] * 0.02142723483099038 : 0.0f;
    sum +=  inside ?  a[i+(-7)] * 0.02200584546748023 : 0.0f;
    sum +=  inside ?  a[i+(-6)] * 0.022519931466643595 : 0.0f;
    sum +=  inside ?  a[i+(-5)] * 0.022964296535426684 : 0.0f;
    sum +=  inside ?  a[i+(-4)] * 0.02333438203744017 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.023626344804671157 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.02383712356067809 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.023964492350286193 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.024007099671875188 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.023964492350286193 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.02383712356067809 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.023626344804671157 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.02333438203744017 : 0.0f;
    sum +=  inside ?  a[i+(5)] * 0.022964296535426684 : 0.0f;
    sum +=  inside ?  a[i+(6)] * 0.022519931466643595 : 0.0f;
    sum +=  inside ?  a[i+(7)] * 0.02200584546748023 : 0.0f;
    sum +=  inside ?  a[i+(8)] * 0.02142723483099038 : 0.0f;
    sum +=  inside ?  a[i+(9)] * 0.020789846162928012 : 0.0f;
    sum +=  inside ?  a[i+(10)] * 0.02009988155624177 : 0.0f;
    sum +=  inside ?  a[i+(11)] * 0.0193638984674273 : 0.0f;
    sum +=  inside ?  a[i+(12)] * 0.018588706546412734 : 0.0f;
    sum +=  inside ?  a[i+(13)] * 0.017781263675176226 : 0.0f;
    sum +=  inside ?  a[i+(14)] * 0.016948573410988158 : 0.0f;
    sum +=  inside ?  a[i+(15)] * 0.016097585912073973 : 0.0f;
    sum +=  inside ?  a[i+(16)] * 0.015235104252536422 : 0.0f;
    sum +=  inside ?  a[i+(17)] * 0.014367697817081183 : 0.0f;
    sum +=  inside ?  a[i+(18)] * 0.013501624213218474 : 0.0f;
    sum +=  inside ?  a[i+(19)] * 0.01264276085878075 : 0.0f;
    sum +=  inside ?  a[i+(20)] * 0.011796547105862186 : 0.0f;
    sum +=  inside ?  a[i+(21)] * 0.010967937458754493 : 0.0f;
    sum +=  inside ?  a[i+(22)] * 0.010161366142887922 : 0.0f;
    sum +=  inside ?  a[i+(23)] * 0.009380722993261873 : 0.0f;
    sum +=  inside ?  a[i+(24)] * 0.008629340362453617 : 0.0f;
    sum +=  inside ?  a[i+(25)] * 0.007909990506887891 : 0.0f;
    sum +=  inside ?  a[i+(26)] * 0.007224892701095954 : 0.0f;
    sum +=  inside ?  a[i+(27)] * 0.0065757291571481155 : 0.0f;
    sum +=  inside ?  a[i+(28)] * 0.00596366869272753 : 0.0f;
    sum +=  inside ?  a[i+(29)] * 0.0053893969973359 : 0.0f;
    sum +=  inside ?  a[i+(30)] * 0.00485315229137409 : 0.0f;
    sum +=  inside ?  a[i+(31)] * 0.004354765155533271 : 0.0f;
    sum +=  inside ?  a[i+(32)] * 0.0038937013251737576 : 0.0f;
    sum +=  inside ?  a[i+(33)] * 0.003469106292374379 : 0.0f;
    sum +=  inside ?  a[i+(34)] * 0.0030798506326501626 : 0.0f;
    sum +=  inside ?  a[i+(35)] * 0.0027245750690657517 : 0.0f;
    sum +=  inside ?  a[i+(36)] * 0.0024017343985059617 : 0.0f;
    sum +=  inside ?  a[i+(37)] * 0.002109639528087963 : 0.0f;
    sum +=  inside ?  a[i+(38)] * 0.0018464969991795126 : 0.0f;
    sum +=  inside ?  a[i+(39)] * 0.0016104455076356643 : 0.0f;
    sum +=  inside ?  a[i+(40)] * 0.0013995890575663427 : 0.0f;
    sum +=  inside ?  a[i+(41)] * 0.001212026508650259 : 0.0f;
    sum +=  inside ?  a[i+(42)] * 0.0010458773907974057 : 0.0f;
    sum +=  inside ?  a[i+(43)] * 0.0008993039625709468 : 0.0f;

    ret[i] = sum;
}

kernel void gauss127(constant int &vertical [[ buffer(0) ]],
                    constant int &width [[ buffer(1) ]],
                    constant int &height [[ buffer(2) ]],
                    const device float* a [[ buffer(3) ]],
                    device float* ret [[ buffer(4) ]],
                    uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    if (!vertical && ((size_t)(63*height) < i) && (i < (n-63*height))){
       // Horizontal convolution
        sum = sum + a[i+(-63*height)] * 0.0009635840700714965;
        sum = sum + a[i+(-62*height)] * 0.0010508933728874807;
        sum = sum + a[i+(-61*height)] * 0.00114452422742914;
        sum = sum + a[i+(-60*height)] * 0.0012447685947797947;
        sum = sum + a[i+(-59*height)] * 0.001351915509170473;
        sum = sum + a[i+(-58*height)] * 0.0014662491497820617;
        sum = sum + a[i+(-57*height)] * 0.0015880467803052096;
        sum = sum + a[i+(-56*height)] * 0.0017175765627405672;
        sum = sum + a[i+(-55*height)] * 0.0018550952537454528;
        sum = sum + a[i+(-54*height)] * 0.00200084579371474;
        sum = sum + a[i+(-53*height)] * 0.0021550548007040924;
        sum = sum + a[i+(-52*height)] * 0.0023179299832409824;
        sum = sum + a[i+(-51*height)] * 0.0024896574879993603;
        sum = sum + a[i+(-50*height)] * 0.0026703992002119785;
        sum = sum + a[i+(-49*height)] * 0.0028602900165330338;
        sum = sum + a[i+(-48*height)] * 0.0030594351118147342;
        sum = sum + a[i+(-47*height)] * 0.0032679072228954287;
        sum = sum + a[i+(-46*height)] * 0.0034857439739843137;
        sum = sum + a[i+(-45*height)] * 0.003712945269538897;
        sum = sum + a[i+(-44*height)] * 0.00394947078163683;
        sum = sum + a[i+(-43*height)] * 0.00419523755971528;
        sum = sum + a[i+(-42*height)] * 0.004450117791161477;
        sum = sum + a[i+(-41*height)] * 0.004713936741562557;
        sum = sum + a[i+(-40*height)] * 0.004986470903438759;
        sum = sum + a[i+(-39*height)] * 0.005267446381971586;
        sum = sum + a[i+(-38*height)] * 0.005556537545581941;
        sum = sum + a[i+(-37*height)] * 0.00585336596819982;
        sum = sum + a[i+(-36*height)] * 0.006157499688689468;
        sum = sum + a[i+(-35*height)] * 0.006468452811148505;
        sum = sum + a[i+(-34*height)] * 0.006785685467688546;
        sum = sum + a[i+(-33*height)] * 0.0071086041628354345;
        sum = sum + a[i+(-32*height)] * 0.00743656251587208;
        sum = sum + a[i+(-31*height)] * 0.007768862414304341;
        sum = sum + a[i+(-30*height)] * 0.008104755588184343;
        sum = sum + a[i+(-29*height)] * 0.008443445611305217;
        sum = sum + a[i+(-28*height)] * 0.00878409033132127;
        sum = sum + a[i+(-27*height)] * 0.00912580472668759;
        sum = sum + a[i+(-26*height)] * 0.009467664183997209;
        sum = sum + a[i+(-25*height)] * 0.009808708184870377;
        sum = sum + a[i+(-24*height)] * 0.010147944387071587;
        sum = sum + a[i+(-23*height)] * 0.010484353080049538;
        sum = sum + a[i+(-22*height)] * 0.010816891990671151;
        sum = sum + a[i+(-21*height)] * 0.011144501410610371;
        sum = sum + a[i+(-20*height)] * 0.011466109612714988;
        sum = sum + a[i+(-19*height)] * 0.011780638519768178;
        sum = sum + a[i+(-18*height)] * 0.012087009585442916;
        sum = sum + a[i+(-17*height)] * 0.0123841498439715;
        sum = sum + a[i+(-16*height)] * 0.0126709980821707;
        sum = sum + a[i+(-15*height)] * 0.012946511085023192;
        sum = sum + a[i+(-14*height)] * 0.01320966990406012;
        sum = sum + a[i+(-13*height)] * 0.013459486096355404;
        sum = sum + a[i+(-12*height)] * 0.01369500788105957;
        sum = sum + a[i+(-11*height)] * 0.013915326160093642;
        sum = sum + a[i+(-10*height)] * 0.014119580349907273;
        sum = sum + a[i+(-9*height)] * 0.014306963972088468;
        sum = sum + a[i+(-8*height)] * 0.014476729952094189;
        sum = sum + a[i+(-7*height)] * 0.014628195577443737;
        sum = sum + a[i+(-6*height)] * 0.014760747069362649;
        sum = sum + a[i+(-5*height)] * 0.01487384372505926;
        sum = sum + a[i+(-4*height)] * 0.014967021591525414;
        sum = sum + a[i+(-3*height)] * 0.015039896635936488;
        sum = sum + a[i+(-2*height)] * 0.015092167382336447;
        sum = sum + a[i+(-1*height)] * 0.015123616989276317;
        sum = sum + a[i+(0*height)] * 0.01513411474837013;
        sum = sum + a[i+(1*height)] * 0.015123616989276317;
        sum = sum + a[i+(2*height)] * 0.015092167382336447;
        sum = sum + a[i+(3*height)] * 0.015039896635936488;
        sum = sum + a[i+(4*height)] * 0.014967021591525414;
        sum = sum + a[i+(5*height)] * 0.01487384372505926;
        sum = sum + a[i+(6*height)] * 0.014760747069362649;
        sum = sum + a[i+(7*height)] * 0.014628195577443737;
        sum = sum + a[i+(8*height)] * 0.014476729952094189;
        sum = sum + a[i+(9*height)] * 0.014306963972088468;
        sum = sum + a[i+(10*height)] * 0.014119580349907273;
        sum = sum + a[i+(11*height)] * 0.013915326160093642;
        sum = sum + a[i+(12*height)] * 0.01369500788105957;
        sum = sum + a[i+(13*height)] * 0.013459486096355404;
        sum = sum + a[i+(14*height)] * 0.01320966990406012;
        sum = sum + a[i+(15*height)] * 0.012946511085023192;
        sum = sum + a[i+(16*height)] * 0.0126709980821707;
        sum = sum + a[i+(17*height)] * 0.0123841498439715;
        sum = sum + a[i+(18*height)] * 0.012087009585442916;
        sum = sum + a[i+(19*height)] * 0.011780638519768178;
        sum = sum + a[i+(20*height)] * 0.011466109612714988;
        sum = sum + a[i+(21*height)] * 0.011144501410610371;
        sum = sum + a[i+(22*height)] * 0.010816891990671151;
        sum = sum + a[i+(23*height)] * 0.010484353080049538;
        sum = sum + a[i+(24*height)] * 0.010147944387071587;
        sum = sum + a[i+(25*height)] * 0.009808708184870377;
        sum = sum + a[i+(26*height)] * 0.009467664183997209;
        sum = sum + a[i+(27*height)] * 0.00912580472668759;
        sum = sum + a[i+(28*height)] * 0.00878409033132127;
        sum = sum + a[i+(29*height)] * 0.008443445611305217;
        sum = sum + a[i+(30*height)] * 0.008104755588184343;
        sum = sum + a[i+(31*height)] * 0.007768862414304341;
        sum = sum + a[i+(32*height)] * 0.00743656251587208;
        sum = sum + a[i+(33*height)] * 0.0071086041628354345;
        sum = sum + a[i+(34*height)] * 0.006785685467688546;
        sum = sum + a[i+(35*height)] * 0.006468452811148505;
        sum = sum + a[i+(36*height)] * 0.006157499688689468;
        sum = sum + a[i+(37*height)] * 0.00585336596819982;
        sum = sum + a[i+(38*height)] * 0.005556537545581941;
        sum = sum + a[i+(39*height)] * 0.005267446381971586;
        sum = sum + a[i+(40*height)] * 0.004986470903438759;
        sum = sum + a[i+(41*height)] * 0.004713936741562557;
        sum = sum + a[i+(42*height)] * 0.004450117791161477;
        sum = sum + a[i+(43*height)] * 0.00419523755971528;
        sum = sum + a[i+(44*height)] * 0.00394947078163683;
        sum = sum + a[i+(45*height)] * 0.003712945269538897;
        sum = sum + a[i+(46*height)] * 0.0034857439739843137;
        sum = sum + a[i+(47*height)] * 0.0032679072228954287;
        sum = sum + a[i+(48*height)] * 0.0030594351118147342;
        sum = sum + a[i+(49*height)] * 0.0028602900165330338;
        sum = sum + a[i+(50*height)] * 0.0026703992002119785;
        sum = sum + a[i+(51*height)] * 0.0024896574879993603;
        sum = sum + a[i+(52*height)] * 0.0023179299832409824;
        sum = sum + a[i+(53*height)] * 0.0021550548007040924;
        sum = sum + a[i+(54*height)] * 0.00200084579371474;
        sum = sum + a[i+(55*height)] * 0.0018550952537454528;
        sum = sum + a[i+(56*height)] * 0.0017175765627405672;
        sum = sum + a[i+(57*height)] * 0.0015880467803052096;
        sum = sum + a[i+(58*height)] * 0.0014662491497820617;
        sum = sum + a[i+(59*height)] * 0.001351915509170473;
        sum = sum + a[i+(60*height)] * 0.0012447685947797947;
        sum = sum + a[i+(61*height)] * 0.00114452422742914;
        sum = sum + a[i+(62*height)] * 0.0010508933728874807;
        sum = sum + a[i+(63*height)] * 0.0009635840700714965;
    }
    if (vertical && (63 < i) && (i < (n-63))) {
       // Vertical convolution
        sum = sum + a[i+(-63)] * 0.0009635840700714965;
        sum = sum + a[i+(-62)] * 0.0010508933728874807;
        sum = sum + a[i+(-61)] * 0.00114452422742914;
        sum = sum + a[i+(-60)] * 0.0012447685947797947;
        sum = sum + a[i+(-59)] * 0.001351915509170473;
        sum = sum + a[i+(-58)] * 0.0014662491497820617;
        sum = sum + a[i+(-57)] * 0.0015880467803052096;
        sum = sum + a[i+(-56)] * 0.0017175765627405672;
        sum = sum + a[i+(-55)] * 0.0018550952537454528;
        sum = sum + a[i+(-54)] * 0.00200084579371474;
        sum = sum + a[i+(-53)] * 0.0021550548007040924;
        sum = sum + a[i+(-52)] * 0.0023179299832409824;
        sum = sum + a[i+(-51)] * 0.0024896574879993603;
        sum = sum + a[i+(-50)] * 0.0026703992002119785;
        sum = sum + a[i+(-49)] * 0.0028602900165330338;
        sum = sum + a[i+(-48)] * 0.0030594351118147342;
        sum = sum + a[i+(-47)] * 0.0032679072228954287;
        sum = sum + a[i+(-46)] * 0.0034857439739843137;
        sum = sum + a[i+(-45)] * 0.003712945269538897;
        sum = sum + a[i+(-44)] * 0.00394947078163683;
        sum = sum + a[i+(-43)] * 0.00419523755971528;
        sum = sum + a[i+(-42)] * 0.004450117791161477;
        sum = sum + a[i+(-41)] * 0.004713936741562557;
        sum = sum + a[i+(-40)] * 0.004986470903438759;
        sum = sum + a[i+(-39)] * 0.005267446381971586;
        sum = sum + a[i+(-38)] * 0.005556537545581941;
        sum = sum + a[i+(-37)] * 0.00585336596819982;
        sum = sum + a[i+(-36)] * 0.006157499688689468;
        sum = sum + a[i+(-35)] * 0.006468452811148505;
        sum = sum + a[i+(-34)] * 0.006785685467688546;
        sum = sum + a[i+(-33)] * 0.0071086041628354345;
        sum = sum + a[i+(-32)] * 0.00743656251587208;
        sum = sum + a[i+(-31)] * 0.007768862414304341;
        sum = sum + a[i+(-30)] * 0.008104755588184343;
        sum = sum + a[i+(-29)] * 0.008443445611305217;
        sum = sum + a[i+(-28)] * 0.00878409033132127;
        sum = sum + a[i+(-27)] * 0.00912580472668759;
        sum = sum + a[i+(-26)] * 0.009467664183997209;
        sum = sum + a[i+(-25)] * 0.009808708184870377;
        sum = sum + a[i+(-24)] * 0.010147944387071587;
        sum = sum + a[i+(-23)] * 0.010484353080049538;
        sum = sum + a[i+(-22)] * 0.010816891990671151;
        sum = sum + a[i+(-21)] * 0.011144501410610371;
        sum = sum + a[i+(-20)] * 0.011466109612714988;
        sum = sum + a[i+(-19)] * 0.011780638519768178;
        sum = sum + a[i+(-18)] * 0.012087009585442916;
        sum = sum + a[i+(-17)] * 0.0123841498439715;
        sum = sum + a[i+(-16)] * 0.0126709980821707;
        sum = sum + a[i+(-15)] * 0.012946511085023192;
        sum = sum + a[i+(-14)] * 0.01320966990406012;
        sum = sum + a[i+(-13)] * 0.013459486096355404;
        sum = sum + a[i+(-12)] * 0.01369500788105957;
        sum = sum + a[i+(-11)] * 0.013915326160093642;
        sum = sum + a[i+(-10)] * 0.014119580349907273;
        sum = sum + a[i+(-9)] * 0.014306963972088468;
        sum = sum + a[i+(-8)] * 0.014476729952094189;
        sum = sum + a[i+(-7)] * 0.014628195577443737;
        sum = sum + a[i+(-6)] * 0.014760747069362649;
        sum = sum + a[i+(-5)] * 0.01487384372505926;
        sum = sum + a[i+(-4)] * 0.014967021591525414;
        sum = sum + a[i+(-3)] * 0.015039896635936488;
        sum = sum + a[i+(-2)] * 0.015092167382336447;
        sum = sum + a[i+(-1)] * 0.015123616989276317;
        sum = sum + a[i+(0)] * 0.01513411474837013;
        sum = sum + a[i+(1)] * 0.015123616989276317;
        sum = sum + a[i+(2)] * 0.015092167382336447;
        sum = sum + a[i+(3)] * 0.015039896635936488;
        sum = sum + a[i+(4)] * 0.014967021591525414;
        sum = sum + a[i+(5)] * 0.01487384372505926;
        sum = sum + a[i+(6)] * 0.014760747069362649;
        sum = sum + a[i+(7)] * 0.014628195577443737;
        sum = sum + a[i+(8)] * 0.014476729952094189;
        sum = sum + a[i+(9)] * 0.014306963972088468;
        sum = sum + a[i+(10)] * 0.014119580349907273;
        sum = sum + a[i+(11)] * 0.013915326160093642;
        sum = sum + a[i+(12)] * 0.01369500788105957;
        sum = sum + a[i+(13)] * 0.013459486096355404;
        sum = sum + a[i+(14)] * 0.01320966990406012;
        sum = sum + a[i+(15)] * 0.012946511085023192;
        sum = sum + a[i+(16)] * 0.0126709980821707;
        sum = sum + a[i+(17)] * 0.0123841498439715;
        sum = sum + a[i+(18)] * 0.012087009585442916;
        sum = sum + a[i+(19)] * 0.011780638519768178;
        sum = sum + a[i+(20)] * 0.011466109612714988;
        sum = sum + a[i+(21)] * 0.011144501410610371;
        sum = sum + a[i+(22)] * 0.010816891990671151;
        sum = sum + a[i+(23)] * 0.010484353080049538;
        sum = sum + a[i+(24)] * 0.010147944387071587;
        sum = sum + a[i+(25)] * 0.009808708184870377;
        sum = sum + a[i+(26)] * 0.009467664183997209;
        sum = sum + a[i+(27)] * 0.00912580472668759;
        sum = sum + a[i+(28)] * 0.00878409033132127;
        sum = sum + a[i+(29)] * 0.008443445611305217;
        sum = sum + a[i+(30)] * 0.008104755588184343;
        sum = sum + a[i+(31)] * 0.007768862414304341;
        sum = sum + a[i+(32)] * 0.00743656251587208;
        sum = sum + a[i+(33)] * 0.0071086041628354345;
        sum = sum + a[i+(34)] * 0.006785685467688546;
        sum = sum + a[i+(35)] * 0.006468452811148505;
        sum = sum + a[i+(36)] * 0.006157499688689468;
        sum = sum + a[i+(37)] * 0.00585336596819982;
        sum = sum + a[i+(38)] * 0.005556537545581941;
        sum = sum + a[i+(39)] * 0.005267446381971586;
        sum = sum + a[i+(40)] * 0.004986470903438759;
        sum = sum + a[i+(41)] * 0.004713936741562557;
        sum = sum + a[i+(42)] * 0.004450117791161477;
        sum = sum + a[i+(43)] * 0.00419523755971528;
        sum = sum + a[i+(44)] * 0.00394947078163683;
        sum = sum + a[i+(45)] * 0.003712945269538897;
        sum = sum + a[i+(46)] * 0.0034857439739843137;
        sum = sum + a[i+(47)] * 0.0032679072228954287;
        sum = sum + a[i+(48)] * 0.0030594351118147342;
        sum = sum + a[i+(49)] * 0.0028602900165330338;
        sum = sum + a[i+(50)] * 0.0026703992002119785;
        sum = sum + a[i+(51)] * 0.0024896574879993603;
        sum = sum + a[i+(52)] * 0.0023179299832409824;
        sum = sum + a[i+(53)] * 0.0021550548007040924;
        sum = sum + a[i+(54)] * 0.00200084579371474;
        sum = sum + a[i+(55)] * 0.0018550952537454528;
        sum = sum + a[i+(56)] * 0.0017175765627405672;
        sum = sum + a[i+(57)] * 0.0015880467803052096;
        sum = sum + a[i+(58)] * 0.0014662491497820617;
        sum = sum + a[i+(59)] * 0.001351915509170473;
        sum = sum + a[i+(60)] * 0.0012447685947797947;
        sum = sum + a[i+(61)] * 0.00114452422742914;
        sum = sum + a[i+(62)] * 0.0010508933728874807;
        sum = sum + a[i+(63)] * 0.0009635840700714965;
    }
    ret[i] = sum;
}

kernel void gauss127_h(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    float sum = 0.0f;
    int x = X(i);
    int y = Y(i);
    // Horizontal convolution
    int inside = (x >= 63) && (x<width-63) && (y >= 63) && (y < height-63);
    //int inside = (((size_t)(63*height) < i) && (i < (n-63*height)));
    sum += inside ? a[i+(-63*height)] * 0.0009635840700714965 : 0.0f;
    sum += inside ? a[i+(-62*height)] * 0.0010508933728874807 : 0.0f;
    sum += inside ? a[i+(-61*height)] * 0.00114452422742914 : 0.0f;
    sum += inside ? a[i+(-60*height)] * 0.0012447685947797947 : 0.0f;
    sum += inside ? a[i+(-59*height)] * 0.001351915509170473 : 0.0f;
    sum += inside ? a[i+(-58*height)] * 0.0014662491497820617 : 0.0f;
    sum += inside ? a[i+(-57*height)] * 0.0015880467803052096 : 0.0f;
    sum += inside ? a[i+(-56*height)] * 0.0017175765627405672 : 0.0f;
    sum += inside ? a[i+(-55*height)] * 0.0018550952537454528 : 0.0f;
    sum += inside ? a[i+(-54*height)] * 0.00200084579371474 : 0.0f;
    sum += inside ? a[i+(-53*height)] * 0.0021550548007040924 : 0.0f;
    sum += inside ? a[i+(-52*height)] * 0.0023179299832409824 : 0.0f;
    sum += inside ? a[i+(-51*height)] * 0.0024896574879993603 : 0.0f;
    sum += inside ? a[i+(-50*height)] * 0.0026703992002119785 : 0.0f;
    sum += inside ? a[i+(-49*height)] * 0.0028602900165330338 : 0.0f;
    sum += inside ? a[i+(-48*height)] * 0.0030594351118147342 : 0.0f;
    sum += inside ? a[i+(-47*height)] * 0.0032679072228954287 : 0.0f;
    sum += inside ? a[i+(-46*height)] * 0.0034857439739843137 : 0.0f;
    sum += inside ? a[i+(-45*height)] * 0.003712945269538897 : 0.0f;
    sum += inside ? a[i+(-44*height)] * 0.00394947078163683 : 0.0f;
    sum += inside ? a[i+(-43*height)] * 0.00419523755971528 : 0.0f;
    sum += inside ? a[i+(-42*height)] * 0.004450117791161477 : 0.0f;
    sum += inside ? a[i+(-41*height)] * 0.004713936741562557 : 0.0f;
    sum += inside ? a[i+(-40*height)] * 0.004986470903438759 : 0.0f;
    sum += inside ? a[i+(-39*height)] * 0.005267446381971586 : 0.0f;
    sum += inside ? a[i+(-38*height)] * 0.005556537545581941 : 0.0f;
    sum += inside ? a[i+(-37*height)] * 0.00585336596819982 : 0.0f;
    sum += inside ? a[i+(-36*height)] * 0.006157499688689468 : 0.0f;
    sum += inside ? a[i+(-35*height)] * 0.006468452811148505 : 0.0f;
    sum += inside ? a[i+(-34*height)] * 0.006785685467688546 : 0.0f;
    sum += inside ? a[i+(-33*height)] * 0.0071086041628354345 : 0.0f;
    sum += inside ? a[i+(-32*height)] * 0.00743656251587208 : 0.0f;
    sum += inside ? a[i+(-31*height)] * 0.007768862414304341 : 0.0f;
    sum += inside ? a[i+(-30*height)] * 0.008104755588184343 : 0.0f;
    sum += inside ? a[i+(-29*height)] * 0.008443445611305217 : 0.0f;
    sum += inside ? a[i+(-28*height)] * 0.00878409033132127 : 0.0f;
    sum += inside ? a[i+(-27*height)] * 0.00912580472668759 : 0.0f;
    sum += inside ? a[i+(-26*height)] * 0.009467664183997209 : 0.0f;
    sum += inside ? a[i+(-25*height)] * 0.009808708184870377 : 0.0f;
    sum += inside ? a[i+(-24*height)] * 0.010147944387071587 : 0.0f;
    sum += inside ? a[i+(-23*height)] * 0.010484353080049538 : 0.0f;
    sum += inside ? a[i+(-22*height)] * 0.010816891990671151 : 0.0f;
    sum += inside ? a[i+(-21*height)] * 0.011144501410610371 : 0.0f;
    sum += inside ? a[i+(-20*height)] * 0.011466109612714988 : 0.0f;
    sum += inside ? a[i+(-19*height)] * 0.011780638519768178 : 0.0f;
    sum += inside ? a[i+(-18*height)] * 0.012087009585442916 : 0.0f;
    sum += inside ? a[i+(-17*height)] * 0.0123841498439715 : 0.0f;
    sum += inside ? a[i+(-16*height)] * 0.0126709980821707 : 0.0f;
    sum += inside ? a[i+(-15*height)] * 0.012946511085023192 : 0.0f;
    sum += inside ? a[i+(-14*height)] * 0.01320966990406012 : 0.0f;
    sum += inside ? a[i+(-13*height)] * 0.013459486096355404 : 0.0f;
    sum += inside ? a[i+(-12*height)] * 0.01369500788105957 : 0.0f;
    sum += inside ? a[i+(-11*height)] * 0.013915326160093642 : 0.0f;
    sum += inside ? a[i+(-10*height)] * 0.014119580349907273 : 0.0f;
    sum += inside ? a[i+(-9*height)] * 0.014306963972088468 : 0.0f;
    sum += inside ? a[i+(-8*height)] * 0.014476729952094189 : 0.0f;
    sum += inside ? a[i+(-7*height)] * 0.014628195577443737 : 0.0f;
    sum += inside ? a[i+(-6*height)] * 0.014760747069362649 : 0.0f;
    sum += inside ? a[i+(-5*height)] * 0.01487384372505926 : 0.0f;
    sum += inside ? a[i+(-4*height)] * 0.014967021591525414 : 0.0f;
    sum += inside ? a[i+(-3*height)] * 0.015039896635936488 : 0.0f;
    sum += inside ? a[i+(-2*height)] * 0.015092167382336447 : 0.0f;
    sum += inside ? a[i+(-1*height)] * 0.015123616989276317 : 0.0f;
    sum += inside ? a[i+(0*height)] * 0.01513411474837013 : 0.0f;
    sum += inside ? a[i+(1*height)] * 0.015123616989276317 : 0.0f;
    sum += inside ? a[i+(2*height)] * 0.015092167382336447 : 0.0f;
    sum += inside ? a[i+(3*height)] * 0.015039896635936488 : 0.0f;
    sum += inside ? a[i+(4*height)] * 0.014967021591525414 : 0.0f;
    sum += inside ? a[i+(5*height)] * 0.01487384372505926 : 0.0f;
    sum += inside ? a[i+(6*height)] * 0.014760747069362649 : 0.0f;
    sum += inside ? a[i+(7*height)] * 0.014628195577443737 : 0.0f;
    sum += inside ? a[i+(8*height)] * 0.014476729952094189 : 0.0f;
    sum += inside ? a[i+(9*height)] * 0.014306963972088468 : 0.0f;
    sum += inside ? a[i+(10*height)] * 0.014119580349907273 : 0.0f;
    sum += inside ? a[i+(11*height)] * 0.013915326160093642 : 0.0f;
    sum += inside ? a[i+(12*height)] * 0.01369500788105957 : 0.0f;
    sum += inside ? a[i+(13*height)] * 0.013459486096355404 : 0.0f;
    sum += inside ? a[i+(14*height)] * 0.01320966990406012 : 0.0f;
    sum += inside ? a[i+(15*height)] * 0.012946511085023192 : 0.0f;
    sum += inside ? a[i+(16*height)] * 0.0126709980821707 : 0.0f;
    sum += inside ? a[i+(17*height)] * 0.0123841498439715 : 0.0f;
    sum += inside ? a[i+(18*height)] * 0.012087009585442916 : 0.0f;
    sum += inside ? a[i+(19*height)] * 0.011780638519768178 : 0.0f;
    sum += inside ? a[i+(20*height)] * 0.011466109612714988 : 0.0f;
    sum += inside ? a[i+(21*height)] * 0.011144501410610371 : 0.0f;
    sum += inside ? a[i+(22*height)] * 0.010816891990671151 : 0.0f;
    sum += inside ? a[i+(23*height)] * 0.010484353080049538 : 0.0f;
    sum += inside ? a[i+(24*height)] * 0.010147944387071587 : 0.0f;
    sum += inside ? a[i+(25*height)] * 0.009808708184870377 : 0.0f;
    sum += inside ? a[i+(26*height)] * 0.009467664183997209 : 0.0f;
    sum += inside ? a[i+(27*height)] * 0.00912580472668759 : 0.0f;
    sum += inside ? a[i+(28*height)] * 0.00878409033132127 : 0.0f;
    sum += inside ? a[i+(29*height)] * 0.008443445611305217 : 0.0f;
    sum += inside ? a[i+(30*height)] * 0.008104755588184343 : 0.0f;
    sum += inside ? a[i+(31*height)] * 0.007768862414304341 : 0.0f;
    sum += inside ? a[i+(32*height)] * 0.00743656251587208 : 0.0f;
    sum += inside ? a[i+(33*height)] * 0.0071086041628354345 : 0.0f;
    sum += inside ? a[i+(34*height)] * 0.006785685467688546 : 0.0f;
    sum += inside ? a[i+(35*height)] * 0.006468452811148505 : 0.0f;
    sum += inside ? a[i+(36*height)] * 0.006157499688689468 : 0.0f;
    sum += inside ? a[i+(37*height)] * 0.00585336596819982 : 0.0f;
    sum += inside ? a[i+(38*height)] * 0.005556537545581941 : 0.0f;
    sum += inside ? a[i+(39*height)] * 0.005267446381971586 : 0.0f;
    sum += inside ? a[i+(40*height)] * 0.004986470903438759 : 0.0f;
    sum += inside ? a[i+(41*height)] * 0.004713936741562557 : 0.0f;
    sum += inside ? a[i+(42*height)] * 0.004450117791161477 : 0.0f;
    sum += inside ? a[i+(43*height)] * 0.00419523755971528 : 0.0f;
    sum += inside ? a[i+(44*height)] * 0.00394947078163683 : 0.0f;
    sum += inside ? a[i+(45*height)] * 0.003712945269538897 : 0.0f;
    sum += inside ? a[i+(46*height)] * 0.0034857439739843137 : 0.0f;
    sum += inside ? a[i+(47*height)] * 0.0032679072228954287 : 0.0f;
    sum += inside ? a[i+(48*height)] * 0.0030594351118147342 : 0.0f;
    sum += inside ? a[i+(49*height)] * 0.0028602900165330338 : 0.0f;
    sum += inside ? a[i+(50*height)] * 0.0026703992002119785 : 0.0f;
    sum += inside ? a[i+(51*height)] * 0.0024896574879993603 : 0.0f;
    sum += inside ? a[i+(52*height)] * 0.0023179299832409824 : 0.0f;
    sum += inside ? a[i+(53*height)] * 0.0021550548007040924 : 0.0f;
    sum += inside ? a[i+(54*height)] * 0.00200084579371474 : 0.0f;
    sum += inside ? a[i+(55*height)] * 0.0018550952537454528 : 0.0f;
    sum += inside ? a[i+(56*height)] * 0.0017175765627405672 : 0.0f;
    sum += inside ? a[i+(57*height)] * 0.0015880467803052096 : 0.0f;
    sum += inside ? a[i+(58*height)] * 0.0014662491497820617 : 0.0f;
    sum += inside ? a[i+(59*height)] * 0.001351915509170473 : 0.0f;
    sum += inside ? a[i+(60*height)] * 0.0012447685947797947 : 0.0f;
    sum += inside ? a[i+(61*height)] * 0.00114452422742914 : 0.0f;
    sum += inside ? a[i+(62*height)] * 0.0010508933728874807 : 0.0f;
    sum += inside ? a[i+(63*height)] * 0.0009635840700714965 : 0.0f;
    ret[i] = sum;
}

kernel void gauss127_v(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    size_t n = width*height;
    int x = X(i);
    int y = Y(i);
    // Vertical convolution
    float sum = 0.0f;
    int inside = (x >= 63) && (x<width-63) && (y >= 63) && (y < height-63);
    //int inside = (63 < i) && (i < (n-63));
    sum +=  inside ?  a[i+(-63)] * 0.0009635840700714965 : 0.0f;
    sum +=  inside ?  a[i+(-62)] * 0.0010508933728874807 : 0.0f;
    sum +=  inside ?  a[i+(-61)] * 0.00114452422742914 : 0.0f;
    sum +=  inside ?  a[i+(-60)] * 0.0012447685947797947 : 0.0f;
    sum +=  inside ?  a[i+(-59)] * 0.001351915509170473 : 0.0f;
    sum +=  inside ?  a[i+(-58)] * 0.0014662491497820617 : 0.0f;
    sum +=  inside ?  a[i+(-57)] * 0.0015880467803052096 : 0.0f;
    sum +=  inside ?  a[i+(-56)] * 0.0017175765627405672 : 0.0f;
    sum +=  inside ?  a[i+(-55)] * 0.0018550952537454528 : 0.0f;
    sum +=  inside ?  a[i+(-54)] * 0.00200084579371474 : 0.0f;
    sum +=  inside ?  a[i+(-53)] * 0.0021550548007040924 : 0.0f;
    sum +=  inside ?  a[i+(-52)] * 0.0023179299832409824 : 0.0f;
    sum +=  inside ?  a[i+(-51)] * 0.0024896574879993603 : 0.0f;
    sum +=  inside ?  a[i+(-50)] * 0.0026703992002119785 : 0.0f;
    sum +=  inside ?  a[i+(-49)] * 0.0028602900165330338 : 0.0f;
    sum +=  inside ?  a[i+(-48)] * 0.0030594351118147342 : 0.0f;
    sum +=  inside ?  a[i+(-47)] * 0.0032679072228954287 : 0.0f;
    sum +=  inside ?  a[i+(-46)] * 0.0034857439739843137 : 0.0f;
    sum +=  inside ?  a[i+(-45)] * 0.003712945269538897 : 0.0f;
    sum +=  inside ?  a[i+(-44)] * 0.00394947078163683 : 0.0f;
    sum +=  inside ?  a[i+(-43)] * 0.00419523755971528 : 0.0f;
    sum +=  inside ?  a[i+(-42)] * 0.004450117791161477 : 0.0f;
    sum +=  inside ?  a[i+(-41)] * 0.004713936741562557 : 0.0f;
    sum +=  inside ?  a[i+(-40)] * 0.004986470903438759 : 0.0f;
    sum +=  inside ?  a[i+(-39)] * 0.005267446381971586 : 0.0f;
    sum +=  inside ?  a[i+(-38)] * 0.005556537545581941 : 0.0f;
    sum +=  inside ?  a[i+(-37)] * 0.00585336596819982 : 0.0f;
    sum +=  inside ?  a[i+(-36)] * 0.006157499688689468 : 0.0f;
    sum +=  inside ?  a[i+(-35)] * 0.006468452811148505 : 0.0f;
    sum +=  inside ?  a[i+(-34)] * 0.006785685467688546 : 0.0f;
    sum +=  inside ?  a[i+(-33)] * 0.0071086041628354345 : 0.0f;
    sum +=  inside ?  a[i+(-32)] * 0.00743656251587208 : 0.0f;
    sum +=  inside ?  a[i+(-31)] * 0.007768862414304341 : 0.0f;
    sum +=  inside ?  a[i+(-30)] * 0.008104755588184343 : 0.0f;
    sum +=  inside ?  a[i+(-29)] * 0.008443445611305217 : 0.0f;
    sum +=  inside ?  a[i+(-28)] * 0.00878409033132127 : 0.0f;
    sum +=  inside ?  a[i+(-27)] * 0.00912580472668759 : 0.0f;
    sum +=  inside ?  a[i+(-26)] * 0.009467664183997209 : 0.0f;
    sum +=  inside ?  a[i+(-25)] * 0.009808708184870377 : 0.0f;
    sum +=  inside ?  a[i+(-24)] * 0.010147944387071587 : 0.0f;
    sum +=  inside ?  a[i+(-23)] * 0.010484353080049538 : 0.0f;
    sum +=  inside ?  a[i+(-22)] * 0.010816891990671151 : 0.0f;
    sum +=  inside ?  a[i+(-21)] * 0.011144501410610371 : 0.0f;
    sum +=  inside ?  a[i+(-20)] * 0.011466109612714988 : 0.0f;
    sum +=  inside ?  a[i+(-19)] * 0.011780638519768178 : 0.0f;
    sum +=  inside ?  a[i+(-18)] * 0.012087009585442916 : 0.0f;
    sum +=  inside ?  a[i+(-17)] * 0.0123841498439715 : 0.0f;
    sum +=  inside ?  a[i+(-16)] * 0.0126709980821707 : 0.0f;
    sum +=  inside ?  a[i+(-15)] * 0.012946511085023192 : 0.0f;
    sum +=  inside ?  a[i+(-14)] * 0.01320966990406012 : 0.0f;
    sum +=  inside ?  a[i+(-13)] * 0.013459486096355404 : 0.0f;
    sum +=  inside ?  a[i+(-12)] * 0.01369500788105957 : 0.0f;
    sum +=  inside ?  a[i+(-11)] * 0.013915326160093642 : 0.0f;
    sum +=  inside ?  a[i+(-10)] * 0.014119580349907273 : 0.0f;
    sum +=  inside ?  a[i+(-9)] * 0.014306963972088468 : 0.0f;
    sum +=  inside ?  a[i+(-8)] * 0.014476729952094189 : 0.0f;
    sum +=  inside ?  a[i+(-7)] * 0.014628195577443737 : 0.0f;
    sum +=  inside ?  a[i+(-6)] * 0.014760747069362649 : 0.0f;
    sum +=  inside ?  a[i+(-5)] * 0.01487384372505926 : 0.0f;
    sum +=  inside ?  a[i+(-4)] * 0.014967021591525414 : 0.0f;
    sum +=  inside ?  a[i+(-3)] * 0.015039896635936488 : 0.0f;
    sum +=  inside ?  a[i+(-2)] * 0.015092167382336447 : 0.0f;
    sum +=  inside ?  a[i+(-1)] * 0.015123616989276317 : 0.0f;
    sum +=  inside ?  a[i+(0)] * 0.01513411474837013 : 0.0f;
    sum +=  inside ?  a[i+(1)] * 0.015123616989276317 : 0.0f;
    sum +=  inside ?  a[i+(2)] * 0.015092167382336447 : 0.0f;
    sum +=  inside ?  a[i+(3)] * 0.015039896635936488 : 0.0f;
    sum +=  inside ?  a[i+(4)] * 0.014967021591525414 : 0.0f;
    sum +=  inside ?  a[i+(5)] * 0.01487384372505926 : 0.0f;
    sum +=  inside ?  a[i+(6)] * 0.014760747069362649 : 0.0f;
    sum +=  inside ?  a[i+(7)] * 0.014628195577443737 : 0.0f;
    sum +=  inside ?  a[i+(8)] * 0.014476729952094189 : 0.0f;
    sum +=  inside ?  a[i+(9)] * 0.014306963972088468 : 0.0f;
    sum +=  inside ?  a[i+(10)] * 0.014119580349907273 : 0.0f;
    sum +=  inside ?  a[i+(11)] * 0.013915326160093642 : 0.0f;
    sum +=  inside ?  a[i+(12)] * 0.01369500788105957 : 0.0f;
    sum +=  inside ?  a[i+(13)] * 0.013459486096355404 : 0.0f;
    sum +=  inside ?  a[i+(14)] * 0.01320966990406012 : 0.0f;
    sum +=  inside ?  a[i+(15)] * 0.012946511085023192 : 0.0f;
    sum +=  inside ?  a[i+(16)] * 0.0126709980821707 : 0.0f;
    sum +=  inside ?  a[i+(17)] * 0.0123841498439715 : 0.0f;
    sum +=  inside ?  a[i+(18)] * 0.012087009585442916 : 0.0f;
    sum +=  inside ?  a[i+(19)] * 0.011780638519768178 : 0.0f;
    sum +=  inside ?  a[i+(20)] * 0.011466109612714988 : 0.0f;
    sum +=  inside ?  a[i+(21)] * 0.011144501410610371 : 0.0f;
    sum +=  inside ?  a[i+(22)] * 0.010816891990671151 : 0.0f;
    sum +=  inside ?  a[i+(23)] * 0.010484353080049538 : 0.0f;
    sum +=  inside ?  a[i+(24)] * 0.010147944387071587 : 0.0f;
    sum +=  inside ?  a[i+(25)] * 0.009808708184870377 : 0.0f;
    sum +=  inside ?  a[i+(26)] * 0.009467664183997209 : 0.0f;
    sum +=  inside ?  a[i+(27)] * 0.00912580472668759 : 0.0f;
    sum +=  inside ?  a[i+(28)] * 0.00878409033132127 : 0.0f;
    sum +=  inside ?  a[i+(29)] * 0.008443445611305217 : 0.0f;
    sum +=  inside ?  a[i+(30)] * 0.008104755588184343 : 0.0f;
    sum +=  inside ?  a[i+(31)] * 0.007768862414304341 : 0.0f;
    sum +=  inside ?  a[i+(32)] * 0.00743656251587208 : 0.0f;
    sum +=  inside ?  a[i+(33)] * 0.0071086041628354345 : 0.0f;
    sum +=  inside ?  a[i+(34)] * 0.006785685467688546 : 0.0f;
    sum +=  inside ?  a[i+(35)] * 0.006468452811148505 : 0.0f;
    sum +=  inside ?  a[i+(36)] * 0.006157499688689468 : 0.0f;
    sum +=  inside ?  a[i+(37)] * 0.00585336596819982 : 0.0f;
    sum +=  inside ?  a[i+(38)] * 0.005556537545581941 : 0.0f;
    sum +=  inside ?  a[i+(39)] * 0.005267446381971586 : 0.0f;
    sum +=  inside ?  a[i+(40)] * 0.004986470903438759 : 0.0f;
    sum +=  inside ?  a[i+(41)] * 0.004713936741562557 : 0.0f;
    sum +=  inside ?  a[i+(42)] * 0.004450117791161477 : 0.0f;
    sum +=  inside ?  a[i+(43)] * 0.00419523755971528 : 0.0f;
    sum +=  inside ?  a[i+(44)] * 0.00394947078163683 : 0.0f;
    sum +=  inside ?  a[i+(45)] * 0.003712945269538897 : 0.0f;
    sum +=  inside ?  a[i+(46)] * 0.0034857439739843137 : 0.0f;
    sum +=  inside ?  a[i+(47)] * 0.0032679072228954287 : 0.0f;
    sum +=  inside ?  a[i+(48)] * 0.0030594351118147342 : 0.0f;
    sum +=  inside ?  a[i+(49)] * 0.0028602900165330338 : 0.0f;
    sum +=  inside ?  a[i+(50)] * 0.0026703992002119785 : 0.0f;
    sum +=  inside ?  a[i+(51)] * 0.0024896574879993603 : 0.0f;
    sum +=  inside ?  a[i+(52)] * 0.0023179299832409824 : 0.0f;
    sum +=  inside ?  a[i+(53)] * 0.0021550548007040924 : 0.0f;
    sum +=  inside ?  a[i+(54)] * 0.00200084579371474 : 0.0f;
    sum +=  inside ?  a[i+(55)] * 0.0018550952537454528 : 0.0f;
    sum +=  inside ?  a[i+(56)] * 0.0017175765627405672 : 0.0f;
    sum +=  inside ?  a[i+(57)] * 0.0015880467803052096 : 0.0f;
    sum +=  inside ?  a[i+(58)] * 0.0014662491497820617 : 0.0f;
    sum +=  inside ?  a[i+(59)] * 0.001351915509170473 : 0.0f;
    sum +=  inside ?  a[i+(60)] * 0.0012447685947797947 : 0.0f;
    sum +=  inside ?  a[i+(61)] * 0.00114452422742914 : 0.0f;
    sum +=  inside ?  a[i+(62)] * 0.0010508933728874807 : 0.0f;
    sum +=  inside ?  a[i+(63)] * 0.0009635840700714965 : 0.0f;

    ret[i] = sum;
}
