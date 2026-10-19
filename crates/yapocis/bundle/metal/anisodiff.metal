/* Perona-Malik filter (Metal) */

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


/* Perona-Malik filter for 1,2,3,4 channel images*/

kernel void filterImage(constant int &width [[ buffer(0) ]],
                                 constant int &height [[ buffer(1) ]],
                                 const device FLOAT *input [[ buffer(2) ]],
                                 constant float &scale [[ buffer(3) ]],
                                 constant float &step_size [[ buffer(4) ]],
                                 device FLOAT *output [[ buffer(5) ]],
                                 uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT CF;
    FLOAT central = input[i];
    FLOAT invscale = 1.0f/scale;
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      FLOAT n = input[i+1];
      FLOAT s = input[i-1];
      FLOAT e = input[i-height];
      FLOAT w = input[i+height];
      FLOAT di = n-central;
      CF = di*invscale;
      CF = ONE/(ONE + (CF * CF));
      FLOAT accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = ONE/(ONE + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = ONE/(ONE + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = ONE/(ONE + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
}

kernel void filterImage2(constant int &width [[ buffer(0) ]],
                                 constant int &height [[ buffer(1) ]],
                                 const device FLOAT2 *input [[ buffer(2) ]],
                                 constant float &scale [[ buffer(3) ]],
                                 constant float &step_size [[ buffer(4) ]],
                                 device FLOAT2 *output [[ buffer(5) ]],
                                 uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT2 CF;
    FLOAT2 central = input[i];
    FLOAT2 invscale = 1.0f/scale;
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      FLOAT2 n = input[i+1];
      FLOAT2 s = input[i-1];
      FLOAT2 e = input[i-height];
      FLOAT2 w = input[i+height];
      FLOAT2 di = n-central;
      CF = di*invscale;
      CF = ONE2/(ONE2 + (CF * CF));
      FLOAT2 accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = ONE2/(ONE2 + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = ONE2/(ONE2 + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = ONE2/(ONE2 + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
}

kernel void filterImage3(constant int &width [[ buffer(0) ]],
                                 constant int &height [[ buffer(1) ]],
                                 const device FLOAT3 *input [[ buffer(2) ]],
                                 constant float &scale [[ buffer(3) ]],
                                 constant float &step_size [[ buffer(4) ]],
                                 device FLOAT3 *output [[ buffer(5) ]],
                                 uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT3 CF;
    FLOAT3 central = input[i];
    FLOAT3 invscale = 1.0f/scale;
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      FLOAT3 n = input[i+1];
      FLOAT3 s = input[i-1];
      FLOAT3 e = input[i-height];
      FLOAT3 w = input[i+height];
      FLOAT3 di = n-central;
      CF = di*invscale;
      CF = ONE3/(ONE3 + (CF * CF));
      FLOAT3 accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = ONE3/(ONE3 + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = ONE3/(ONE3 + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = ONE3/(ONE3 + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
}

kernel void filterImage4(constant int &width [[ buffer(0) ]],
                                 constant int &height [[ buffer(1) ]],
                                 const device FLOAT4 *input [[ buffer(2) ]],
                                 constant float &scale [[ buffer(3) ]],
                                 constant float &step_size [[ buffer(4) ]],
                                 device FLOAT4 *output [[ buffer(5) ]],
                                 uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT4 CF;
    FLOAT4 central = input[i];
    FLOAT4 invscale = 1.0f/scale;
    if ((0<x) && (x<(width-1)) && ((0<y) && (y<(height-1)))) {
      FLOAT4 n = input[i+1];
      FLOAT4 s = input[i-1];
      FLOAT4 e = input[i-height];
      FLOAT4 w = input[i+height];
      FLOAT4 di = n-central;
      CF = di*invscale;
      CF = ONE4/(ONE4 + (CF * CF));
      FLOAT4 accumulator = di * CF;
      di = s-central;
      CF = (di*invscale);
      CF = ONE4/(ONE4 + (CF * CF));
      accumulator += di * CF;
      di = e-central;
      CF = (di*invscale);
      CF = ONE4/(ONE4 + (CF * CF));
      accumulator += di * CF;
      di = w-central;
      CF = (di*invscale);
      CF = ONE4/(ONE4 + (CF * CF));
      accumulator += di * CF;
      accumulator *= step_size;
      central += accumulator;
    }
    output[i] = central;
}
