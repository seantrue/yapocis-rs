/* Metal kernels to compute direction and magnitude of gradient of an image. */

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


/* Compute gradient for images with 1,2,3,4 channels per pixel. */

kernel void gradient(constant int &width [[ buffer(0) ]],
                              constant int &height [[ buffer(1) ]],
                              const device FLOAT* input [[ buffer(2) ]],
                              constant int &reach [[ buffer(3) ]],
                              device FLOAT* grad [[ buffer(4) ]],
                              device FLOAT* theta [[ buffer(5) ]],
                              uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT gr = ZERO;
    FLOAT th = ZERO;
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FLOAT dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        th = atan2pi(dy, dx);
    }
    grad[i] = select(gr, ZERO, isnan(gr));
    theta[i] = select(th, ZERO, isnan(th));
}
kernel void gradient2(constant int &width [[ buffer(0) ]],
                              constant int &height [[ buffer(1) ]],
                              const device FLOAT2* input [[ buffer(2) ]],
                              constant int &reach [[ buffer(3) ]],
                              device FLOAT2* grad [[ buffer(4) ]],
                              device FLOAT2* theta [[ buffer(5) ]],
                              uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT2 gr = ZERO2;
    FLOAT2 th = ZERO2;
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FLOAT2 dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT2 dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        th = atan2pi(dy, dx);
    }
    grad[i] = select(gr, ZERO2, isnan(gr));
    theta[i] = select(th, ZERO2, isnan(th));
}
kernel void gradient3(constant int &width [[ buffer(0) ]],
                              constant int &height [[ buffer(1) ]],
                              const device FLOAT3* input [[ buffer(2) ]],
                              constant int &reach [[ buffer(3) ]],
                              device FLOAT3* grad [[ buffer(4) ]],
                              device FLOAT3* theta [[ buffer(5) ]],
                              uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT3 gr = ZERO3;
    FLOAT3 th = ZERO3;
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FLOAT3 dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT3 dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        th = atan2pi(dy, dx);
    }
    grad[i] = select(gr, ZERO3, isnan(gr));
    theta[i] = select(th, ZERO3, isnan(th));
}
kernel void gradient4(constant int &width [[ buffer(0) ]],
                              constant int &height [[ buffer(1) ]],
                              const device FLOAT4* input [[ buffer(2) ]],
                              constant int &reach [[ buffer(3) ]],
                              device FLOAT4* grad [[ buffer(4) ]],
                              device FLOAT4* theta [[ buffer(5) ]],
                              uint i [[ thread_position_in_grid ]])
{
    int x = X(i);
    int y = Y(i);
    FLOAT4 gr = ZERO4;
    FLOAT4 th = ZERO4;
    if (0 <= (x-reach)  && (x+reach) < width && 0 <= (y-reach) && (y+reach) < height) {
        FLOAT4 dx = input[ADDRESS(x+reach,y)] - input[ADDRESS(x-reach,y)];
        FLOAT4 dy = input[ADDRESS(x,y+reach)] - input[ADDRESS(x,y-reach)];
        gr = sqrt(dx*dx+dy*dy);
        th = atan2pi(dy, dx);
    }
    grad[i] = select(gr, ZERO4, isnan(gr));
    theta[i] = select(th, ZERO4, isnan(th));
}
