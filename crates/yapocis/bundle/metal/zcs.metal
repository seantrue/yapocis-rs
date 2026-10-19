/* Metal kernels to find zero-crossings in an image. */

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


/* Find zero-crossings for pixels of 1-4 channels. */
#define ADDRESS1(xx,yy) (ADDRESS(xx,yy))*1

kernel void zcs1(constant int &width [[ buffer(0) ]],
                         constant int &height [[ buffer(1) ]],
                         constant float &threshold [[ buffer(2) ]],
                         const device float* input [[ buffer(3) ]],
                         device float* zcs [[ buffer(4) ]],
                         uint i [[ thread_position_in_grid ]])
{
    size_t idx = i*1;
    int x = max((size_t)1,X(idx));
    x = min(x,width-2);
    int y = max((size_t)1,Y(idx));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[idx];
    above = input[ADDRESS1(x,y+1)];
    below = input[ADDRESS1(x,y-1)];
    left = input[ADDRESS1(x-1,y)];
    right = input[ADDRESS1(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
}
#define ADDRESS2(xx,yy) (ADDRESS(xx,yy))*2

kernel void zcs2(constant int &width [[ buffer(0) ]],
                         constant int &height [[ buffer(1) ]],
                         constant float &threshold [[ buffer(2) ]],
                         const device float* input [[ buffer(3) ]],
                         device float* zcs [[ buffer(4) ]],
                         uint i [[ thread_position_in_grid ]])
{
    size_t idx = i*2;
    int x = max((size_t)1,X(idx));
    x = min(x,width-2);
    int y = max((size_t)1,Y(idx));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[idx];
    above = input[ADDRESS2(x,y+1)];
    below = input[ADDRESS2(x,y-1)];
    left = input[ADDRESS2(x-1,y)];
    right = input[ADDRESS2(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
    v = input[idx];
    above = input[ADDRESS2(x,y+1)];
    below = input[ADDRESS2(x,y-1)];
    left = input[ADDRESS2(x-1,y)];
    right = input[ADDRESS2(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
}
#define ADDRESS3(xx,yy) (ADDRESS(xx,yy))*3

kernel void zcs3(constant int &width [[ buffer(0) ]],
                         constant int &height [[ buffer(1) ]],
                         constant float &threshold [[ buffer(2) ]],
                         const device float* input [[ buffer(3) ]],
                         device float* zcs [[ buffer(4) ]],
                         uint i [[ thread_position_in_grid ]])
{
    size_t idx = i*3;
    int x = max((size_t)1,X(idx));
    x = min(x,width-2);
    int y = max((size_t)1,Y(idx));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[idx];
    above = input[ADDRESS3(x,y+1)];
    below = input[ADDRESS3(x,y-1)];
    left = input[ADDRESS3(x-1,y)];
    right = input[ADDRESS3(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
    v = input[idx];
    above = input[ADDRESS3(x,y+1)];
    below = input[ADDRESS3(x,y-1)];
    left = input[ADDRESS3(x-1,y)];
    right = input[ADDRESS3(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
    v = input[idx];
    above = input[ADDRESS3(x,y+1)];
    below = input[ADDRESS3(x,y-1)];
    left = input[ADDRESS3(x-1,y)];
    right = input[ADDRESS3(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
}
#define ADDRESS4(xx,yy) (ADDRESS(xx,yy))*4

kernel void zcs4(constant int &width [[ buffer(0) ]],
                         constant int &height [[ buffer(1) ]],
                         constant float &threshold [[ buffer(2) ]],
                         const device float* input [[ buffer(3) ]],
                         device float* zcs [[ buffer(4) ]],
                         uint i [[ thread_position_in_grid ]])
{
    size_t idx = i*4;
    int x = max((size_t)1,X(idx));
    x = min(x,width-2);
    int y = max((size_t)1,Y(idx));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[idx];
    above = input[ADDRESS4(x,y+1)];
    below = input[ADDRESS4(x,y-1)];
    left = input[ADDRESS4(x-1,y)];
    right = input[ADDRESS4(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
    v = input[idx];
    above = input[ADDRESS4(x,y+1)];
    below = input[ADDRESS4(x,y-1)];
    left = input[ADDRESS4(x-1,y)];
    right = input[ADDRESS4(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
    v = input[idx];
    above = input[ADDRESS4(x,y+1)];
    below = input[ADDRESS4(x,y-1)];
    left = input[ADDRESS4(x-1,y)];
    right = input[ADDRESS4(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
    v = input[idx];
    above = input[ADDRESS4(x,y+1)];
    below = input[ADDRESS4(x,y-1)];
    left = input[ADDRESS4(x-1,y)];
    right = input[ADDRESS4(x+1,y)];
    delta = ((above-below)*(above-below)+(right-left)*(right-left));
    above = sign(above);
    below = sign(below);
    left = sign(left);
    right = sign(right);
    zclt = (delta > threshold) && ((below != above) || (left != right)) ? ONE : ZERO;
    zcs[idx] = zclt;
    idx += 1;
}

kernel void sub(device float* input1 [[ buffer(0) ]],
                device float* input2 [[ buffer(1) ]],
                device float *output [[ buffer(2) ]],
                uint i [[ thread_position_in_grid ]])
{
    output[i] = input1[i] - input2[i];
}
