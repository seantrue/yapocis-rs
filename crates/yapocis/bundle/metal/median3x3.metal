/* Metal kernel to compute 3x3 median filter on a 2D image */

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


static inline float median9(thread float *a, int n) {
    float l,h; bool lt;
            lt = a[1] < a[0];
            l = lt ? a[1] : a[0];
            h = lt ? a[0] : a[1];
            a[0] = l;
            a[1] = h;
            lt = a[2] < a[0];
            l = lt ? a[2] : a[0];
            h = lt ? a[0] : a[2];
            a[0] = l;
            a[2] = h;
            lt = a[3] < a[0];
            l = lt ? a[3] : a[0];
            h = lt ? a[0] : a[3];
            a[0] = l;
            a[3] = h;
            lt = a[4] < a[0];
            l = lt ? a[4] : a[0];
            h = lt ? a[0] : a[4];
            a[0] = l;
            a[4] = h;
            lt = a[5] < a[0];
            l = lt ? a[5] : a[0];
            h = lt ? a[0] : a[5];
            a[0] = l;
            a[5] = h;
            lt = a[6] < a[0];
            l = lt ? a[6] : a[0];
            h = lt ? a[0] : a[6];
            a[0] = l;
            a[6] = h;
            lt = a[7] < a[0];
            l = lt ? a[7] : a[0];
            h = lt ? a[0] : a[7];
            a[0] = l;
            a[7] = h;
            lt = a[8] < a[0];
            l = lt ? a[8] : a[0];
            h = lt ? a[0] : a[8];
            a[0] = l;
            a[8] = h;
            lt = a[2] < a[1];
            l = lt ? a[2] : a[1];
            h = lt ? a[1] : a[2];
            a[1] = l;
            a[2] = h;
            lt = a[3] < a[1];
            l = lt ? a[3] : a[1];
            h = lt ? a[1] : a[3];
            a[1] = l;
            a[3] = h;
            lt = a[4] < a[1];
            l = lt ? a[4] : a[1];
            h = lt ? a[1] : a[4];
            a[1] = l;
            a[4] = h;
            lt = a[5] < a[1];
            l = lt ? a[5] : a[1];
            h = lt ? a[1] : a[5];
            a[1] = l;
            a[5] = h;
            lt = a[6] < a[1];
            l = lt ? a[6] : a[1];
            h = lt ? a[1] : a[6];
            a[1] = l;
            a[6] = h;
            lt = a[7] < a[1];
            l = lt ? a[7] : a[1];
            h = lt ? a[1] : a[7];
            a[1] = l;
            a[7] = h;
            lt = a[8] < a[1];
            l = lt ? a[8] : a[1];
            h = lt ? a[1] : a[8];
            a[1] = l;
            a[8] = h;
            lt = a[3] < a[2];
            l = lt ? a[3] : a[2];
            h = lt ? a[2] : a[3];
            a[2] = l;
            a[3] = h;
            lt = a[4] < a[2];
            l = lt ? a[4] : a[2];
            h = lt ? a[2] : a[4];
            a[2] = l;
            a[4] = h;
            lt = a[5] < a[2];
            l = lt ? a[5] : a[2];
            h = lt ? a[2] : a[5];
            a[2] = l;
            a[5] = h;
            lt = a[6] < a[2];
            l = lt ? a[6] : a[2];
            h = lt ? a[2] : a[6];
            a[2] = l;
            a[6] = h;
            lt = a[7] < a[2];
            l = lt ? a[7] : a[2];
            h = lt ? a[2] : a[7];
            a[2] = l;
            a[7] = h;
            lt = a[8] < a[2];
            l = lt ? a[8] : a[2];
            h = lt ? a[2] : a[8];
            a[2] = l;
            a[8] = h;
            lt = a[4] < a[3];
            l = lt ? a[4] : a[3];
            h = lt ? a[3] : a[4];
            a[3] = l;
            a[4] = h;
            lt = a[5] < a[3];
            l = lt ? a[5] : a[3];
            h = lt ? a[3] : a[5];
            a[3] = l;
            a[5] = h;
            lt = a[6] < a[3];
            l = lt ? a[6] : a[3];
            h = lt ? a[3] : a[6];
            a[3] = l;
            a[6] = h;
            lt = a[7] < a[3];
            l = lt ? a[7] : a[3];
            h = lt ? a[3] : a[7];
            a[3] = l;
            a[7] = h;
            lt = a[8] < a[3];
            l = lt ? a[8] : a[3];
            h = lt ? a[3] : a[8];
            a[3] = l;
            a[8] = h;
            lt = a[5] < a[4];
            l = lt ? a[5] : a[4];
            h = lt ? a[4] : a[5];
            a[4] = l;
            a[5] = h;
            lt = a[6] < a[4];
            l = lt ? a[6] : a[4];
            h = lt ? a[4] : a[6];
            a[4] = l;
            a[6] = h;
            lt = a[7] < a[4];
            l = lt ? a[7] : a[4];
            h = lt ? a[4] : a[7];
            a[4] = l;
            a[7] = h;
            lt = a[8] < a[4];
            l = lt ? a[8] : a[4];
            h = lt ? a[4] : a[8];
            a[4] = l;
            a[8] = h;
    return a[n/2];
}

kernel void median3x3(constant int &width [[ buffer(0) ]],
                      constant int &height [[ buffer(1) ]],
                      const device float* a [[ buffer(2) ]],
                      device float* ret [[ buffer(3) ]],
                      uint i [[ thread_position_in_grid ]])
{
    float aa[9];
    float m;
    size_t left = height + 1;
    size_t right = width - left;
    int h = height;
    m = a[i];
    if ((i >= left) &&  (i < right)) {
        aa[0] =  a[(i-h)-1];
        aa[1] =  a[i-h];
        aa[2] =  a[(i-h)+1];
        aa[3] =  a[i-1];
        aa[4] =  a[i];
        aa[5] =  a[i+1];
        aa[6] =  a[(i+h)-1];
        aa[7] =  a[i+h];
        aa[8] =  a[(i+h)+1];
        m = median9(&aa[0], 9);
    }
    ret[i] = m;
}
