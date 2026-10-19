/* Auto-translated from OpenCL to CUDA (Mako kept intact). */
#ifndef YAPOCIS_CUDA_SHIMS_H
#define YAPOCIS_CUDA_SHIMS_H
#define __kernel __global__
#define __local  __shared__
#define __constant __constant__
// nvcc gets `uint` and M_PI from the host headers; NVRTC (the Rust runtime)
// has no host headers, so provide them here. Both are harmless under nvcc:
// a typedef may be repeated with the same type, and M_PI is guarded.
typedef unsigned int uint;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define barrier(x) __syncthreads()
__device__ __forceinline__ int get_global_id_x(){return blockIdx.x*blockDim.x+threadIdx.x;}
__device__ __forceinline__ int get_global_id_y(){return blockIdx.y*blockDim.y+threadIdx.y;}
__device__ __forceinline__ int get_global_id_z(){return blockIdx.z*blockDim.z+threadIdx.z;}
#define get_global_id(N) ( (N)==0 ? get_global_id_x() : ( (N)==1 ? get_global_id_y() : get_global_id_z() ) )
// The launcher rounds the grid up to whole blocks, so threads can run past the
// end of the arrays. A kernel whose parameter list ends with GLOBAL_SIZE_PARAM
// is passed the real global size (see cu/kernel.py) and should start with
// RETURN_IF_OUTSIDE(get_global_id(0)).
#define GLOBAL_SIZE_PARAM , unsigned long long yapocis_global_size
#define RETURN_IF_OUTSIDE(i) if ((unsigned long long)(i) >= yapocis_global_size) return;
__device__ __forceinline__ int clampi(int v,int lo,int hi){return max(lo,min(hi,v));}
__device__ __forceinline__ float clampf(float v,float lo,float hi){return fminf(hi,fmaxf(lo,v));}
__device__ __forceinline__ float clamp(float v,float lo,float hi){return fminf(hi,fmaxf(lo,v));}
__device__ __forceinline__ float2 clamp(float2 v,float lo,float hi){return make_float2(clamp(v.x,lo,hi),clamp(v.y,lo,hi));}
__device__ __forceinline__ float3 clamp(float3 v,float lo,float hi){return make_float3(clamp(v.x,lo,hi),clamp(v.y,lo,hi),clamp(v.z,lo,hi));}
__device__ __forceinline__ float4 clamp(float4 v,float lo,float hi){return make_float4(clamp(v.x,lo,hi),clamp(v.y,lo,hi),clamp(v.z,lo,hi),clamp(v.w,lo,hi));}
template<typename V> __device__ __forceinline__ V  vload(int off, const void* base){return reinterpret_cast<const V*>(base)[off];}
template<typename V> __device__ __forceinline__ void vstore(V v, int off, void* base){reinterpret_cast<V*>(base)[off]=v;}
#define vload4(OFF,PTR)       vload<float4>((OFF),(const void*)(PTR))
#define vstore4(VAL,OFF,PTR)  vstore<float4>((VAL),(OFF),(void*)(PTR))

// Atomic operations (OpenCL to CUDA translation)
__device__ __forceinline__ int atomic_inc(int* addr) { return atomicAdd(addr, 1); }
__device__ __forceinline__ unsigned int atomic_inc(unsigned int* addr) { return atomicAdd(addr, 1u); }

// Math function mappings (OpenCL to CUDA)
__device__ __forceinline__ float atan2pi_scalar(float y, float x) { return atan2f(y, x) / M_PI; }
#define atan2pi atan2pi_scalar
#define sign(x) (((x) > 0) - ((x) < 0))
#define signf(x) (((x) > 0.0f) - ((x) < 0.0f))

// Vector math function overloads
__device__ __forceinline__ float2 sqrt(float2 a) { return make_float2(sqrtf(a.x), sqrtf(a.y)); }
__device__ __forceinline__ float3 sqrt(float3 a) { return make_float3(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z)); }
__device__ __forceinline__ float4 sqrt(float4 a) { return make_float4(sqrtf(a.x), sqrtf(a.y), sqrtf(a.z), sqrtf(a.w)); }

__device__ __forceinline__ bool isnan(float2 a) { return isnan(a.x) || isnan(a.y); }
__device__ __forceinline__ bool isnan(float3 a) { return isnan(a.x) || isnan(a.y) || isnan(a.z); }
__device__ __forceinline__ bool isnan(float4 a) { return isnan(a.x) || isnan(a.y) || isnan(a.z) || isnan(a.w); }

__device__ __forceinline__ float2 atan2pi(float2 y, float2 x) { return make_float2(atan2pi_scalar(y.x, x.x), atan2pi_scalar(y.y, x.y)); }
__device__ __forceinline__ float3 atan2pi(float3 y, float3 x) { return make_float3(atan2pi_scalar(y.x, x.x), atan2pi_scalar(y.y, x.y), atan2pi_scalar(y.z, x.z)); }
__device__ __forceinline__ float4 atan2pi(float4 y, float4 x) { return make_float4(atan2pi_scalar(y.x, x.x), atan2pi_scalar(y.y, x.y), atan2pi_scalar(y.z, x.z), atan2pi_scalar(y.w, x.w)); }

// fract() - fractional part (OpenCL compatibility)
__device__ __forceinline__ float fract(float x, float* iptr) { *iptr = floorf(x); return x - *iptr; }
__device__ __forceinline__ float2 fract(float2 x, float2* iptr) {
    return make_float2(fract(x.x, &iptr->x), fract(x.y, &iptr->y));
}
__device__ __forceinline__ float3 fract(float3 x, float3* iptr) {
    return make_float3(fract(x.x, &iptr->x), fract(x.y, &iptr->y), fract(x.z, &iptr->z));
}
__device__ __forceinline__ float4 fract(float4 x, float4* iptr) {
    return make_float4(fract(x.x, &iptr->x), fract(x.y, &iptr->y), fract(x.z, &iptr->z), fract(x.w, &iptr->w));
}

// mix() - linear interpolation (OpenCL compatibility)
__device__ __forceinline__ float mix(float a, float b, float t) { return a + t * (b - a); }
__device__ __forceinline__ float2 mix(float2 a, float2 b, float t) {
    return make_float2(mix(a.x, b.x, t), mix(a.y, b.y, t));
}
__device__ __forceinline__ float3 mix(float3 a, float3 b, float t) {
    return make_float3(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t));
}
__device__ __forceinline__ float4 mix(float4 a, float4 b, float t) {
    return make_float4(mix(a.x, b.x, t), mix(a.y, b.y, t), mix(a.z, b.z, t), mix(a.w, b.w, t));
}

// floor() for vectors
__device__ __forceinline__ float2 floor(float2 a) { return make_float2(floorf(a.x), floorf(a.y)); }
__device__ __forceinline__ float3 floor(float3 a) { return make_float3(floorf(a.x), floorf(a.y), floorf(a.z)); }
__device__ __forceinline__ float4 floor(float4 a) { return make_float4(floorf(a.x), floorf(a.y), floorf(a.z), floorf(a.w)); }

// fabs() for vectors
__device__ __forceinline__ float2 fabs(float2 a) { return make_float2(fabsf(a.x), fabsf(a.y)); }
__device__ __forceinline__ float3 fabs(float3 a) { return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z)); }
__device__ __forceinline__ float4 fabs(float4 a) { return make_float4(fabsf(a.x), fabsf(a.y), fabsf(a.z), fabsf(a.w)); }

// OpenCL native_powr compatibility (fast power function)
#define native_powr(x, y) powf((x), (y))

#endif



// Define memory usage constants (empty for CUDA)
#define READONLY
#define READWRITE

// Vector type definitions
#define FLOAT float
#define FLOAT2 float2
#define FLOAT3 float3
#define FLOAT4 float4

// Vector constants
#define ZERO 0.0f
#define ZERO2 make_float2(ZERO,ZERO)
#define ZERO3 make_float3(ZERO,ZERO,ZERO)
#define ZERO4 make_float4(ZERO,ZERO,ZERO,ZERO)
#define ONE 1.0f
#define ONE2 make_float2(ONE,ONE)
#define ONE3 make_float3(ONE,ONE,ONE)
#define ONE4 make_float4(ONE,ONE,ONE,ONE)

// Scalar broadcast helpers
__device__ __forceinline__ float2 make_float2_broadcast(float x) { return make_float2(x, x); }
__device__ __forceinline__ float3 make_float3_broadcast(float x) { return make_float3(x, x, x); }
__device__ __forceinline__ float4 make_float4_broadcast(float x) { return make_float4(x, x, x, x); }

// Vector arithmetic operators
__device__ __forceinline__ float2 operator/(float a, float2 b) { return make_float2(a/b.x, a/b.y); }
__device__ __forceinline__ float3 operator/(float a, float3 b) { return make_float3(a/b.x, a/b.y, a/b.z); }
__device__ __forceinline__ float4 operator/(float a, float4 b) { return make_float4(a/b.x, a/b.y, a/b.z, a/b.w); }
__device__ __forceinline__ float2 operator/(float2 a, float b) { return make_float2(a.x/b, a.y/b); }
__device__ __forceinline__ float3 operator/(float3 a, float b) { return make_float3(a.x/b, a.y/b, a.z/b); }
__device__ __forceinline__ float4 operator/(float4 a, float b) { return make_float4(a.x/b, a.y/b, a.z/b, a.w/b); }

__device__ __forceinline__ float2 operator*(float a, float2 b) { return make_float2(a*b.x, a*b.y); }
__device__ __forceinline__ float3 operator*(float a, float3 b) { return make_float3(a*b.x, a*b.y, a*b.z); }
__device__ __forceinline__ float4 operator*(float a, float4 b) { return make_float4(a*b.x, a*b.y, a*b.z, a*b.w); }
__device__ __forceinline__ float2 operator*(float2 a, float b) { return make_float2(a.x*b, a.y*b); }
__device__ __forceinline__ float3 operator*(float3 a, float b) { return make_float3(a.x*b, a.y*b, a.z*b); }
__device__ __forceinline__ float4 operator*(float4 a, float b) { return make_float4(a.x*b, a.y*b, a.z*b, a.w*b); }

__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return make_float2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float4 operator+(float4 a, float4 b) { return make_float4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }

__device__ __forceinline__ float2 operator-(float2 a, float2 b) { return make_float2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float4 operator-(float4 a, float4 b) { return make_float4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }

__device__ __forceinline__ float2 operator*(float2 a, float2 b) { return make_float2(a.x*b.x, a.y*b.y); }
__device__ __forceinline__ float3 operator*(float3 a, float3 b) { return make_float3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ __forceinline__ float4 operator*(float4 a, float4 b) { return make_float4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }

__device__ __forceinline__ float2 operator/(float2 a, float2 b) { return make_float2(a.x/b.x, a.y/b.y); }
__device__ __forceinline__ float3 operator/(float3 a, float3 b) { return make_float3(a.x/b.x, a.y/b.y, a.z/b.z); }
__device__ __forceinline__ float4 operator/(float4 a, float4 b) { return make_float4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }

// Compound assignment operators
__device__ __forceinline__ void operator+=(float2& a, float2 b) { a = a + b; }
__device__ __forceinline__ void operator+=(float3& a, float3 b) { a = a + b; }
__device__ __forceinline__ void operator+=(float4& a, float4 b) { a = a + b; }

__device__ __forceinline__ void operator-=(float2& a, float2 b) { a = a - b; }
__device__ __forceinline__ void operator-=(float3& a, float3 b) { a = a - b; }
__device__ __forceinline__ void operator-=(float4& a, float4 b) { a = a - b; }

__device__ __forceinline__ void operator*=(float2& a, float b) { a = a * b; }
__device__ __forceinline__ void operator*=(float3& a, float b) { a = a * b; }
__device__ __forceinline__ void operator*=(float4& a, float b) { a = a * b; }

__device__ __forceinline__ void operator/=(float2& a, float b) { a = a / b; }
__device__ __forceinline__ void operator/=(float3& a, float b) { a = a / b; }
__device__ __forceinline__ void operator/=(float4& a, float b) { a = a / b; }

__device__ __forceinline__ void operator/=(float2& a, float2 b) { a = a / b; }
__device__ __forceinline__ void operator/=(float3& a, float3 b) { a = a / b; }
__device__ __forceinline__ void operator/=(float4& a, float4 b) { a = a / b; }

// Integer vector arithmetic operators
__device__ __forceinline__ int2 operator+(int2 a, int2 b) { return make_int2(a.x+b.x, a.y+b.y); }
__device__ __forceinline__ int3 operator+(int3 a, int3 b) { return make_int3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ int4 operator+(int4 a, int4 b) { return make_int4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }

__device__ __forceinline__ int2 operator-(int2 a, int2 b) { return make_int2(a.x-b.x, a.y-b.y); }
__device__ __forceinline__ int3 operator-(int3 a, int3 b) { return make_int3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ int4 operator-(int4 a, int4 b) { return make_int4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }

__device__ __forceinline__ int2 operator*(int2 a, int2 b) { return make_int2(a.x*b.x, a.y*b.y); }
__device__ __forceinline__ int3 operator*(int3 a, int3 b) { return make_int3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ __forceinline__ int4 operator*(int4 a, int4 b) { return make_int4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }

#define eps 1.0e-10

#ifndef YAPOCIS_CUDA_HELPERS_H
#define YAPOCIS_CUDA_HELPERS_H

/* Helpful macros and in-line functions */

// Address macros
#define ADDRESS(xx,yy) (((yy)*height)+(xx))

#define X(addr) ((addr)%height)
#define Y(addr) ((addr)/height)

#define SWAPINT(xxx,yyy) {int tmp = xxx; xxx=yyy; yyy=tmp;}
#define SWAPFLOAT(xxx,yyy) {float tmp = xxx; xxx=yyy; yyy=tmp;}

#define ABS(x) (((x) < 0) ? -(x) : (x))

#ifdef DXDY2
static __device__ __inline__ float dxdy2(float2 a, float2 b) {
    float2 dxdy =  (a-b)*(a-b);
    return dxdy.x + dxdy.y;
}
#endif

#ifdef DOT3
static __device__ __inline__ float dot3(float3 a, float3 b) {
   return (float) (a.x*b.x + a.y*b.y + a.z*b.z);
}
#endif

#ifdef VXM3
static __device__ __inline__ float3 vxm3(float3 v, float3 a, float3 b, float3 c) {
    float3 result = make_float3(dot3(v,a), dot3(v,b), dot3(v,c));
    return result;
}
#endif

#ifdef DEBUG
static __inline void p3(__constant char *s, float3 v) {
    printf("%s %.5f %.5f %.5f\n", s, v.x, v.y, v.z);
}

static __inline void p4(__constant char *s, float4 v) {
    printf("%s %.5f %.5f %.5f %.5f\n", s, v.x, v.y, v.z, v.w);
}
#endif

#define ANYISLESS(a,b) (a.x < b.x || a.y < b.y)
#define ANYISGTE(a,b) (a.x >= b.x || a.y >= b.y)

// Anti-aliasing helper functions
static __device__ __inline__ float smoothstep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Note: mix() functions are defined in constants.mako

static __device__ __inline__ float dot(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

static __device__ __inline__ float length(float2 v) {
    return sqrtf(v.x * v.x + v.y * v.y);
}


#endif


/* Multiple one-dimensional convolutions given names and kernels at compile time */


extern "C" __kernel
void hamming5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.035714287;
       sum = sum + a[i+(-1)] * 0.24107143;
       sum = sum + a[i+(0)] * 0.44642857;
       sum = sum + a[i+(1)] * 0.24107143;
       sum = sum + a[i+(2)] * 0.035714287;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void hamming9(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-4;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void hamming13(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-6;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void hamming17(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-8;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void hamming25(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-12;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void hann5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.0;
       sum = sum + a[i+(-1)] * 0.25;
       sum = sum + a[i+(0)] * 0.5;
       sum = sum + a[i+(1)] * 0.25;
       sum = sum + a[i+(2)] * 0.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void hann9(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-4;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void hann13(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-6;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void hann17(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-8;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void hann25(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-12;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void blackman7(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void blackman11(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-5;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void blackman15(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-7;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void blackman21(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-10;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void bartlett5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.0;
       sum = sum + a[i+(-1)] * 0.25;
       sum = sum + a[i+(0)] * 0.5;
       sum = sum + a[i+(1)] * 0.25;
       sum = sum + a[i+(2)] * 0.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void bartlett9(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-4;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void bartlett13(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-6;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void bartlett17(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-8;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void box3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * 0.33333334;
       sum = sum + a[i+(0)] * 0.33333334;
       sum = sum + a[i+(1)] * 0.33333334;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void box5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.2;
       sum = sum + a[i+(-1)] * 0.2;
       sum = sum + a[i+(0)] * 0.2;
       sum = sum + a[i+(1)] * 0.2;
       sum = sum + a[i+(2)] * 0.2;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void box7(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void box9(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-4;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void box11(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-5;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void box15(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-7;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void box21(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-10;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void triangle3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * 0.0;
       sum = sum + a[i+(0)] * 1.0;
       sum = sum + a[i+(1)] * 0.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void triangle5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.0;
       sum = sum + a[i+(-1)] * 0.25;
       sum = sum + a[i+(0)] * 0.5;
       sum = sum + a[i+(1)] * 0.25;
       sum = sum + a[i+(2)] * 0.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void triangle7(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void triangle9(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-4;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void triangle11(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-5;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void binomial3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * 0.25;
       sum = sum + a[i+(0)] * 0.5;
       sum = sum + a[i+(1)] * 0.25;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void binomial5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.0625;
       sum = sum + a[i+(-1)] * 0.25;
       sum = sum + a[i+(0)] * 0.375;
       sum = sum + a[i+(1)] * 0.25;
       sum = sum + a[i+(2)] * 0.0625;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void binomial7(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void binomial9(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-4;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void deriv13(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * -0.5;
       sum = sum + a[i+(0)] * 0.0;
       sum = sum + a[i+(1)] * 0.5;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void deriv15(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.083333336;
       sum = sum + a[i+(-1)] * -0.6666667;
       sum = sum + a[i+(0)] * 0.0;
       sum = sum + a[i+(1)] * 0.6666667;
       sum = sum + a[i+(2)] * -0.083333336;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void deriv17(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void deriv23(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * 1.0;
       sum = sum + a[i+(0)] * -2.0;
       sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void deriv25(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * -0.083333336;
       sum = sum + a[i+(-1)] * 1.3333334;
       sum = sum + a[i+(0)] * -2.5;
       sum = sum + a[i+(1)] * 1.3333334;
       sum = sum + a[i+(2)] * -0.083333336;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void deriv27(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void sobel3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * -1.0;
       sum = sum + a[i+(0)] * 0.0;
       sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void prewitt3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * -1.0;
       sum = sum + a[i+(0)] * 0.0;
       sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void sharpen3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * -1.0;
       sum = sum + a[i+(0)] * 3.0;
       sum = sum + a[i+(1)] * -1.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void sharpen5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * -1.0;
       sum = sum + a[i+(-1)] * -1.0;
       sum = sum + a[i+(0)] * 5.0;
       sum = sum + a[i+(1)] * -1.0;
       sum = sum + a[i+(2)] * -1.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void unsharp3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * -0.33333334;
       sum = sum + a[i+(0)] * 1.6666666;
       sum = sum + a[i+(1)] * -0.33333334;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void unsharp5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * -0.2;
       sum = sum + a[i+(-1)] * -0.2;
       sum = sum + a[i+(0)] * 1.8;
       sum = sum + a[i+(1)] * -0.2;
       sum = sum + a[i+(2)] * -0.2;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void unsharp7(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void dog5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * -0.25;
       sum = sum + a[i+(-1)] * -0.25;
       sum = sum + a[i+(0)] * 1.0;
       sum = sum + a[i+(1)] * -0.25;
       sum = sum + a[i+(2)] * -0.25;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void dog7(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void dog9(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-4;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void dog11(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-5;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}

extern "C" __kernel
void edge3(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-1;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * -1.0;
       sum = sum + a[i+(0)] * 0.0;
       sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void edge5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * -1.0;
       sum = sum + a[i+(-1)] * -0.5;
       sum = sum + a[i+(0)] * 0.0;
       sum = sum + a[i+(1)] * 0.5;
       sum = sum + a[i+(2)] * 1.0;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void savgol5(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-2;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
    if ((i > 2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * -0.06382979;
       sum = sum + a[i+(-1)] * 0.25531915;
       sum = sum + a[i+(0)] * 0.3617021;
       sum = sum + a[i+(1)] * 0.25531915;
       sum = sum + a[i+(2)] * -0.06382979;
    }
    ret[i] = sum;
    return;
}

extern "C" __kernel
void savgol7(int width, float* a, float* ret )
{
    size_t i;
    float sum;
    size_t right;
    right = width-3;
    i = get_global_id(0);
    if (i >= (size_t)width) return;
    sum = 0.0;
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
    return;
}
