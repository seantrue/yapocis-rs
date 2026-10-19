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

#define VXM3
#define DOT3
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


/* A variety of colorspace transforms */

__constant float transforms[] =
{
    // adobe1998_d65.rgb2xyz
    0.5767309,0.2973769,0.0270343,
    0.185554,0.6273491,0.0706872,
    0.1881852,0.0752741,0.9911085,
    // adobe1998_d65.xyz2rgb
    2.041369,-0.969266,0.0134474,
    -0.5649464,1.8760108,-0.1183897,
    -0.3446944,0.041556,1.0154096,
    // apple_d65.rgb2xyz
    0.4497288,0.2446525,0.0251848,
    0.3162486,0.6720283,0.1411824,
    0.1844926,0.0833192,0.9224628,
    // apple_d65.xyz2rgb
    2.9515373,-1.0851093,0.0854934,
    -1.2894116,1.9908566,-0.2694964,
    -0.4738445,0.0372026,1.0912975,
    // best_d50.rgb2xyz
    0.6326696,0.2284569,0.0,
    0.2045558,0.7373523,0.0095142,
    0.1269946,0.0341908,0.8156958,
    // best_d50.xyz2rgb
    1.7552599,-0.5441336,0.0063467,
    -0.4836786,1.5068789,-0.0175761,
    -0.253,0.0215528,1.2256959,
    // beta_d50.rgb2xyz
    0.6712537,0.3032726,0.0,
    0.1745834,0.6637861,0.040701,
    0.1183829,0.0329413,0.784509,
    // beta_d50.xyz2rgb
    1.683227,-0.7710229,0.0400013,
    -0.4282363,1.7065571,-0.0885376,
    -0.2360185,0.04469,1.272364,
    // bruce_d65.rgb2xyz
    0.4674162,0.2410115,0.0219101,
    0.2944512,0.6835475,0.0736128,
    0.1886026,0.075441,0.9933071,
    // bruce_d65.xyz2rgb
    2.7454669,-0.969266,0.0112723,
    -1.1358136,1.8760108,-0.1139754,
    -0.4350269,0.041556,1.0132541,
    // cie_e.rgb2xyz
    0.488718,0.1762044,0.0,
    0.3106803,0.8129847,0.0102048,
    0.2006017,0.0108109,0.9897952,
    // cie_e.xyz2rgb
    2.3706743,-0.513885,0.0052982,
    -0.9000405,1.4253036,-0.0146949,
    -0.4706338,0.0885814,1.0093968,
    // colormatch_d50.rgb2xyz
    0.5093439,0.274884,0.0242545,
    0.3209071,0.6581315,0.1087821,
    0.1339691,0.0669845,0.6921735,
    // colormatch_d50.xyz2rgb
    2.6422874,-1.1119763,0.0821699,
    -1.223427,2.0590183,-0.2807254,
    -0.3930143,0.0159614,1.4559877,
    // donrgb3_d50.rgb2xyz
    0.6457711,0.2783496,0.0037113,
    0.1933511,0.6879702,0.0179861,
    0.1250978,0.0336802,0.8035125,
    // donrgb3_d50.xyz2rgb
    1.7603902,-0.7126288,0.0078207,
    -0.4881198,1.6527432,-0.0347411,
    -0.2536126,0.0416715,1.2447743,
    // eci_d50.rgb2xyz
    0.6502043,0.3202499,0.0,
    0.1780774,0.6020711,0.067839,
    0.1359384,0.0776791,0.757371,
    // eci_d50.xyz2rgb
    1.7827618,-0.9593623,0.0859317,
    -0.4969847,1.9477962,-0.1744674,
    -0.2690101,-0.0275807,1.3228273,
    // ektaspaceps5_d50.rgb2xyz
    0.5938914,0.2606286,0.0,
    0.2729801,0.7349465,0.0419969,
    0.0973485,0.0044249,0.7832131,
    // ektaspaceps5_d50.xyz2rgb
    2.0043819,-0.7110285,0.0381263,
    -0.7304844,1.6202126,-0.086878,
    -0.2450052,0.0792227,1.2725438,
    // ntsc_c.rgb2xyz
    0.6068909,0.2989164,0.0,
    0.1735011,0.586599,0.0660957,
    0.200348,0.1144845,1.1162243,
    // ntsc_c.xyz2rgb
    1.9099961,-0.9846663,0.0583056,
    -0.5324542,1.999171,-0.1183781,
    -0.2882091,-0.0283082,0.8975535,
    // palsecam_d65.rgb2xyz
    0.430619,0.2220379,0.0201853,
    0.3415419,0.7066384,0.1295504,
    0.1783091,0.0713236,0.9390944,
    // palsecam_d65.xyz2rgb
    3.0628971,-0.969266,0.0678775,
    -1.3931791,1.8760108,-0.2288548,
    -0.4757517,0.041556,1.069349,
    // prophoto_d50.rgb2xyz
    0.7976749,0.2880402,0.0,
    0.1351917,0.7118741,0.0,
    0.0313534,8.57e-05,0.82521,
    // prophoto_d50.xyz2rgb
    1.3459433,-0.5445989,0.0,
    -0.2556075,1.5081673,0.0,
    -0.0511118,0.0205351,1.2118128,
    // smpte-c_d65.rgb2xyz
    0.3935891,0.2124132,0.0187423,
    0.3652497,0.7010437,0.1119313,
    0.1916313,0.0865432,0.9581563,
    // smpte-c_d65.xyz2rgb
    3.505396,-1.0690722,0.05632,
    -1.7394894,1.9778245,-0.1970226,
    -0.543964,0.0351722,1.0502026,
    // srgb_d65.rgb2xyz
    0.4124564,0.2126729,0.0193339,
    0.3575761,0.7151522,0.119192,
    0.1804375,0.072175,0.9503041,
    // srgb_d65.xyz2rgb
    3.2404542,-0.969266,0.0556434,
    -1.5371385,1.8760108,-0.2040259,
    -0.4985314,0.041556,1.0572252,
    // wide_d50.rgb2xyz
    0.7161046,0.2581874,0.0,
    0.1009296,0.7249378,0.0517813,
    0.1471858,0.0168748,0.7734287,
    // wide_d50.xyz2rgb
    1.4628067,-0.5217933,0.0349342,
    -0.1840623,1.4472381,-0.096893,
    -0.2743606,0.0677227,1.2884099,
    // adobe_d50.rgb2xyz
    0.6097559,0.3111242,0.0194811,
    0.2052401,0.625656,0.0608902,
    0.149224,0.0632197,0.7448387,
    // adobe_d50.xyz2rgb
    1.9624274,-0.9787684,0.0286869,
    -0.6105343,1.9161415,-0.1406752,
    -0.3413404,0.033454,1.3487655,
    // applergb_d50.rgb2xyz
    0.4755678,0.2551812,0.0184697,
    0.3396722,0.6725693,0.1133771,
    0.14898,0.0722496,0.6933632,
    // applergb_d50.xyz2rgb
    2.8510695,-1.092768,0.1027403,
    -1.3605261,2.0348871,-0.2964984,
    -0.4708281,0.0227598,1.4510659,
    // bruce_d50.rgb2xyz
    0.4941816,0.2521531,0.0157886,
    0.3204834,0.6844869,0.0629304,
    0.149555,0.06336,0.7464909,
    // bruce_d50.xyz2rgb
    2.6502856,-0.9787684,0.026457,
    -1.2014485,1.9161415,-0.1361227,
    -0.4289936,0.033454,1.3458542,
    // cie_d50.rgb2xyz
    0.486887,0.1746583,-0.0012563,
    0.3062984,0.8247541,0.0169832,
    0.1710347,0.0005877,0.8094831,
    // cie_d50.xyz2rgb
    2.3638081,-0.500594,0.0141712,
    -0.867603,1.3962369,-0.03064,
    -0.4988161,0.1047562,1.2323842,
    // ntsc_d50.rgb2xyz
    0.6343706,0.3109496,-0.0011817,
    0.1852204,0.5915984,0.0555518,
    0.144629,0.097452,0.7708399,
    // ntsc_d50.xyz2rgb
    1.8464881,-0.982663,0.0736477,
    -0.5521299,2.0044755,-0.145302,
    -0.2766458,-0.0690396,1.3018376,
    // palsecam_d50.rgb2xyz
    0.4552773,0.2323025,0.0145457,
    0.36755,0.7077956,0.1049154,
    0.1413926,0.0599019,0.7057489,
    // palsecam_d50.xyz2rgb
    2.9603944,-0.9787684,0.0844874,
    -1.4678519,1.9161415,-0.2545973,
    -0.4685105,0.033454,1.4216174,
    // smptec_d50.rgb2xyz
    0.416329,0.2216999,0.0136576,
    0.3931464,0.7032549,0.0913604,
    0.1547446,0.0750452,0.720192,
    // smptec_d50.xyz2rgb
    3.392194,-1.0770996,0.0723073,
    -1.8264027,2.0213975,-0.2217902,
    -0.5385522,0.0207989,1.3960932,
    // srgb_d50.rgb2xyz
    0.4360747,0.2225045,0.0139322,
    0.3850649,0.7168786,0.0971045,
    0.1430804,0.0606169,0.7141733,
    // srgb_d50.xyz2rgb
    3.1338561,-0.9787684,0.0719453,
    -1.6168667,1.9161415,-0.2289914,
    -0.4906146,0.033454,1.4052427,
    // unity
    1,0,0,
    0,1,0,
    0,0,1
};

extern "C" __kernel void transform(float *input, int transform_id, float *output GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
    size_t i = get_global_id(0)*3;
    size_t tid  = transform_id*9;
    // transforms[] stores each 3x3 matrix column by column, so the rows a, b, c are
    // gathered across it; vxm3 then dots the pixel with each row.
    float3 a = make_float3(transforms[tid+0],transforms[tid+3],transforms[tid+6]);
    float3 b = make_float3(transforms[tid+1],transforms[tid+4],transforms[tid+7]);
    float3 c = make_float3(transforms[tid+2],transforms[tid+5],transforms[tid+8]);
    float3 rgb = make_float3(input[i], input[i+1], input[i+2]);
    float3 xyz = vxm3(rgb, a, b, c);
    output[i]   = xyz.x;
    output[i+1] = xyz.y;
    output[i+2] = xyz.z;
}

// x = X/(X+Y+Z)
// y = Y/(X+Y+Z)
// y = y
extern "C" __kernel void xyz2xyY(float *input, float *output GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
    size_t i = get_global_id(0)*3;
    float x = input[i];
    float y = input[i+1];
    float z = input[i+2];
    float sum = x+y+z;
    output[i]   = x/sum;
    output[i+1] = y/sum;
    output[i+2] = y;
}

// X = x*Y/y
// Y = Y
// Z = ((1−x−y)*Y)/y
extern "C" __kernel void xyY2xyz(float *input, float *output GLOBAL_SIZE_PARAM) {
    RETURN_IF_OUTSIDE(get_global_id(0))
    size_t i = get_global_id(0)*3;
    float x = input[i];
    float y = input[i+1];
    float Y = input[i+2];
    float X = x*Y/y;
    float Z = ((1.0-x-y)*Y)/y;
    output[i]   = y != 0.0 ? X : 0.0;
    output[i+1] = y != 0.0 ? Y : 0.0;
    output[i+2] = y != 0.0 ? Z : 0.0;
}
