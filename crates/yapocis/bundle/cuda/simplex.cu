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


// ============================================================================
// Simplex Noise Implementation (Ken Perlin, 2001) - CUDA Backend
// ============================================================================
// IMPORTANT: CUDA uses row-major storage order (opposite of OpenCL/Metal)

// Hash function for pseudo-random gradient generation
__device__ inline int hash_int(int x, int y, int z, int seed)
{
    int h = seed;
    h ^= x * 374761393;
    h ^= y * 668265263;
    h ^= z * 1264258069;
    h = (h ^ (h >> 13)) * 1274126177;
    return h ^ (h >> 16);
}

// 2D gradient vectors (12 directions)
__constant__ float2 grad2[] = {
    {  1.0f,  1.0f }, { -1.0f,  1.0f },
    {  1.0f, -1.0f }, { -1.0f, -1.0f },
    {  1.0f,  0.0f }, { -1.0f,  0.0f },
    {  0.0f,  1.0f }, {  0.0f, -1.0f },
    {  1.0f,  1.0f }, {  0.0f, -1.0f },
    { -1.0f,  1.0f }, {  0.0f, -1.0f }
};

// 3D gradient vectors (12 directions)
__constant__ float3 grad3[] = {
    {  1.0f,  1.0f,  0.0f }, { -1.0f,  1.0f,  0.0f },
    {  1.0f, -1.0f,  0.0f }, { -1.0f, -1.0f,  0.0f },
    {  1.0f,  0.0f,  1.0f }, { -1.0f,  0.0f,  1.0f },
    {  1.0f,  0.0f, -1.0f }, { -1.0f,  0.0f, -1.0f },
    {  0.0f,  1.0f,  1.0f }, {  0.0f, -1.0f,  1.0f },
    {  0.0f,  1.0f, -1.0f }, {  0.0f, -1.0f, -1.0f }
};

// Dot product helper for 2D
__device__ inline float dot2(float2 a, float2 b)
{
    return a.x * b.x + a.y * b.y;
}

// Dot product helper for 3D
__device__ inline float dot3(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// 2D simplex noise (single octave)
__device__ inline float simplex2d_single(float x, float y, int seed, int tileable, int tile_w, int tile_h)
{
    const float F2 = 0.3660254037844386f;
    const float G2 = 0.21132486540518713f;

    float s = (x + y) * F2;
    int i = floorf(x + s);
    int j = floorf(y + s);

    float t = (i + j) * G2;
    float X0 = i - t;
    float Y0 = j - t;
    float x0 = x - X0;
    float y0 = y - Y0;

    int i1, j1;
    if (x0 > y0) {
        i1 = 1; j1 = 0;
    } else {
        i1 = 0; j1 = 1;
    }

    float x1 = x0 - i1 + G2;
    float y1 = y0 - j1 + G2;
    float x2 = x0 - 1.0f + 2.0f * G2;
    float y2 = y0 - 1.0f + 2.0f * G2;

    int ii = i, jj = j;
    if (tileable) {
        ii = i % tile_w;
        jj = j % tile_h;
        if (ii < 0) ii += tile_w;
        if (jj < 0) jj += tile_h;
    }

    float n0, n1, n2;

    float t0 = 0.5f - x0 * x0 - y0 * y0;
    if (t0 < 0.0f) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        int gi0 = hash_int(ii, jj, 0, seed) & 11;
        n0 = t0 * t0 * dot2(grad2[gi0], make_float2(x0, y0));
    }

    float t1 = 0.5f - x1 * x1 - y1 * y1;
    if (t1 < 0.0f) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        int ii1 = tileable ? (ii + i1) % tile_w : (ii + i1);
        int jj1 = tileable ? (jj + j1) % tile_h : (jj + j1);
        if (tileable && ii1 < 0) ii1 += tile_w;
        if (tileable && jj1 < 0) jj1 += tile_h;
        int gi1 = hash_int(ii1, jj1, 0, seed) & 11;
        n1 = t1 * t1 * dot2(grad2[gi1], make_float2(x1, y1));
    }

    float t2 = 0.5f - x2 * x2 - y2 * y2;
    if (t2 < 0.0f) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        int ii2 = tileable ? (ii + 1) % tile_w : (ii + 1);
        int jj2 = tileable ? (jj + 1) % tile_h : (jj + 1);
        if (tileable && ii2 < 0) ii2 += tile_w;
        if (tileable && jj2 < 0) jj2 += tile_h;
        int gi2 = hash_int(ii2, jj2, 0, seed) & 11;
        n2 = t2 * t2 * dot2(grad2[gi2], make_float2(x2, y2));
    }

    return 70.0f * (n0 + n1 + n2);
}

// 3D simplex noise (single octave)
__device__ inline float simplex3d_single(float x, float y, float z, int seed)
{
    const float F3 = 0.3333333333333333f;
    const float G3 = 0.16666666666666666f;

    float s = (x + y + z) * F3;
    int i = floorf(x + s);
    int j = floorf(y + s);
    int k = floorf(z + s);

    float t = (i + j + k) * G3;
    float X0 = i - t;
    float Y0 = j - t;
    float Z0 = k - t;
    float x0 = x - X0;
    float y0 = y - Y0;
    float z0 = z - Z0;

    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1=1; j1=0; k1=0; i2=1; j2=1; k2=0;
        } else if (x0 >= z0) {
            i1=1; j1=0; k1=0; i2=1; j2=0; k2=1;
        } else {
            i1=0; j1=0; k1=1; i2=1; j2=0; k2=1;
        }
    } else {
        if (y0 < z0) {
            i1=0; j1=0; k1=1; i2=0; j2=1; k2=1;
        } else if (x0 < z0) {
            i1=0; j1=1; k1=0; i2=0; j2=1; k2=1;
        } else {
            i1=0; j1=1; k1=0; i2=1; j2=1; k2=0;
        }
    }

    float x1 = x0 - i1 + G3;
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0f * G3;
    float y2 = y0 - j2 + 2.0f * G3;
    float z2 = z0 - k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3;
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    float n0, n1, n2, n3;

    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    if (t0 < 0.0f) {
        n0 = 0.0f;
    } else {
        t0 *= t0;
        int gi0 = hash_int(i, j, k, seed) % 12;
        n0 = t0 * t0 * dot3(grad3[gi0], make_float3(x0, y0, z0));
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    if (t1 < 0.0f) {
        n1 = 0.0f;
    } else {
        t1 *= t1;
        int gi1 = hash_int(i + i1, j + j1, k + k1, seed) % 12;
        n1 = t1 * t1 * dot3(grad3[gi1], make_float3(x1, y1, z1));
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    if (t2 < 0.0f) {
        n2 = 0.0f;
    } else {
        t2 *= t2;
        int gi2 = hash_int(i + i2, j + j2, k + k2, seed) % 12;
        n2 = t2 * t2 * dot3(grad3[gi2], make_float3(x2, y2, z2));
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    if (t3 < 0.0f) {
        n3 = 0.0f;
    } else {
        t3 *= t3;
        int gi3 = hash_int(i + 1, j + 1, k + 1, seed) % 12;
        n3 = t3 * t3 * dot3(grad3[gi3], make_float3(x3, y3, z3));
    }

    return 32.0f * (n0 + n1 + n2 + n3);
}

// ============================================================================
// 2D Simplex Noise Kernel (CUDA - Row-major)
// ============================================================================

extern "C" __global__ void simplex2d(int width, int height, float *output,
                                     int seed, float scale, float offset_x, float offset_y,
                                     int octaves, float persistence, float lacunarity,
                                     int tileable)
{
    size_t gid = get_global_id(0);
    int x = X(gid);
    int y = Y(gid);

    if (x >= width || y >= height) return;

    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = scale;
    float max_value = 0.0f;

    for (int octave = 0; octave < octaves; octave++) {
        float nx = (x / (float)width) * frequency + offset_x;
        float ny = (y / (float)height) * frequency + offset_y;

        int tile_w = (int)(width / frequency);
        int tile_h = (int)(height / frequency);
        if (tile_w < 1) tile_w = 1;
        if (tile_h < 1) tile_h = 1;

        float noise = simplex2d_single(nx, ny, seed + octave, tileable, tile_w, tile_h);
        total += noise * amplitude;

        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    float value = (total / max_value + 1.0f) * 0.5f;
    value = fminf(fmaxf(value, 0.0f), 1.0f);  // clamp

    output[gid] = value;
}

// ============================================================================
// 3D Simplex Noise Kernel (CUDA - Row-major)
// ============================================================================

extern "C" __global__ void simplex3d(int width, int height, int depth, float *output,
                                     int seed, float scale, float offset_x, float offset_y, float offset_z,
                                     int octaves, float persistence, float lacunarity)
{
    size_t gid = get_global_id(0);

    // Decode 3D coordinates from linear index (row-major)
    int z = gid / (width * height);
    int remainder = gid % (width * height);
    int x = X(remainder);
    int y = Y(remainder);

    if (x >= width || y >= height || z >= depth) return;

    float total = 0.0f;
    float amplitude = 1.0f;
    float frequency = scale;
    float max_value = 0.0f;

    for (int octave = 0; octave < octaves; octave++) {
        float nx = (x / (float)width) * frequency + offset_x;
        float ny = (y / (float)height) * frequency + offset_y;
        float nz = (z / (float)depth) * frequency + offset_z;

        float noise = simplex3d_single(nx, ny, nz, seed + octave);
        total += noise * amplitude;

        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    float value = (total / max_value + 1.0f) * 0.5f;
    value = fminf(fmaxf(value, 0.0f), 1.0f);  // clamp

    output[gid] = value;
}
