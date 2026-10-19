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


/* CUDA FFT Kernels - Production Implementation */

#define M_PI 3.14159265358979323846f
#define M_2PI 6.28318530717958647692f

/* Complex multiplication */
__device__ __inline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/* Complex addition */
__device__ __inline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

/* Complex subtraction */
__device__ __inline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

/* Bit reversal for FFT */
__device__ __inline__ uint bit_reverse(uint x, uint bits) {
    uint result = 0;
    for (uint i = 0; i < bits; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

/* Compute twiddle factor W_N^k = e^(-i*2π*k/N) for forward FFT
 * For inverse FFT (direction = -1), we use the conjugate: e^(+i*2π*k/N)
 */
__device__ __inline__ float2 twiddle_factor(uint k, uint N, int direction) {
    // Forward FFT: direction = 1,  angle = -2πk/N (negative)
    // Inverse FFT: direction = -1, angle = +2πk/N (positive)
    float angle = -direction * M_2PI * (float)k / (float)N;
    return make_float2(cosf(angle), sinf(angle));
}

/* Optimized Radix-2 FFT using shared memory and barriers
 * Complexity: O(N log N) vs O(N²) for DFT
 *
 * This implements the Cooley-Tukey decimation-in-time algorithm with:
 * - Shared memory for efficient communication
 * - Parallel butterfly operations
 * - Proper synchronization barriers
 * - In-place computation
 *
 * Block size must equal FFT size (for sizes up to 1024)
 * direction: 1 for forward FFT, -1 for inverse FFT
 */
extern "C" __global__ void fft1d(float2 *input, int n, float2 *output, int direction)
{
    __shared__ float2 shared_data[1024];  // Static shared memory allocation

    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint lid = threadIdx.x;

    if (gid >= n) return;

    // Compute log2(n)
    uint log2n = 0;
    uint temp = n;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }

    // Bit-reversed input loading into shared memory
    uint rev_idx = bit_reverse(lid, log2n);

    // Use optimized FFT only if we have exactly one block of size n
    // This is the only configuration where the shared memory algorithm works correctly
    if (n <= 1024 && gridDim.x == 1 && blockDim.x == n) {
        // Optimized path for small FFTs (≤ 1024) with correct block config
        shared_data[lid] = input[rev_idx];
        __syncthreads();

        // Cooley-Tukey FFT: log2(n) stages
        for (uint stage = 0; stage < log2n; stage++) {
            uint m = 1 << (stage + 1);  // Butterfly span
            uint m2 = m >> 1;            // Half span

            // Calculate position in butterfly
            uint k = lid % m2;           // Twiddle index
            uint block = (lid / m) * m;  // Start of butterfly block
            uint j = block + k;          // Upper butterfly index

            // Only process if within bounds
            if (j + m2 < n) {
                float2 w = twiddle_factor(k, m, direction);
                float2 t = cmul(w, shared_data[j + m2]);
                float2 u = shared_data[j];

                shared_data[j] = cadd(u, t);
                shared_data[j + m2] = csub(u, t);
            }

            __syncthreads();
        }

        // Normalize for inverse FFT
        float2 result = shared_data[lid];
        if (direction == -1) {
            result.x /= (float)n;
            result.y /= (float)n;
        }

        output[gid] = result;
    } else {
        // Fallback to DFT for sizes > 1024
        float2 sum = make_float2(0.0f, 0.0f);
        for (uint k = 0; k < n; k++) {
            float angle = -direction * M_2PI * gid * k / (float)n;
            float2 w = make_float2(cosf(angle), sinf(angle));
            sum = cadd(sum, cmul(input[k], w));
        }

        if (direction == -1) {
            sum.x /= (float)n;
            sum.y /= (float)n;
        }

        output[gid] = sum;
    }
}

/* Window function kernel
 * window_type: 0=rectangular, 1=hann, 2=hamming, 3=blackman
 */
extern "C" __global__ void window(float *input, int const window_type, float *output GLOBAL_SIZE_PARAM)
{
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    RETURN_IF_OUTSIDE(gid)
    // The real length, not the block-rounded thread count
    uint n = (uint)yapocis_global_size;

    float w = 1.0f;
    float pos = (float)gid / (n - 1);

    switch (window_type) {
        case 1: // Hann window
            w = 0.5f * (1.0f - cosf(M_2PI * pos));
            break;
        case 2: // Hamming window
            w = 0.54f - 0.46f * cosf(M_2PI * pos);
            break;
        case 3: // Blackman window
            w = 0.42f - 0.5f * cosf(M_2PI * pos) + 0.08f * cosf(4.0f * M_PI * pos);
            break;
        default: // Rectangular (no windowing)
            w = 1.0f;
            break;
    }

    output[gid] = input[gid] * w;
}

/* Power spectrum kernel - compute magnitude squared */
extern "C" __global__ void power_spectrum(float2 *fft_output, float *power GLOBAL_SIZE_PARAM)
{
    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    // The real length, not the block-rounded thread count
    uint n = (uint)yapocis_global_size;

    if (gid >= n) return;

    float2 val = fft_output[gid];
    power[gid] = val.x * val.x + val.y * val.y;
}

/* Batch 1D FFT - Process multiple FFTs in a single kernel launch
 *
 * This kernel processes multiple independent FFTs simultaneously, amortizing
 * the GPU launch overhead across many FFTs. This is crucial for performance
 * when processing many small FFTs (e.g., spectrograms, batch signal processing).
 *
 * Performance improvement: 10-100x for batches compared to individual launches.
 *
 * Organization:
 * - Grid: (batch_size * fft_size) total threads
 * - Blocks: batch_size blocks, each of size fft_size
 * - Each block processes one independent FFT
 *
 * Memory:
 * - Input/output are 2D arrays: [batch_size][fft_size]
 * - Each block uses shared memory for its FFT
 * - No inter-block communication needed
 *
 * Limitations:
 * - fft_size must be power of 2
 * - fft_size <= 256 (current radix-2 implementation limit)
 * - batch_size limited by available GPU memory
 */
extern "C" __global__ void fft1d_batch(
    float2 *input,
    int const batch_size,
    int const fft_size,
    int const direction,
    float2 *output
) {
    __shared__ float2 shared_data[256];  // Static shared memory allocation

    uint gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint lid = threadIdx.x;
    uint group_id = blockIdx.x;

    // Determine which batch this thread belongs to
    uint batch_idx = group_id;  // Each block handles one FFT
    uint n = fft_size;

    if (batch_idx >= batch_size || lid >= n) return;

    // Calculate offset into input/output arrays
    uint batch_offset = batch_idx * n;

    // Compute log2(n)
    uint log2n = 0;
    uint temp = n;
    while (temp > 1) {
        temp >>= 1;
        log2n++;
    }

    // Bit-reversed input loading into shared memory
    uint rev_idx = bit_reverse(lid, log2n);

    if (n <= 256) {
        // Optimized radix-2 FFT path
        shared_data[lid] = input[batch_offset + rev_idx];
        __syncthreads();

        // Cooley-Tukey FFT: log2(n) stages
        for (uint stage = 0; stage < log2n; stage++) {
            uint m = 1 << (stage + 1);  // Butterfly span
            uint m2 = m >> 1;            // Half span

            // Calculate position in butterfly
            uint k = lid % m2;           // Twiddle index
            uint block = (lid / m) * m;  // Start of butterfly block
            uint j = block + k;          // Upper butterfly index

            // Only process if within bounds
            if (j + m2 < n) {
                float2 w = twiddle_factor(k, m, direction);
                float2 t = cmul(w, shared_data[j + m2]);
                float2 u = shared_data[j];

                shared_data[j] = cadd(u, t);
                shared_data[j + m2] = csub(u, t);
            }

            __syncthreads();
        }

        // Normalization for inverse FFT
        float2 result = shared_data[lid];
        if (direction == -1) {
            result.x /= (float)n;
            result.y /= (float)n;
        }

        output[batch_offset + lid] = result;
    } else {
        // Fallback to DFT for sizes > 256
        float2 sum = make_float2(0.0f, 0.0f);
        for (uint k = 0; k < n; k++) {
            float angle = -direction * M_2PI * lid * k / (float)n;
            float2 w = make_float2(cosf(angle), sinf(angle));
            sum = cadd(sum, cmul(input[batch_offset + k], w));
        }

        if (direction == -1) {
            sum.x /= (float)n;
            sum.y /= (float)n;
        }

        output[batch_offset + lid] = sum;
    }
}
