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


// Unified CNN CUDA Kernels
// Converted from OpenCL to CUDA
// Combines VGG16 and modern CNN layers into a comprehensive deep learning toolkit

// CUDA-specific definitions
#ifndef UINT32_MAX
#define UINT32_MAX 0xFFFFFFFFu
#endif

// =================================================================
// Convolutional Layers
// =================================================================

// 3x3 Convolution (VGG16, ResNet, most CNNs)
extern "C" __global__ void conv2d_3x3(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint out_w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint out_c = (blockIdx.z * blockDim.z + threadIdx.z);

    if (out_h >= input_height || out_w >= input_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // 3x3 convolution with padding
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (uint kh = 0; kh < 3; kh++) {
            for (uint kw = 0; kw < 3; kw++) {
                int in_h = (int)out_h + (int)kh - 1;  // padding = 1
                int in_w = (int)out_w + (int)kw - 1;  // padding = 1

                // Check bounds (zero padding)
                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {

                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = out_c * input_channels * 9 +
                                     in_c * 9 + kh * 3 + kw;

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// Optimized 3x3 Convolution (simplified version)
extern "C" __global__ void conv2d_3x3_optimized(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint out_w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint out_c = (blockIdx.z * blockDim.z + threadIdx.z);

    if (out_h >= input_height || out_w >= input_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // 3x3 convolution with padding
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (int kh = -1; kh <= 1; kh++) {
            for (int kw = -1; kw <= 1; kw++) {
                int in_h = (int)out_h + kh;
                int in_w = (int)out_w + kw;

                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {

                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = out_c * input_channels * 9 +
                                     in_c * 9 +
                                     (uint)(kh + 1) * 3 + (uint)(kw + 1);

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// 1x1 Convolution (ResNet, MobileNet)
extern "C" __global__ void conv2d_1x1(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint out_w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint out_c = (blockIdx.z * blockDim.z + threadIdx.z);

    if (out_h >= input_height || out_w >= input_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // 1x1 convolution is just a weighted sum across input channels
    for (uint in_c = 0; in_c < input_channels; in_c++) {
        uint input_idx = in_c * input_height * input_width +
                        out_h * input_width + out_w;
        uint weight_idx = out_c * input_channels + in_c;

        sum += input[input_idx] * weights[weight_idx];
    }

    uint output_idx = out_c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// Depthwise 3x3 Convolution (MobileNet, EfficientNet)
extern "C" __global__ void depthwise_conv2d_3x3(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint out_w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint c = (blockIdx.z * blockDim.z + threadIdx.z);

    if (out_h >= input_height || out_w >= input_width || c >= channels)
        return;

    float sum = bias[c];

    // 3x3 depthwise convolution - only convolve within the same channel
    for (uint kh = 0; kh < 3; kh++) {
        for (uint kw = 0; kw < 3; kw++) {
            int in_h = (int)out_h + (int)kh - 1;  // padding = 1
            int in_w = (int)out_w + (int)kw - 1;  // padding = 1

            if (in_h >= 0 && in_h < (int)input_height &&
                in_w >= 0 && in_w < (int)input_width) {

                uint input_idx = c * input_height * input_width +
                                (uint)in_h * input_width + (uint)in_w;
                uint weight_idx = c * 9 + kh * 3 + kw;

                sum += input[input_idx] * weights[weight_idx];
            }
        }
    }

    uint output_idx = c * input_height * input_width +
                      out_h * input_width + out_w;
    output[output_idx] = sum;
}

// =================================================================
// Activation Functions
// =================================================================

// ReLU Activation
extern "C" __global__ void relu_activation(
    const float* input,
    float* output,
    const uint size
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= size)
        return;

    output[gid] = fmaxf(0.0f, input[gid]);
}

// GELU Activation (BERT, GPT, transformers)
extern "C" __global__ void gelu(
    const float* input,
    float* output,
    const uint size
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= size)
        return;

    float x = input[gid];

    // GELU approximation: 0.5 * x * (1 + tanhf(√(2/π) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + 0.044715f * x_cubed);
    float gelu_val = 0.5f * x * (1.0f + tanhf(inner));

    output[gid] = gelu_val;
}

// Softmax Activation
extern "C" __global__ void softmax(
    const float* input,
    float* output,
    const uint size
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= size)
        return;

    // Find maximum for numerical stability
    float max_val = input[0];
    for (uint i = 1; i < size; i++) {
        max_val = fmaxf(max_val, input[i]);
    }

    // Calculate exponentials and sum
    float sum = 0.0f;
    for (uint i = 0; i < size; i++) {
        sum += expf(input[i] - max_val);
    }

    // Calculate softmax
    output[gid] = expf(input[gid] - max_val) / sum;
}

// Sigmoid Activation (RetinaNet, binary classification)
extern "C" __global__ void sigmoid(
    const float* input,
    float* output,
    const uint size
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= size)
        return;

    output[gid] = 1.0f / (1.0f + expf(-input[gid]));
}

// =================================================================
// Pooling Layers
// =================================================================

// 2x2 Max Pooling (VGG16, traditional CNNs)
extern "C" __global__ void max_pool_2x2(
    const float* input,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint out_w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint c = (blockIdx.z * blockDim.z + threadIdx.z);

    uint output_height = input_height / 2;
    uint output_width = input_width / 2;

    if (out_h >= output_height || out_w >= output_width || c >= channels)
        return;

    float max_val = -1e38f;

    // 2x2 pooling window
    for (uint kh = 0; kh < 2; kh++) {
        for (uint kw = 0; kw < 2; kw++) {
            uint in_h = out_h * 2 + kh;
            uint in_w = out_w * 2 + kw;

            uint input_idx = c * input_height * input_width +
                            in_h * input_width + in_w;

            max_val = fmaxf(max_val, input[input_idx]);
        }
    }

    uint output_idx = c * output_height * output_width +
                      out_h * output_width + out_w;
    output[output_idx] = max_val;
}

// 3x3 Max Pooling with stride=2, padding=1 (ResNet)
extern "C" __global__ void max_pool_3x3_stride2(
    const float* input,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint out_w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint c = (blockIdx.z * blockDim.z + threadIdx.z);

    uint output_height = input_height / 2;
    uint output_width = input_width / 2;

    if (out_h >= output_height || out_w >= output_width || c >= channels)
        return;

    float max_val = -1e38f;

    // 3x3 pooling window with stride=2, padding=1
    for (uint kh = 0; kh < 3; kh++) {
        for (uint kw = 0; kw < 3; kw++) {
            int in_h = (int)(out_h * 2) - 1 + (int)kh;
            int in_w = (int)(out_w * 2) - 1 + (int)kw;

            // Check bounds (padding)
            if (in_h >= 0 && in_h < (int)input_height &&
                in_w >= 0 && in_w < (int)input_width) {
                uint input_idx = c * input_height * input_width +
                                (uint)in_h * input_width + (uint)in_w;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
    }

    uint output_idx = c * output_height * output_width +
                      out_h * output_width + out_w;
    output[output_idx] = max_val;
}

// Global Average Pooling (ResNet, MobileNet)
extern "C" __global__ void global_avg_pool(
    const float* input,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint c = (blockIdx.x * blockDim.x + threadIdx.x);

    if (c >= channels)
        return;

    float sum = 0.0f;
    uint spatial_size = input_height * input_width;

    // Sum all spatial positions
    for (uint h = 0; h < input_height; h++) {
        for (uint w = 0; w < input_width; w++) {
            uint input_idx = c * spatial_size + h * input_width + w;
            sum += input[input_idx];
        }
    }

    // Average over spatial dimensions
    output[c] = sum / (float)spatial_size;
}

// Global Max Pooling
extern "C" __global__ void global_max_pool(
    const float* input,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    const uint c = (blockIdx.x * blockDim.x + threadIdx.x);

    if (c >= channels)
        return;

    float max_val = -1e38f;
    uint spatial_size = input_height * input_width;

    // Find max across all spatial positions
    for (uint h = 0; h < input_height; h++) {
        for (uint w = 0; w < input_width; w++) {
            uint input_idx = c * spatial_size + h * input_width + w;
            max_val = fmaxf(max_val, input[input_idx]);
        }
    }

    output[c] = max_val;
}

// =================================================================
// Normalization Layers
// =================================================================

// Batch Normalization
extern "C" __global__ void batch_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* variance,
    float* output,
    const uint height,
    const uint width,
    const uint channels,
    const float epsilon
) {
    const uint h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint c = (blockIdx.z * blockDim.z + threadIdx.z);

    if (h >= height || w >= width || c >= channels)
        return;

    uint idx = c * height * width + h * width + w;

    float normalized = (input[idx] - mean[c]) / sqrtf(variance[c] + epsilon);
    output[idx] = gamma[c] * normalized + beta[c];
}

// Layer Normalization
extern "C" __global__ void layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint channels,
    const float epsilon
) {
    const uint h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint w = (blockIdx.y * blockDim.y + threadIdx.y);

    if (h >= input_height || w >= input_width)
        return;

    // Compute mean across channels
    float mean = 0.0f;
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        mean += input[idx];
    }
    mean /= (float)channels;

    // Compute variance across channels
    float variance = 0.0f;
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        float diff = input[idx] - mean;
        variance += diff * diff;
    }
    variance /= (float)channels;

    // Normalize and apply scale/shift
    float std = sqrtf(variance + epsilon);
    for (uint c = 0; c < channels; c++) {
        uint idx = c * input_height * input_width + h * input_width + w;
        float normalized = (input[idx] - mean) / std;
        output[idx] = gamma[c] * normalized + beta[c];
    }
}

// 1D Layer Normalization (for transformers/sequences)
extern "C" __global__ void layer_norm_1d(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    const uint size,
    const float epsilon
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid > 0)
        return;

    // Compute mean
    float mean = 0.0f;
    for (uint i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= (float)size;

    // Compute variance
    float variance = 0.0f;
    for (uint i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)size;

    // Normalize and apply scale/shift
    float std = sqrtf(variance + epsilon);
    for (uint i = 0; i < size; i++) {
        float normalized = (input[i] - mean) / std;
        output[i] = gamma[i] * normalized + beta[i];
    }
}

// =================================================================
// Fully Connected Layers
// =================================================================

// Fully Connected (Dense) Layer
extern "C" __global__ void fully_connected(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_size,
    const uint output_size
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= output_size)
        return;

    float sum = bias[gid];

    for (uint i = 0; i < input_size; i++) {
        uint weight_idx = gid * input_size + i;
        sum += input[i] * weights[weight_idx];
    }

    output[gid] = sum;
}

// Matrix Multiplication with SIMD (simplified for OpenCL)
extern "C" __global__ void matrix_multiply_simd(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    const uint M,
    const uint N,
    const uint K
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    uint row = gid % M;
    uint col = gid / M;

    if (row >= M || col >= N)
        return;

    float sum = 0.0f;

    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum + bias[col];
}

// =================================================================
// Regularization
// =================================================================

// Linear congruential generator for random numbers
__device__ __inline__ uint lcg_random(uint seed) {
    return seed * 1664525u + 1013904223u;
}

// Dropout
extern "C" __global__ void dropout(
    const float* input,
    float* output,
    const float rate,
    const uint seed,
    const uint training,
    const uint size
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= size)
        return;

    // Inference mode: identity
    if (training == 0) {
        output[gid] = input[gid];
        return;
    }

    // Training mode: randomly drop elements
    uint random_state = lcg_random(seed + gid);
    float random_val = (float)random_state / (float)UINT32_MAX;

    if (random_val < rate) {
        output[gid] = 0.0f;
    } else {
        output[gid] = input[gid] / (1.0f - rate);
    }
}

// Spatial Dropout
extern "C" __global__ void spatial_dropout(
    const float* input,
    float* output,
    const uint channels,
    const uint height,
    const uint width,
    const float rate,
    const uint seed,
    const uint training
) {
    const uint h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint w = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint c = (blockIdx.z * blockDim.z + threadIdx.z);

    if (h >= height || w >= width || c >= channels)
        return;

    uint idx = c * height * width + h * width + w;

    // Inference mode: identity
    if (training == 0) {
        output[idx] = input[idx];
        return;
    }

    // Generate random value per channel
    uint random_state = lcg_random(seed + c);
    float random_val = (float)random_state / (float)UINT32_MAX;

    if (random_val < rate) {
        output[idx] = 0.0f;
    } else {
        output[idx] = input[idx] / (1.0f - rate);
    }
}
// =================================================================
// ResNet-specific Convolution Kernels
// =================================================================

// 7x7 Convolution with stride=2, padding=3
extern "C" __global__ void conv2d_7x7_stride2(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);  // height
    const uint out_c = (blockIdx.y * blockDim.y + threadIdx.y);  // channels
    const uint out_w = (blockIdx.z * blockDim.z + threadIdx.z);  // width

    const uint output_height = input_height / 2;
    const uint output_width = input_width / 2;

    if (out_h >= output_height || out_w >= output_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // Stride=2 means we sample every 2nd input pixel
    int in_h_start = (int)out_h * 2 - 3;  // padding=3
    int in_w_start = (int)out_w * 2 - 3;

    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (uint kh = 0; kh < 7; kh++) {
            for (uint kw = 0; kw < 7; kw++) {
                int in_h = in_h_start + (int)kh;
                int in_w = in_w_start + (int)kw;

                // Zero padding
                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {
                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = ((out_c * input_channels + in_c) * 7 + kh) * 7 + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * output_height * output_width + out_h * output_width + out_w;
    output[output_idx] = sum;
}

// 3x3 Convolution with stride=2, padding=1
extern "C" __global__ void conv2d_3x3_stride2(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);  // height
    const uint out_c = (blockIdx.y * blockDim.y + threadIdx.y);  // channels
    const uint out_w = (blockIdx.z * blockDim.z + threadIdx.z);  // width

    // Output size with padding=1, kernel=3, stride=2: (input - 1) / 2 + 1
    const uint output_height = (input_height - 1) / 2 + 1;
    const uint output_width = (input_width - 1) / 2 + 1;

    if (out_h >= output_height || out_w >= output_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    int in_h_start = (int)out_h * 2 - 1;  // padding=1
    int in_w_start = (int)out_w * 2 - 1;

    for (uint in_c = 0; in_c < input_channels; in_c++) {
        for (uint kh = 0; kh < 3; kh++) {
            for (uint kw = 0; kw < 3; kw++) {
                int in_h = in_h_start + (int)kh;
                int in_w = in_w_start + (int)kw;

                if (in_h >= 0 && in_h < (int)input_height &&
                    in_w >= 0 && in_w < (int)input_width) {
                    uint input_idx = in_c * input_height * input_width +
                                    (uint)in_h * input_width + (uint)in_w;
                    uint weight_idx = ((out_c * input_channels + in_c) * 3 + kh) * 3 + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    uint output_idx = out_c * output_height * output_width + out_h * output_width + out_w;
    output[output_idx] = sum;
}

// 1x1 Convolution with stride=2
extern "C" __global__ void conv2d_1x1_stride2(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint input_channels,
    const uint output_channels
) {
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);  // height
    const uint out_c = (blockIdx.y * blockDim.y + threadIdx.y);  // channels
    const uint out_w = (blockIdx.z * blockDim.z + threadIdx.z);  // width

    const uint output_height = input_height / 2;
    const uint output_width = input_width / 2;

    if (out_h >= output_height || out_w >= output_width || out_c >= output_channels)
        return;

    float sum = bias[out_c];

    // Stride=2: sample every 2nd pixel
    uint in_h = out_h * 2;
    uint in_w = out_w * 2;

    for (uint in_c = 0; in_c < input_channels; in_c++) {
        uint input_idx = in_c * input_height * input_width + in_h * input_width + in_w;
        uint weight_idx = out_c * input_channels + in_c;
        sum += input[input_idx] * weights[weight_idx];
    }

    uint output_idx = out_c * output_height * output_width + out_h * output_width + out_w;
    output[output_idx] = sum;
}

// =================================================================
// Element-wise Operations
// =================================================================

// Element-wise addition (for residual connections)
extern "C" __global__ void add_tensors(
    const float* input1,
    const float* input2,
    float* output,
    const uint size
) {
    const uint gid = (blockIdx.x * blockDim.x + threadIdx.x);

    if (gid >= size) return;

    output[gid] = input1[gid] + input2[gid];
}

// =================================================================
// Upsampling / Spatial Operations
// =================================================================

// 2x Bilinear Upsampling (FPN, U-Net, semantic segmentation)
extern "C" __global__ void upsample_2x(
    const float* input,
    float* output,
    const uint input_height,
    const uint input_width,
    const uint channels
) {
    // dispatch_size is (height_of output, depth_of output, width_of output)
    // which maps to (out_h, channels, out_w)
    const uint out_h = (blockIdx.x * blockDim.x + threadIdx.x);
    const uint c = (blockIdx.y * blockDim.y + threadIdx.y);
    const uint out_w = (blockIdx.z * blockDim.z + threadIdx.z);

    const uint output_height = input_height * 2;
    const uint output_width = input_width * 2;

    if (out_h >= output_height || c >= channels || out_w >= output_width)
        return;

    // Nearest neighbor interpolation (matches torchvision FPN)
    // Each input pixel maps to 4 output pixels in a 2×2 block
    uint in_h = out_h / 2;
    uint in_w = out_w / 2;

    // Read input pixel and replicate to output
    uint input_idx = c * input_height * input_width + in_h * input_width + in_w;
    uint output_idx = c * output_height * output_width + out_h * output_width + out_w;

    output[output_idx] = input[input_idx];
}
