/* Perona-Malik filter */
/*

The Perona-Malik (PM) filter algorithm is a type of edge-preserving, nonlinear diffusion filter used for image
processing. It was introduced by Pietro Perona and Jitendra Malik in 1990 ("Scale-space and edge
detection using anisotropic diffusion", IEEE Trans. PAMI 12(7)).

The PM filter algorithm is designed to remove noise from an image while preserving its edges. The algorithm works by
diffusing the image over time, with the diffusion process being regulated by a diffusion coefficient that varies
depending on the image gradient. In regions where the gradient is high, such as edges and boundaries, the diffusion
coefficient is small, which limits the amount of diffusion and preserves the edges. In regions where the gradient is
low, such as smooth, homogeneous regions, the diffusion coefficient is large, which allows more diffusion
and helps to reduce noise.

The PM filter algorithm is governed by the following partial differential equation:

∂I/∂t = div(c(x,y,t)∇I)

where I is the image, t is time, div is the divergence operator, ∇ is the gradient operator,
and c(x,y,t) is the diffusion coefficient at position (x,y) and time t. The diffusion coefficient c(x,y,t) is a function
 of the gradient magnitude of the image and a parameter that controls the strength of the diffusion.

The PM filter algorithm is applied iteratively over the image until the desired level of noise reduction is achieved.
The output of the algorithm is a smoothed image that preserves the edges of the original image.

Explanation by chatGPT 2023-02-15

*/



// Define memory usage constants
#define READONLY __global
#define READWRITE __global


// Define vector constants
#define ZERO 0.0f
#define ZERO2 (float2)(ZERO,ZERO)
#define ZERO3 (float3)(ZERO,ZERO,ZERO)
#define ZERO4 (float4)(ZERO,ZERO,ZERO,ZERO)
#define ONE 1.0f
#define ONE2 (float2)(ONE,ONE)
#define ONE3 (float3)(ONE,ONE,ONE)
#define ONE4 (float4)(ONE,ONE,ONE,ONE)
#define FLOAT float
#define FLOAT2 float2
#define FLOAT3 float3
#define FLOAT4 float4

#define eps 1.0e-10

/* Helpful macros and in-line functions */

// Address macros
#define ADDRESS(xx,yy) (((xx)*height)+(yy))

#define X(addr) ((addr)/height)
#define Y(addr) ((addr)%height)

#define SWAPINT(xxx,yyy) {int tmp = xxx; xxx=yyy; yyy=tmp;}
#define SWAPFLOAT(xxx,yyy) {float tmp = xxx; xxx=yyy; yyy=tmp;}

#define ABS(x) (((x) < 0) ? -(x) : (x))

#ifdef DXDY2
static __inline float dxdy2(float2 a, float2 b) {
    float2 dxdy =  (a-b)*(a-b);
    return dxdy.x + dxdy.y;
}
#endif

#ifdef DOT3
static __inline float dot3(float3 a, float3 b) {
   return (float) (a.x*b.x + a.y*b.y + a.z*b.z);
}
#endif

#ifdef VXM3
static __inline float3 vxm3(float3 v, float3 a, float3 b, float3 c) {
    float3 result = (float3) (dot3(v,a), dot3(v,b), dot3(v,c));
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

// Note: smoothstep(), mix(), dot(), and length() are built-in OpenCL functions

/* Perona-Malik filter for 1,2,3,4 channel images*/

__kernel 
void filterImage(int width, int height, __global FLOAT *input, float scale, float step_size, __global FLOAT *output)
{
    size_t i = get_global_id(0);
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
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel 
void filterImage2(int width, int height, __global FLOAT2 *input, float scale, float step_size, __global FLOAT2 *output)
{
    size_t i = get_global_id(0);
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
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel 
void filterImage3(int width, int height, __global FLOAT3 *input, float scale, float step_size, __global FLOAT3 *output)
{
    size_t i = get_global_id(0);
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
    barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel 
void filterImage4(int width, int height, __global FLOAT4 *input, float scale, float step_size, __global FLOAT4 *output)
{
    size_t i = get_global_id(0);
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
    barrier(CLK_GLOBAL_MEM_FENCE);
}
