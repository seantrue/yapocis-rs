/* Kernels to compute direction and magnitude of gradient of an image. */



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

/* Compute gradient for images with 1,2,3,4 channels per pixel. */

__kernel
void gradient(int width, int height, __global FLOAT* input, int reach, __global FLOAT* grad, __global FLOAT *theta )
{
    size_t i = get_global_id(0);
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
    grad[i] = isnan(gr) ? ZERO: gr;
    theta[i] = isnan(th) ? ZERO: th;
}
__kernel
void gradient2(int width, int height, __global FLOAT2* input, int reach, __global FLOAT2* grad, __global FLOAT2 *theta )
{
    size_t i = get_global_id(0);
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
    grad[i] = isnan(gr) ? ZERO2: gr;
    theta[i] = isnan(th) ? ZERO2: th;
}
__kernel
void gradient3(int width, int height, __global FLOAT3* input, int reach, __global FLOAT3* grad, __global FLOAT3 *theta )
{
    size_t i = get_global_id(0);
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
    grad[i] = isnan(gr) ? ZERO3: gr;
    theta[i] = isnan(th) ? ZERO3: th;
}
__kernel
void gradient4(int width, int height, __global FLOAT4* input, int reach, __global FLOAT4* grad, __global FLOAT4 *theta )
{
    size_t i = get_global_id(0);
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
    grad[i] = isnan(gr) ? ZERO4: gr;
    theta[i] = isnan(th) ? ZERO4: th;
}
