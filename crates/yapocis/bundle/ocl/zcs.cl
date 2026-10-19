/* Kernels to find zero-crossings in an image. */



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

/* Find zero-crossings for pixels of 1-4 channels. */
#define ADDRESS1(xx,yy) (ADDRESS(xx,yy))*1

__kernel
void zcs1(int width, int height, float threshold, __global float* input, __global float* zcs )
{
    size_t i = get_global_id(0)*1;
    int x = max((size_t)1,X(i));
    x = min(x,width-2);
    int y = max((size_t)1,Y(i));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
}
#define ADDRESS2(xx,yy) (ADDRESS(xx,yy))*2

__kernel
void zcs2(int width, int height, float threshold, __global float* input, __global float* zcs )
{
    size_t i = get_global_id(0)*2;
    int x = max((size_t)1,X(i));
    x = min(x,width-2);
    int y = max((size_t)1,Y(i));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
}
#define ADDRESS3(xx,yy) (ADDRESS(xx,yy))*3

__kernel
void zcs3(int width, int height, float threshold, __global float* input, __global float* zcs )
{
    size_t i = get_global_id(0)*3;
    int x = max((size_t)1,X(i));
    x = min(x,width-2);
    int y = max((size_t)1,Y(i));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
}
#define ADDRESS4(xx,yy) (ADDRESS(xx,yy))*4

__kernel
void zcs4(int width, int height, float threshold, __global float* input, __global float* zcs )
{
    size_t i = get_global_id(0)*4;
    int x = max((size_t)1,X(i));
    x = min(x,width-2);
    int y = max((size_t)1,Y(i));
    y = min(y, height-2);
    float zclt;
    float v, below, left, above, right;
    float delta;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
    v = input[i];
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
    zcs[i] = zclt;
    i += 1;
}

__kernel
void sub(__global float* input1, __global float* input2, __global float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] - input2[i];
}
