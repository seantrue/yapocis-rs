// https://stackoverflow.com/questions/15095909/from-rgb-to-hsv-in-opengl-glsl
// For color math, see: http://www.brucelindbloom.com/index.html?Math.html
// For white points, see: https://en.wikipedia.org/wiki/Standard_illuminant



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

#define DOT3
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

#define eps 1.0e-10


#define RGBHSIK ((float4)(0.0f, -1.0f / 3.0f, 2.0f / 3.0f, -1.0f))
__inline float3 _rgb2hsi(float3 c) {
    float4 p = c.y < c.z ? (float4)(c.zy,RGBHSIK.wz) : (float4)(c.yz, RGBHSIK.xy);
    float4 q = c.x < p.x ? (float4)(p.xyw, c.x) : (float4)(c.x, p.yzx);
    float d = q.x - min(q.w, q.y);
    float hh = q.z + (q.w - q.y) / (6.0f * d + eps);
    hh = hh < 0.0 ? -hh : hh;
    float3 hsi = (float3)(hh , d / (q.x + eps), q.x);
    hsi = clamp(hsi, 0.0f, 1.0f-eps);
    return hsi;
}

#define HSIRGBK ((float4)(1.0f, 1.0f / 3.0f, 2.0f / 3.0f, 3.0f))
__inline float3 _hsi2rgb(float3 hsi) {
    float3 floorv;
    float3 p = fabs(fract(hsi.xxx + HSIRGBK.xyz, &floorv) * 6.0f - HSIRGBK.www);
    float3 c =  hsi.z * mix(HSIRGBK.xxx, clamp(p - HSIRGBK.xxx, 0.0f, 1.0f), hsi.y);
    return c;
}




#define XK (float3)(.4124,.3576,.1805)
#define YK (float3)(.2126,.7152,.0722)
#define ZK (float3)(.0193,.1192,.9505)
#define REFERENCE_WHITE (float3)(95.047,100.0,108.883)

static __inline float adjustrgb(float v) {
    v = v <= .04045f ? v/12.92f : native_powr((v + 0.055f) / 1.055f, 2.4f);
    return v*100.0;
}

static __inline float3 _rgb2xyz(float3 rgb) {
   float3 rgb3 = ZERO3;
   rgb3.x = adjustrgb(rgb.x);
   rgb3.y = adjustrgb(rgb.y);
   rgb3.z = adjustrgb(rgb.z);
   float X = dot3(rgb3,XK);
   float Y = dot3(rgb3,YK);
   float Z = dot3(rgb3,ZK);
   float3 XYZ = (float3)(X,Y,Z);
   XYZ /= REFERENCE_WHITE;
   return XYZ;
}

static __inline float adjustxyz(float v) {
    v = v <= .008856f ? (7.787f * v) + (16.f / 116.f) : native_powr(v, 0.3333333333333333f);
    return v;
}

static __inline float3 _xyz2lab(float3 xyz) {
   float3 xyz3 = ZERO3;
   xyz3.x = adjustxyz(xyz.x);
   xyz3.y = adjustxyz(xyz.y);
   xyz3.z = adjustxyz(xyz.z);
   float3 lab = ZERO3;
   lab.x = (116*xyz3.y) -16;
   lab.y = 500 * (xyz3.x - xyz3.y);
   lab.z = 200 * (xyz3.y-xyz3.z);
   return lab;
}


__kernel
void rgb2hsi(__global float *ra, __global float *ga, __global float *ba, __global float *ha, __global float *sa, __global float *ia) {
    size_t i = get_global_id(0);
    float3 c = (float3)(ra[i], ba[i], ga[i]);
    float3 hsi = _rgb2hsi(c);
    ha[i] = hsi.x;
    sa[i] = hsi.y;
    ia[i] = hsi.z;
}


__kernel	
void hsi2rgb(__global float *ha, __global float *sa, __global float *ia, __global float *ra, __global float *ga, __global float *ba) {
	size_t i = get_global_id(0);
	float3 hsi = (float3)(ha[i],sa[i],ia[i]);
    float3 c = _hsi2rgb(hsi);
    ra[i] = c.x;
    ga[i] = c.y;
    ba[i] = c.z;
}



__kernel
void rgb2hsi3(int depth, __global float *input, __global float *output) {
    size_t i = get_global_id(0) * depth;
    float3 c = (float3)(input[i], input[i+2],input[i+1]);
    float3 hsi = _rgb2hsi(c);
    output[i] =   hsi.x;
    output[i+1] = hsi.y;
    output[i+2]=  hsi.z;
    if (depth > 3) {
        output[i+3] = input[i+3];  // Copy alpha channel through
    }
}


__kernel
void hsi2rgb3(int depth, __global float *input, __global float *output) {
	size_t i = get_global_id(0)*depth;
	float3 hsi = (float3)(input[i],input[i+1],input[i+2]);
    float3 c = _hsi2rgb(hsi);
    output[i] =   c.x;
    output[i+1] = c.y;
    output[i+2] = c.z;
    if (depth > 3) {
        output[i+3] = input[i+3];  // Copy alpha channel through
    }
}

__kernel
void rgb2lab3(int depth, __global float *input, __global float *output) {
    size_t i = get_global_id(0) * depth;
    float3 c = (float3)(input[i], input[i+2],input[i+1]);
    float3 xyz = _rgb2xyz(c);
    float3 lab = _xyz2lab(xyz);
    output[i] =   lab.x;
    output[i+1] = lab.y;
    output[i+2]=  lab.z;
}


//__kernel
//void lab2rgb3(int depth, __global float *input, __global float *output) {
// size_t i = get_global_id(0)*depth;
//	float3 lab = (float3)(input[i],input[i+1],input[i+2]);
//    float3 c = _lab2rgb(lab);
//    output[i] =   c.x;
//    output[i+1] = c.y;
//    output[i+2] = c.z;
//}
