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

#define DXDY2
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


// Kernels for operating on HxWx4 arrays
// array[,,3] is the "alpha" channel
// Colors are [N,4] where color[,3] is the alpha

#define ONES ONE4
#define ZEROS ZERO4

// Porter-Duff "over" at opacity e, as the OCL/Metal kernels blend:
// rgb = src.rgb*e + dst.rgb*(1-e), alpha = e + dst.a*(1-e)
__device__ __forceinline__ float4 blend_over(float4 dst, float4 src, float e) {
  return make_float4(src.x * e + dst.x * (1.0f - e),
                     src.y * e + dst.y * (1.0f - e),
                     src.z * e + dst.z * (1.0f - e),
                     e + dst.w * (1.0f - e));
}

extern "C" __kernel void lines(int width, int height, READWRITE float4 *a, int nelements,
                    READONLY float2 *coords, READONLY float *rdist,
                    READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  if (i >= (size_t)width * height) return;
  int x = X(i);
  int y = Y(i);
  float4 color = a[i];

  float2 point = make_float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float threshold = ((*rdist++ * .5) + .5);
    float aa_width = *aa_widths++;
    float threshold2 = threshold * threshold;
    float2 linestart = *coords++;
    float2 lineend = *coords++;
    float4 rgba = *colors++;
    float linemag2 = dxdy2(lineend, linestart);
    float2 delta = lineend - linestart;
    float2 u2 = (point - linestart) * (delta);
    // A zero-length line is a dot at its start (as numbax draws it); 0/0 would skip it
    float u = linemag2 > 0.0f ? (u2.x + u2.y) / linemag2 : 0.0f;
    float2 intersection = linestart + (u * delta);
    float dist2 = dxdy2(point,intersection);

    if (u >= 0.0f && u <= 1.0f) {
      if (aa_width > 0.0f) {
        // Anti-aliased edge using smoothstep
        float dist = sqrt(dist2);
        float edge_outer = threshold + aa_width;
        float edge_inner = threshold - aa_width;
        float t = smoothstep(edge_outer, edge_inner, dist);
        color = blend_over(color, rgba, t * rgba.w);
      } else {
        // Hard edge (original behavior, fast path)
        if (dist2 < threshold2) color = blend_over(color, rgba, rgba.w);
      }
    }
  }
  a[i] = color;
}

extern "C" __kernel void rects(int width, int height, READWRITE float4 *a, int nelements,
                    READONLY float4 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  if (i >= (size_t)width * height) return;
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 region = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

    float l = min(region.x, region.z);
    float b = min(region.y, region.w);
    float r = max(region.x, region.z);
    float t = max(region.y, region.w);

    bool inside = (l <= x && x <= r && b <= y && y <= t);

    if (aa_width > 0.0f && inside) {
      // Anti-aliased edges using distance to nearest edge
      float dist_l = (float)x - l;
      float dist_r = r - (float)x;
      float dist_b = (float)y - b;
      float dist_t = t - (float)y;
      float edge_dist = min(min(dist_l, dist_r), min(dist_b, dist_t));

      float blend_t = smoothstep(0.0f, aa_width, edge_dist);
      pixel = blend_over(pixel, rgba, blend_t * rgba.w);
    } else if (inside) {
      // Hard edge (original behavior, fast path)
      pixel = blend_over(pixel, rgba, rgba.w);
    }
  }
  a[i] = pixel;
}

extern "C" __kernel void circles(int width, int height, READWRITE float4 *a, int nelements,
                      READONLY float *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  if (i >= (size_t)width * height) return;
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  float2 point = make_float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float cx = *coords++;
    float cy = *coords++;
    float radius = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;
    float2 center = make_float2(cx, cy);
    float dist = sqrt(dxdy2(center, point));

    if (aa_width > 0.0f) {
      // Anti-aliased edge using smoothstep
      float edge_outer = radius + aa_width;
      float edge_inner = radius - aa_width;
      float t = smoothstep(edge_outer, edge_inner, dist);
      pixel = blend_over(pixel, rgba, t * rgba.w);
    } else {
      // Hard edge (original behavior, fast path)
      if (dist <= radius) pixel = blend_over(pixel, rgba, rgba.w);
    }
  }
  a[i] = pixel;
}

#define signex(p1, p2, p3)                                                       ((p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y))

// Triangle fill rule. A pixel exactly on an edge shared by two triangles belongs to
// only one of them, so translucent fills are not blended twice along the seam.
// tri_edge is exactly antisymmetric in (a, b): it is computed with the endpoints in a
// canonical order, so the two triangles get exactly opposite values on a shared edge.
static __device__ __forceinline__ float tri_edge(float ax, float ay, float bx, float by, float px, float py) {
  bool swap = (bx < ax) || (bx == ax && by < ay);
  float sx = swap ? bx : ax, sy = swap ? by : ay;
  float ex = swap ? ax : bx, ey = swap ? ay : by;
  float v = (ex - sx) * (py - sy) - (ey - sy) * (px - sx);
  return swap ? -v : v;
}

// With both triangles wound the same way, a shared edge runs in opposite directions,
// and exactly one direction owns pixels lying on it
static __device__ __forceinline__ bool tri_owns(float dx, float dy) {
  return dy > 0.0f || (dy == 0.0f && dx > 0.0f);
}

static __device__ __forceinline__ bool tri_inside(float px, float py, float x1, float y1, float x2, float y2,
                    float x3, float y3) {
  float area = tri_edge(x1, y1, x2, y2, x3, y3);
  if (area == 0.0f) return false;
  float s = area > 0.0f ? 1.0f : -1.0f;  // normalise the winding
  float e1 = s * tri_edge(x1, y1, x2, y2, px, py);
  float e2 = s * tri_edge(x2, y2, x3, y3, px, py);
  float e3 = s * tri_edge(x3, y3, x1, y1, px, py);
  return (e1 > 0.0f || (e1 == 0.0f && tri_owns(s * (x2 - x1), s * (y2 - y1)))) &&
         (e2 > 0.0f || (e2 == 0.0f && tri_owns(s * (x3 - x2), s * (y3 - y2)))) &&
         (e3 > 0.0f || (e3 == 0.0f && tri_owns(s * (x1 - x3), s * (y1 - y3))));
}

// Helper function to compute distance from point to line segment
__device__ static inline float dist_to_segment(float2 p, float2 a, float2 b) {
  float2 pa = p - a;
  float2 ba = b - a;
  float h = fmaxf(0.0f, fminf(1.0f, dot(pa, ba) / dot(ba, ba)));
  return length(pa - h * ba);
}

extern "C" __kernel void triangles(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY float2 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  if (i >= (size_t)width * height) return;
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = make_float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float2 v1 = *coords++;
    float2 v2 = *coords++;
    float2 v3 = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

    float d1 = signex(point, v1, v2);
    float d2 = signex(point, v2, v3);
    float d3 = signex(point, v3, v1);
    bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    bool has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    bool inside = !(has_neg && has_pos);

    if (aa_width > 0.0f && inside) {
      // Anti-aliased edges using distance to nearest edge
      float dist1 = dist_to_segment(point, v1, v2);
      float dist2 = dist_to_segment(point, v2, v3);
      float dist3 = dist_to_segment(point, v3, v1);
      float edge_dist = min(min(dist1, dist2), dist3);

      float blend_t = smoothstep(0.0f, aa_width, edge_dist);
      pixel = blend_over(pixel, rgba, blend_t * rgba.w);
    } else if (inside) {
      // Hard edge (original behavior, fast path)
      pixel = blend_over(pixel, rgba, rgba.w);
    }
  }
  a[i] = pixel;
}

// mesh_triangles: triangles with a top-left fill rule, for meshes (a triangulated
// polygon). A pixel exactly on an edge shared by two triangles belongs to only one
// of them, so translucent fills are not blended twice along the seams. Unlike
// triangles, pixels on a triangle's non-owning edges are not drawn.
extern "C" __kernel void mesh_triangles(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY float2 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  if (i >= (size_t)width * height) return;
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = make_float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float2 v1 = *coords++;
    float2 v2 = *coords++;
    float2 v3 = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

    bool inside = tri_inside(point.x, point.y, v1.x, v1.y, v2.x, v2.y, v3.x, v3.y);

    if (aa_width > 0.0f && inside) {
      // Anti-aliased edges using distance to nearest edge
      float dist1 = dist_to_segment(point, v1, v2);
      float dist2 = dist_to_segment(point, v2, v3);
      float dist3 = dist_to_segment(point, v3, v1);
      float edge_dist = min(min(dist1, dist2), dist3);

      float blend_t = smoothstep(0.0f, aa_width, edge_dist);
      pixel = blend_over(pixel, rgba, blend_t * rgba.w);
    } else if (inside) {
      // Hard edge (original behavior, fast path)
      pixel = blend_over(pixel, rgba, rgba.w);
    }
  }
  a[i] = pixel;
}

// (x-h)^2/a^2 + (y-k)^2/b^2 <= 1
extern "C" __kernel void ellipses(int width, int height, READWRITE float4 *a, int nelements,
                       READONLY float4 *coords, READONLY float *aa_widths, READONLY float4 *colors) {
  size_t i = get_global_id(0);
  if (i >= (size_t)width * height) return;
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);
  float2 point = make_float2((float)x, (float)y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    float4 center_axes = *coords++;
    float aa_width = *aa_widths++;
    float4 rgba = *colors++;

    float2 center = make_float2(center_axes.x,center_axes.y);
    float2 axes = make_float2(center_axes.z,center_axes.w);
    float2 delta = (point - center) / axes;
    float d = length(delta);

    if (aa_width > 0.0f) {
      // Anti-aliased edge using smoothstep
      // d < 1: inside, d = 1: on boundary, d > 1: outside
      float avg_radius = (axes.x + axes.y) * 0.5f;
      float dist_to_edge = (d - 1.0f) * avg_radius;
      float edge_outer = aa_width;
      float edge_inner = -aa_width;
      float t = smoothstep(edge_outer, edge_inner, dist_to_edge);
      pixel = blend_over(pixel, rgba, t * rgba.w);
    } else {
      // Hard edge (original behavior, fast path)
      if (d <= 1.0f) pixel = blend_over(pixel, rgba, rgba.w);
    }
  }
  a[i] = pixel;
}

extern "C" __kernel void composite(int width, int height, READWRITE float4 *a, int nelements,
                        READONLY int4 *coords, READWRITE float4 *pixels,
                        READONLY int *ipixels, int npixels) {
  size_t i = get_global_id(0);
  if (i >= (size_t)width * height) return;
  float4 pixel = a[i];
  int x = X(i);
  int y = Y(i);

  int2 xy = make_int2(x,y);
  for (int ielement = 0; ielement < nelements; ielement++) {
    // Get the x,y,w,h of the destination
    int4 coord = *coords++;
    int ipixel = *ipixels++;

    int2 lb = make_int2(coord.x, coord.y);
    int2 yt = lb + make_int2(coord.z, coord.w);
    // Is the current xy inside the destination area?
    int outside = ANYISLESS(xy,lb) + ANYISGTE(xy,yt);
    int2 source_location = xy - lb;
    // CUDA uses row-major order, so Y varies along outer dimension
    int source_offset = (source_location.y * coord.z) + source_location.x;
    int ipixel_source = ipixel + source_offset;
    int inside = !outside && 0 <= ipixel_source && ipixel_source < npixels;

    float4 rgba = inside ? pixels[ipixel_source] : ZEROS;
    if (inside) pixel = blend_over(pixel, rgba, rgba.w);
  }
  a[i] = pixel;
}
