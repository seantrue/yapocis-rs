/* One dimensional convolution with a kernel conv that is determined at compile time */


__kernel
void edge3(int width,  __global float* a, __global float* ret )
{
    size_t i = get_global_id(0);
    float sum;
    size_t right = width-1;
    sum = 0.0;
    if ((i > (size_t)1) &&  (i < right)) {
       sum = sum + a[i+(-1)] * -1.0;
       sum = sum + a[i+(0)] * 0.0;
       sum = sum + a[i+(1)] * 1.0;
    }
    ret[i] = sum;
    return;
}

/* Work item g computes ret[g*chunk .. g*chunk+chunk-1], each pixel exactly as edge3
   computes it, and stops at width. Needs a global size of at least ceil(width/chunk). */
__kernel void edge3_fast(int width,  __global float* a, __global float* ret, int chunk)
{
    int n = chunk > 0 ? chunk : 1;
    size_t right = width-1;
    size_t start = get_global_id(0) * (size_t)n;
    for (int j = 0; j < n; j++) {
        size_t i = start + (size_t)j;
        if (i >= (size_t)width) return;
        float sum = 0.0;
        if ((i > (size_t)1) &&  (i < right)) {
           sum = sum + a[i+(-1)] * -1.0;
           sum = sum + a[i+(0)] * 0.0;
           sum = sum + a[i+(1)] * 1.0;
        }
        ret[i] = sum;
    }
    return;
}
