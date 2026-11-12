/* One dimensional convolution with a kernel conv that is determined at compile time */


__kernel
void box5(int width,  __global float* a, __global float* ret )
{
    size_t i = get_global_id(0);
    float sum;
    size_t right = width-2;
    sum = 0.0;
    if ((i > (size_t)2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.2;
       sum = sum + a[i+(-1)] * 0.2;
       sum = sum + a[i+(0)] * 0.2;
       sum = sum + a[i+(1)] * 0.2;
       sum = sum + a[i+(2)] * 0.2;
    }
    ret[i] = sum;
    return;
}

/* Work item g computes ret[g*chunk .. g*chunk+chunk-1], each pixel exactly as box5
   computes it, and stops at width. Needs a global size of at least ceil(width/chunk). */
__kernel void box5_fast(int width,  __global float* a, __global float* ret, int chunk)
{
    int n = chunk > 0 ? chunk : 1;
    size_t right = width-2;
    size_t start = get_global_id(0) * (size_t)n;
    for (int j = 0; j < n; j++) {
        size_t i = start + (size_t)j;
        if (i >= (size_t)width) return;
        float sum = 0.0;
        if ((i > (size_t)2) &&  (i < right)) {
           sum = sum + a[i+(-2)] * 0.2;
           sum = sum + a[i+(-1)] * 0.2;
           sum = sum + a[i+(0)] * 0.2;
           sum = sum + a[i+(1)] * 0.2;
           sum = sum + a[i+(2)] * 0.2;
        }
        ret[i] = sum;
    }
    return;
}
