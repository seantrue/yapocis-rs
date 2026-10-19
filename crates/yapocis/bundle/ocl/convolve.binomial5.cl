/* One dimensional convolution with a kernel conv that is determined at compile time */


#define FAST 1000
__kernel
void binomial5(int width,  __global float* a, __global float* ret )
{
    size_t i = get_global_id(0);
    float sum;
    size_t right = width-2;
    sum = 0.0;
    if ((i > (size_t)2) &&  (i < right)) {
       sum = sum + a[i+(-2)] * 0.0625;
       sum = sum + a[i+(-1)] * 0.25;
       sum = sum + a[i+(0)] * 0.375;
       sum = sum + a[i+(1)] * 0.25;
       sum = sum + a[i+(2)] * 0.0625;
    }
    ret[i] = sum;
    return;
}

__kernel void binomial5_fast(int width,  __global float* a, __global float* ret, int chunk)
{
    size_t i = get_global_id(0) * chunk;
    float sum;
    size_t right = width-2;
    volatile __local float buffer[FAST];
    for (int j = 0; j < 5; j++) {
        buffer[j] = i+j >= width ? 0.0 : a[i+j];
    }
    for (int j = 0; j < chunk-5; j++) {
        sum = 0.0;
        int ipixel = i+j >= right ? right-1 : i+j;
        if ((i > (size_t)2) &&  (i < right)) {
           sum = sum + a[j] * 0.0625;
           sum = sum + a[j] * 0.25;
           sum = sum + a[j] * 0.375;
           sum = sum + a[j] * 0.25;
           sum = sum + a[j] * 0.0625;
        }
        ret[ipixel] = sum;
    }
    return;
}
