/* Kernels to implement pixel-wise operations on images */

__kernel
void add(__global float* input1, __global float* input2, __global float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] + input2[i];
}
__kernel
void sub(__global float* input1, __global float* input2, __global float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] - input2[i];
}
__kernel
void mul(__global float* input1, __global float* input2, __global float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] * input2[i];
}
__kernel
void divop(__global float* input1, __global float* input2, __global float *output)
{
    size_t i;
    i = get_global_id(0);
    output[i] = input1[i] / input2[i];
}
