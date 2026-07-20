#include <stdint.h>

#ifndef AICORE
#define AICORE [aicore]
#endif

extern "C" __global__ AICORE void
softmax__kernel0(__gm__ half *__restrict__ data,
                 __gm__ half *__restrict__ cast_1);

extern "C" void LaunchSoftmaxKernel(void *data, void *cast_1, void *stream) {
  softmax__kernel0<<<8, nullptr, stream>>>(
      (__gm__ half *)data, (__gm__ half *)cast_1);
}
