//
// Created by yiak on 2021/6/11.
//
// CPU implementation of Farthest. In the last presentation, I have talked that there is no need to perform point wise
// searching in CUDA device. The number of points from a lidar frame is up to 50,000 points recorded by 64 lines LIDAR device.
// It is a small number for GPGPU.
//
// Here is benchmark test:
//             op name calls | total time span | avg time span | 50% timespan
//   CPU op
//   GPU op
//
#include "farthest_point_sampling_op.h"

namespace pp3d {
namespace operators {

__global__ void farthestPointSamplingKernel(int* shape, int size, int m, double* dataset_in_ptr, int* dataset_in_indices_ptr) {

}

#ifdef PADDLE_WITH_CUDA
// GPU Kernel
template<>
void farthestPointSampling<platform::CUDADeviceContext>(int* shape, int size, int m, double* dataset_in_ptr, int* dataset_in_indices_ptr) {
    farthestPointSamplingKernel<<<32, 512>>>(shape, size, m, dataset_in_ptr, dataset_in_indices_ptr);
}
#endif

void test_farthestPointSamplingKernel(int* shape, int size, int m, double* dataset_in_ptr, int* dataset_in_indices_ptr) {
    farthestPointSamplingKernel<<<32, 512>>>(shape, size, m, dataset_in_ptr, dataset_in_indices_ptr);
}

} // operators
} // pp3d