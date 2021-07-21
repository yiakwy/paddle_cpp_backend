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
#include <stdio.h>

#include <glog/logging.h>

// try to call thrust reduction in device side

#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuComplex.h>
#include <thrust/complex.h>

#include "base/helper_cuda.h"
#include "base/timmer.h"

#include "farthest_point_sampling_op.h"
#include "flags/ops_sampling_flags.h"

//#if defined(__CUDA_ARCH__)
// #undef __CUDA_ARCH__
//#endif

namespace pp3d {
namespace operators {

using namespace base::timmer;


struct MaxIdxOp {
    float* values = nullptr;

    __host__ __device__
    int operator() THRUST_PREVENT_MACRO_SUBSTITUTION(const int &lhs_idx, const int &rhs_idx) {
        return values[rhs_idx] < values[lhs_idx] ? rhs_idx : lhs_idx;
    }
};


// to avoid C pointer aliasing, use __restrict__ keyword to make sure codes execute in parallel is safe, also see CPU kernel
// in "farthest_point_sampling_op.cc
__global__ void farthestPointSamplingKernel(int* shape, int size, int m, double* __restrict__ dataset_in_ptr, int* __restrict__ dataset_in_indices_ptr) {
    if (size < 3) {
# if __CUDA_ARCH__ >= 200
       printf("[INFO] [%d] [farthestPointSamplingKernel] size of shape should be larger than 3, but found is %", __LINE__, size);
#endif
    }

    int batch_size, height, width;
    batch_size = shape[0];
    height = shape[1];
    width = shape[2];

    if (width != 3 && width != 4) {
        return;
    }

    // to make best of usage of the device, we prefer allocating shared memory in runtime
    // the shared buf is shared by all batches, hence we need to process batches one after another
    extern __shared__ float shared_buf[];

    // make sure blockSize.x >= height * width
    int num_floats = height * width;
    if (blockDim.x < num_floats || blockDim.x % num_floats != 0) {
# if __CUDA_ARCH__ >= 200
        printf("[WARNING] Block size is not multiples of height * width!");
#endif
    }

    unsigned int tid = threadIdx.x;
    unsigned int pt_idx = tid + blockIdx.x * blockDim.x; // thread id in a block


    for (int i=0; i < batch_size; i += gridDim.x) {
        printf("[INFO] [%d] [farthestPointSamplingKernel]", __LINE__, size);

        double* buf_points_in_ptr = dataset_in_ptr + i * height * width;
        int* temp_points_in_indices_ptr = dataset_in_indices_ptr + i * m;

        // shared memory buffer
        float* dists_ptr = (float*)&shared_buf[0];
        int* dists_indices_ptr = (int*)&shared_buf[height];

        int first_sampled_idx = 0;
        int last_point_idx = -1;

        if (threadIdx.x == 0) {
            last_point_idx = first_sampled_idx;
            temp_points_in_indices_ptr[0] = last_point_idx;
        }

        dists_ptr[pt_idx] = 1e9;
        __syncthreads();

        for (size_t j=0; j < m; j++) {
            double x, y, z, dx, dy, dz;

            x = buf_points_in_ptr[pt_idx * width + 0];
            y = buf_points_in_ptr[pt_idx * width + 1];
            z = buf_points_in_ptr[pt_idx * width + 2];

            double x_j, y_j, z_j;
            dx = x - x_j;
            dy = y - y_j;
            dz = z - z_j;

            double dist_j = dx*dx + dy*dy + dz*dz;
            if (dists_ptr[pt_idx] > dist_j) {
                dists_ptr[pt_idx] = dist_j;
                dists_indices_ptr[pt_idx] = pt_idx;
            }
            // wait for updating distances for all points
            __syncthreads();

            MaxIdxOp op;
            op.values = dists_ptr;

            // update sampled points indices, starting from thrust 1.8.0, user can call host side interface inside kernel
            int max_idx = thrust::reduce(thrust::seq, dists_indices_ptr, dists_indices_ptr+height, 0, op);

            last_point_idx = max_idx;

            // write the sampled point idx back to global memory
            if (tid == 0) { temp_points_in_indices_ptr[j] = last_point_idx; }

        }


    }


}

// if the source code compiled with CUDA
#ifdef PADDLE_WITH_CUDA
// GPU Kernel
template<>
void farthestPointSampling<platform::CUDADeviceContext>(int* shape, int size, int m, double* dataset_in_ptr, int* dataset_in_indices_ptr) {
    // the values will be changed according to tests report from `luanch_farthestSamplingKernel`
    farthestPointSamplingKernel<<<32, 512>>>(shape, size, m, dataset_in_ptr, dataset_in_indices_ptr);
}
#endif

// when CUDADeviceContext is not availabe, i.e. your paddle distribution is not compiled with CUDA device
void launch_farthestPointSamplingKernel(int* shape, int size, int m, double* dataset_in_cpu_ptr, int* dataset_in_cpu_indices_ptr,
        int minGridSize=-1, int blockSize=-1) {
    if (size < 3) {
        LOG(FATAL) << "[farthestPointSampling] size of shape should be larger than 3, but found is " << size;
    }

    int batch_size, height, width;
    batch_size = shape[0];
    height = shape[1];
    width = shape[2];

    if (width != 3 && width != 4) {
        return;
    }

    double* dataset_in_ptr;
    int* dataset_in_indices_ptr;

    // Allocating memory on GPU device memory proxier using unified API
    cudaMallocManaged(&dataset_in_ptr, batch_size * height * width * sizeof(float));
    base::checkCudaErr("Allocating GPU memory failed!");
    cudaMallocManaged(&dataset_in_indices_ptr, batch_size * m);
    base::checkCudaErr("Allocating GPU memory failed!");

    memcpy(dataset_in_ptr, dataset_in_cpu_ptr, batch_size * height * width * sizeof(float));
    base::checkCudaErr("Initialization failed!");

    memset(dataset_in_indices_ptr, 0, batch_size * m);

    // define gpu threads grid
    if (blockSize == -1) {
        blockSize = 512;
        LOG(INFO) << "set `blockSize` to default value : " << blockSize;
    }
    if (FLAGS_threads_grid_search) {
        LOG(INFO) << "perform threads occupancy testing ...";
        TicToc timer;
        timer.tic();
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, farthestPointSamplingKernel, 0, 0);
        LOG(INFO) << "Threads occupancy test is done, elapsed: " << timer.toc().c_str();
        LOG(INFO) << "set blockSize to : " << blockSize;
        LOG(INFO) << "set minGridSize to : " << minGridSize;
    }

    if (blockSize < height) {
        LOG(WARNING) << "block size (%d) is smaller than points size %d";
    }
    size_t blocksPerGrid = std::floor(1. * height + blockSize - 1) / blockSize;
    size_t sharedMemPerBlock = 2 * height * sizeof(float);

    farthestPointSamplingKernel<<<blocksPerGrid, blockSize, sharedMemPerBlock>>>(shape, size, m, dataset_in_ptr, dataset_in_indices_ptr);

    base::checkCudaErr(cudaDeviceSynchronize(), "Synchronization");
    base::checkCudaErr(cudaGetLastError(), "farthest point sampling GPU kernel");

    memcpy(dataset_in_cpu_indices_ptr, dataset_in_indices_ptr, batch_size * m);

    cudaFree(dataset_in_ptr);
    cudaFree(dataset_in_indices_ptr);
}


} // operators
} // pp3d