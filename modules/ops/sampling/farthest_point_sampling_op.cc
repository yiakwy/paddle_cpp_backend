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
#include <limits>

#include <glog/logging.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <functional>

#include "farthest_point_sampling_op.h"

namespace pp3d {
namespace operators {

// CPU Kernel
template<>
void farthestPointSampling<platform::CPUDeviceContext>(int* shape, int size, int m, double* dataset_in_ptr, int* dataset_in_indices_ptr) {
    if (size < 3) {
        LOG(FATAL) << "[farthestPointSampling] size of shape should be larger than 3, but found is " << size;
    }

    int batch_size, height, width;
    batch_size = shape[0];
    height = shape[1];
    width = shape[2];

    // @todo TODO refactor algorithm to width dependent
    CHECK(width == 3 || width == 4);
    // setup Farthest Sampling Engine

    for (int i=0; i < batch_size; i++) {
        // Initialize shared buffer for all threads
        LOG(INFO) << "==Batch " << i << "===";

        double* buf_points_in_ptr = dataset_in_ptr + i * height * width;
        int* temp_points_in_indices_ptr = dataset_in_indices_ptr + i * m;

        Eigen::VectorXd dists;
        Eigen::VectorXi dists_indices;

        dists.resize(height);
        double *dists_ptr = dists.data();
        dists_indices.resize(height);
        int *dists_indices_ptr = dists_indices.data();

        // @todo TODO generate first point using sampling engine
        int first_sampled_idx = 0;
        temp_points_in_indices_ptr[0] = first_sampled_idx;

        auto initialize_dist_cpu_kernel = [=](const tbb::blocked_range<size_t> &range) {
            for (size_t j = range.begin(); j != range.end(); j++) {
                dists_ptr[j] = std::numeric_limits<size_t>::max() - 1;
            }
        };

        tbb::parallel_for(tbb::blocked_range<size_t>(0, height), initialize_dist_cpu_kernel);

        // sample (batch_size, sampled_size) points from (batch_size, height)
        // in a RTX GeForece 2080 Ti GPU equipped machine, sampled_size == cpu_batch_size == gpu_batch_size == 1024

        // repeat sampling process m times

        int last_point_idx = first_sampled_idx;
        for (size_t j = 0; j < m; j++) {
            // similiar to CUDA impl, each point is deemed to be assigned to a cpu thread
            LOG(INFO) << "Add sample point " << j << ", index: " << last_point_idx;

            // step 1: find point idx with farthest distance to the last added point
            double farthest_dist = -1;

            // Note since CPU has limited threads, the reduction algorithm is dramatically different from that of GPU version
            //
            // In GPU, we have shared buffer for in-place update. Half of the size operations are conducted
            // in parallel and it is repeated the log2 times until we got the final reduced value
            //
            // While, in CPU, we split the array into ranges and perform CPU reduction in parallel. Then we merge the
            // computed results by applying the same reduction again.
            MaxVal farthest_point = ParallelReduce(buf_points_in_ptr, height, [=] (const tbb::blocked_range<size_t>& r, MaxVal init_val) -> MaxVal {
                double x, y, z, dx, dy, dz;
                MaxVal tmp_farthest_point;
                double tmp_farthest_dist = farthest_dist;
                double tmp_idx = -1;

                for (size_t k = r.begin(); k != r.end(); k++) {
                    x = buf_points_in_ptr[k*width + 0];
                    y = buf_points_in_ptr[k*width + 1];
                    z = buf_points_in_ptr[k*width + 2];

                    // select the minimum distance to previous sampled points
                    double x_j, y_j, z_j;
                    x_j = buf_points_in_ptr[temp_points_in_indices_ptr[j] * width + 0];
                    y_j = buf_points_in_ptr[temp_points_in_indices_ptr[j] * width + 1];
                    z_j = buf_points_in_ptr[temp_points_in_indices_ptr[j] * width + 2];

                    dx = x - x_j;
                    dy = y - y_j;
                    dz = z - z_j;

                    double dist_j = dx * dx + dy * dy + dz * dz;
                    if (dists_ptr[k] > dist_j) {
                        dists_ptr[k] = dist_j;
                        dists_indices_ptr[k] = k;
                    }

                    if (dists_ptr[k] > tmp_farthest_dist) {
                        tmp_idx = k;
                        tmp_farthest_dist = dists_ptr[k];
                    }

                }

                tmp_farthest_point.idx = tmp_idx;
                tmp_farthest_point.val = tmp_farthest_dist;
                return tmp_farthest_point;

            }, 1);

            farthest_dist = farthest_point.val;
            last_point_idx = farthest_point.idx;

            // update sampled points indices
            temp_points_in_indices_ptr[j] = last_point_idx;
        }

    }

};

} // operators
} // pp3d

#ifndef REGISTER_FARTHEST_POINT_SAMPLING_OP
#define REGISTER_FARTHEST_POINT_SAMPLING_OP
REGISTER_OPERATOR(farthest_point_sampling,
                  pp3d::operators::FarthestPointSamplingOp,
                  pp3d::operators::FarthestPointSamplingOpMaker);
#endif

REGISTER_OP_CPU_KERNEL(farthest_point_sampling,
                       pp3d::operators::FarthestPointSamplingKernel<pp3d::operators::platform::CPUDeviceContext, float>,
                       pp3d::operators::FarthestPointSamplingKernel<pp3d::operators::platform::CPUDeviceContext, double>);