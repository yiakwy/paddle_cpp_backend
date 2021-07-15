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
#pragma once

#ifndef PP_OPS_FARTHEST_POINT_SAMPLING_OP_IMPL_HPP
#define PP_OPS_FARTHEST_POINT_SAMPLING_OP_IMPL_HPP

#include <limits>

#include <glog/logging.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <functional>

#include "ops/sampling/farthest_point_sampling_op.h"

namespace pp3d {
namespace operators {

template<typename _Tp>
struct minimum : public std::binary_function<_Tp, _Tp, _Tp>
{
    _GLIBCXX14_CONSTEXPR
    _Tp
    operator()(const _Tp& __x, const _Tp& __y) const
    { return std::min(__x, __y); }
};

template<typename _Tp>
struct maximum : public std::binary_function<_Tp, _Tp, _Tp>
{
    _GLIBCXX14_CONSTEXPR
    _Tp
    operator()(const _Tp& __x, const _Tp& __y) const
    { return std::max(__x, __y); }
};

template <typename Fn>
double ParallelReduce(double* values, int size, Fn reducer, size_t grainsize=1)
{
    return tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, size, grainsize),
            0.f,
            reducer,
            maximum<double>()
    );
}

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
        int* temp_points_in_indices_ptr = dataset_in_indices_ptr + i * height * m;

        Eigen::VectorXd dists;
        Eigen::VectorXd dists_indices;

        dists.resize(height);
        double *dists_ptr = dists.data();
        dists_indices.resize(height);
        double *dists_indices_ptr = dists_indices.data();

        // @todo TODO generate first point using sampling engine
        int first_sampled_idx = 0;
        temp_points_in_indices_ptr[0] = first_sampled_idx;

        auto initialize_dist_cpu_kernel = [=](const tbb::blocked_range<size_t> &range) {
            for (size_t j = range.begin(); j != range.end(); j++) {
                dists_ptr[0] = std::numeric_limits<size_t>::max() - 1;
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
            std::atomic<int> idx{0};
            idx.store(-1);
            double val = ParallelReduce(buf_points_in_ptr, height, [=, &idx] (const tbb::blocked_range<size_t>& r, double pre_val) {
                double x, y, z, dx, dy, dz;
                double tmp_farthest_dist = farthest_dist;
                double tmp_idx = -1;
                for (size_t k = r.begin(); k != r.end(); k++) {
                    x = buf_points_in_ptr[k*width + 0];
                    y = buf_points_in_ptr[k*width + 1];
                    z = buf_points_in_ptr[k*width + 2];

                    // select the minimum distance to previous sampled points
                    double dist = std::numeric_limits<size_t>::max() - 1;
                    for (int sampled_idx=j; sampled_idx >=0; sampled_idx--) {
                        double x_i, y_i, z_i;
                        size_t point_idx = temp_points_in_indices_ptr[sampled_idx];
                        x_i = buf_points_in_ptr[point_idx*width + 0];
                        y_i = buf_points_in_ptr[point_idx*width + 1];
                        z_i = buf_points_in_ptr[point_idx*width + 2];

                        dx = x - x_i;
                        dy = y - y_i;
                        dz = z - z_i;

                        double dist_i = dx*dx + dy*dy + dz*dz;
                        if (dist_i < dist) {
                            dist = dist_i;
                        }
                    }

                    // select the maximum distance to this batch of points
                    if (dist > tmp_farthest_dist) {
                        tmp_idx = k;
                        tmp_farthest_dist = dist;
                    }
                }

                dists_ptr[r.begin()] = tmp_idx;
                dists_indices_ptr[r.begin()] = tmp_farthest_dist;

                if (tmp_farthest_dist > pre_val) {
                    idx.store(tmp_idx);
                    return tmp_farthest_dist;
                } else {
                    return pre_val;
                }

            }, 1);

            farthest_dist = val;
            last_point_idx = idx.load();

            // update sampled points indices
            temp_points_in_indices_ptr[j] = last_point_idx;
        }

    }

};

template <typename DeviceContext, typename T>
void FarthestPointSamplingKernel<DeviceContext, T>::Compute(const ExecutionContext& ctx) const {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "This kernel only runs on CPU device");

    auto *input = ctx.Input<Tensor>("X");
    auto *output = ctx.Output<Tensor>("Output");
    if (input->numel() == 0) return;

    auto *points_indices_ptr = output->mutable_data<int>(ctx.GetPlace());

    int shape[3] = {0};
    shape[0] = input->dims()[0];
    shape[1] = input->dims()[1];
    shape[2] = input->dims()[2];

    auto *points_ptr = input->data<T>();

    int sample_size = ctx.Attr<int>("sampled_point_num");

    farthestPointSampling<DeviceContext>(&shape[0], 3, sample_size, (double*)points_ptr, (int*)points_indices_ptr);

}


} // operators
} // pp3d
#endif //PP_OPS_FARTHEST_POINT_SAMPLING_OP_IMPL_HPP

