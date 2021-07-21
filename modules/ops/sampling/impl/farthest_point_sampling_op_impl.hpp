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

// @todo TODO replace to imperative form to reduce the copy of data
// Ideal(threads are always available) Space complexity : O(num_of_threads * log(N))
// Ideal(threads are always available) Times complexity :
struct MaxVal {
    size_t idx;
    float val;

    MaxVal() : idx(0), val(0) {}
    MaxVal(size_t a, float b) {
        idx = a;
        val = b;
    }
};

template <typename Fn>
MaxVal ParallelReduce(double* values, int size, Fn reducer, size_t grainsize=1)
{
    return tbb::parallel_reduce(
            tbb::blocked_range<size_t>(0, size),
            MaxVal(0, 0),
            reducer,
            //maximum<double>()
            [] (MaxVal x, MaxVal y) -> MaxVal {
                if (x.val > y.val) {
                    return x;
                } else {
                    return y;
                }
            }
    );
}


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

