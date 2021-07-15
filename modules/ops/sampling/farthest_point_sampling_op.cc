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

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <functional>

#include "farthest_point_sampling_op.h"

namespace pp3d {
namespace operators {



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