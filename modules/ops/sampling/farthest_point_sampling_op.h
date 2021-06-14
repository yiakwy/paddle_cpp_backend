//
// Created by yiak on 2021/6/11.
// The file is borrowed and adapted from
//   https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/3d_vision/PointNet++/ext_op/src/farthest_point_sampling_op.cc
//
// to implement new implementation of farthestSampling operator both in CPU and GPU device
//
// Credits to orignal author
//
#pragma once

#ifndef PP_OPS_FARTHEST_POINT_SAMPLING_OP_H
#define PP_OPS_FARTHEST_POINT_SAMPLING_OP_H

#include <memory>

#include "flags/ops_sampling_flags.h"

#include "paddle/fluid/framework/op_registry.h"

namespace pp3d {
namespace operators {

using namespace paddle;
using ExecutionContext = framework::ExecutionContext;
using InferShapeContext = framework::InferShapeContext;
using Tensor = framework::Tensor;

class FarthestPointSamplingOpMaker : public framework::OpProtoAndCheckerMaker {
public:
    void Make() override {
        AddInput("X",
                "(Tensor)input point cloud dataset with shape (B, N, 3/4)"
                 "B is batch size, N is points's nums, 3/4 is (x,y,z)/(x,y,z,intensity) coordinate");
        AddOutput("Output",
                  "(Tensor)return sampled points with shape (B, M)"
                  "B is batch size, M is sampled points's nums");
        AddAttr<int>("sampled_point_num", "sampling points's num, denoted as M internally")
                .SetDefault(0)
                .EqualGreaterThan(0);
        AddComment(
                R"Doc(
            Sampling point based on
            its max eucliden distance with other points.)Doc");
    }
};

class FarthestPointSamplingOp : public framework::OperatorWithKernel {
public:
    // use base constructor
    using framework::OperatorWithKernel::OperatorWithKernel;

protected:
    void InferShape(InferShapeContext *ctx) const override  {
        PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X should not be null");
        auto x_dims = ctx->GetInputDim("X");
        PADDLE_ENFORCE(x_dims.size() == 3 || x_dims.size() == 4,
                       "Input(X) of FathestPointSamplingOp should be 3/4-D Tensor");
        const int m = ctx->Attrs().Get<int>("sampled_point_num");
        ctx->SetOutputDim("Output", {x_dims[0], m});
    }

    framework::OpKernelType GetExpectedKernelType( const ExecutionContext& ctx) const override {
        auto input_data_type = ctx.Input<Tensor>("X")->type();
        return framework::OpKernelType(input_data_type, ctx.GetPlace());
    }
};

/*
 * Concret implementation headers for devices
 *   implementation for cpu device resides in ${OpName}.cc
 *   implementation for gpu device resides in impl/${OpName}.${DeviceSuffix}
 */
template<typename DeviceContext>
void farthestPointSampling(int* shape, int size, int m, double* dataset_in_ptr, int* dataset_in_indices_ptr);

template <typename DeviceContext, typename T>
class FarthestPointSamplingKernel : public framework::OpKernel<T> {
public:
    void Compute(const ExecutionContext& ctx) const override;
};

} // operators
} // pp3d



#endif //PP_OPS_FARTHEST_POINT_SAMPLING_OP_H
#include "impl/farthest_point_sampling_op_impl.hpp"