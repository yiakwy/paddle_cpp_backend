//
// Created by yiakwy on 2021/3/16.
//
#pragma once

#ifndef PADDLE_TWISTED_CONV2D_H
#define PADDLE_TWISTED_CONV2D_H

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/layout_utils.h"

#include "third_party/install/mklml/include/mkl_cblas.h"

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

// paddle extension file
#include "paddle/extension.h"

#include "flags/ops_sampling_flags.h"

namespace pp3d {
namespace operators {

using namespace paddle;

using Tensor = framework::Tensor;
constexpr int kConvMKLDNNFP32 = 1;
constexpr int kConvMKLDNNINT8 = 2;
constexpr int MaxKeyLength = 256;

class TwistedConv2d {};

} // operators
} // pp3d
#endif  // PADDLE_TWISTED_CONV2D_H
