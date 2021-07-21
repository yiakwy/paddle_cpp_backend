//
// Created by yiak on 2021/6/13.
//
#pragma once

#ifndef PADDLE_CPP_BACKEND_OPS_SAMPLING_FLAGS_H
#define PADDLE_CPP_BACKEND_OPS_SAMPLING_FLAGS_H

#include <gflags/gflags.h>

DECLARE_int32(call_stack_level);

namespace pp3d {
namespace operators {

DECLARE_bool(debug_gpu_kernel);
DECLARE_bool(threads_grid_search);

} // operators
} // pp3d

#endif //PADDLE_CPP_BACKEND_OPS_SAMPLING_FLAGS_H
