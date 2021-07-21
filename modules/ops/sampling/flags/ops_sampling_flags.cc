//
// Created by yiak on 2021/6/13.
//

#include "ops_sampling_flags.h"

/*
 * Manually set call_stack_level before use C++ code base
 */
// DEFINE_int32(call_stack_level, 2, "print stacktrace");
using namespace fLI;

namespace pp3d {
namespace operators {

DEFINE_bool(debug_gpu_kernel, true, "debug gpu kernel");
DEFINE_bool(threads_grid_search, true, "debug gpu kernel");

} // operators
} // pp3d
