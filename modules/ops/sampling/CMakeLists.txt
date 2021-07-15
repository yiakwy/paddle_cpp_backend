project(pp_ops CXX C)

file(GLOB_RECURSE FILES *.cpp *.cc *.c *.hpp *.h)
message(STATUS "pp_ops SRCS: ${FILES}")

list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/farthest_point_sampling_op_test.cc")
# removed files for building kernels with paddle-gpu-1.8.5
list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/twisted_conv2d.h")
list(REMOVE_ITEM FILES "${CMAKE_CURRENT_SOURCE_DIR}/twisted_conv2d.cc")

add_definitions(-DPADDLE_WITH_MKLDNN)
message(STATUS "MATH_LIB : ${MATH_LIB}")
message(STATUS "MKLDNN_LIB : ${MKLDNN_LIB}")

link_directories("${PADDLE_LIB_DIR}")

add_library(${PROJECT_NAME} SHARED
        ${FILES}
        "${PADDLE_LIB}/third_party/install/mklml/include/mkl_cblas.h"
        )
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX PRIVATE ${CMAKE_CXX_FLAGS})

# add cuda support
cuda_include_directories(${CMAKE_CURRENT_SOURCE_DIR})
cuda_add_library(sampling_cu_kernel
        "${CMAKE_CURRENT_SOURCE_DIR}/farthest_point_sampling_op.cu"
        )

# add paddle libraries support

include_directories("${PADDLE_LIB}/third_party/install/mklml/include")

# target_include_directories(${PROJECT_NAME} PUBLIC ${THIRD_INCR})
target_include_directories(${PROJECT_NAME} PUBLIC ${PADDLE_LIB}/third_party/dlpack/src/extern_dlpack/include/)
target_include_directories(${PROJECT_NAME} PRIVATE
        "${PADDLE_INCR}"
        "${PADDLE_INCR}/third_party"
        )

target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${MACOS_LD_FLAGS}
        ${ARCHIVE_START}
        # use this static library > Paddle 2.0
        # ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_inference.so
        ${ARCHIVE_END}
        ${MATH_LIB}
        ${MKLDNN_LIB}
        # use this static library <= Paddle 2.0
        paddle_framework
        gflags glog protobuf xxhash iomp5
        ${CUDA_LIBRARIES}
        ${TBB_LIBRARIES}
        ${EXTERNAL_LIB}
        ${ZLIB_LIBRARIES}
        )

### farthest point sampling local test ###
add_executable(farthest_point_sampling_op_test
        "farthest_point_sampling_op_test.cc")

## add links
set_target_properties(farthest_point_sampling_op_test PROPERTIES LINKER_LANGUAGE CXX)
#[[
target_include_directories(farthest_point_sampling_op_test
        PRIVATE
        ${Boost_INCLUDE_DIRS})
target_link_directories(farthest_point_sampling_op_test
        PUBLIC
        "${Boost_LIBRARY_DIRS}")
]]
echo ("PADDLE_LIB_DIR : ${PADDLE_LIB_DIR}")
target_link_directories(farthest_point_sampling_op_test
        PUBLIC
        ${PADDLE_LIB_DIR})
echo ("PCL_LIBRARIES : ${PCL_LIBRARIES}")
target_link_libraries(farthest_point_sampling_op_test
        ${PROJECT_NAME}
        ${PCL_LIBRARIES}
        /usr/local/lib/libz.so.1.2.9
        )