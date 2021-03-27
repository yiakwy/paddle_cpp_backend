project(application CXX C)

add_executable(least_square_trainer
        least_square_trainer.cc
        )
set_target_properties(least_square_trainer PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(least_square_trainer PUBLIC ${PADDLE_LIB}/third_party/dlpack/src/extern_dlpack/include/)
# add paddle libraries support
target_link_libraries(least_square_trainer
        ${MACOS_LD_FLAGS}
        ${ARCHIVE_START}
        ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_inference.so
        ${ARCHIVE_END}
        ${MATH_LIB}
        ${MKLDNN_LIB}
        gflags glog protobuf z xxhash
        ${EXTERNAL_LIB})