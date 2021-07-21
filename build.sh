#!/bin/bash

set -x

PADDLE_ROOT=/home/yiak/WorkSpace/Github/Paddle #-Released-2.0
TURN_ON_MKL=ON # use MKL or Openblas

# build paddle_cpp_backend
BUILD=cmake-build-debug 
# BUILD=build
paddle_install_dir=${PADDLE_ROOT}/build

mkdir -p $BUILD
cd $BUILD
cmake .. -DPADDLE_LIB=$paddle_install_dir \
         -DWITH_MKLDNN=$TURN_ON_MKL \
         -DWITH_MKL=$TURN_ON_MKL \
         -DCMAKE_BUILD_TYPE=Debug
# make

cd ..
