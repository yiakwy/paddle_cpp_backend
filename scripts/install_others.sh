#!/bin/bash

sudo apt install -y bzip2

# use cmake 3.16 for this compileration.
# note we will use gcc-8.2. since gcc is ABI compatible,
# we are opted use different compiler for different.
# note native cmake from ubuntu 18.04 does not support http
num_cores=`expr $(grep -c ^processor /proc/cpuinfo)`
if (( $num_cores > 32 )); then
  num_cores=32
fi

cmake_pkg_source="https://cmake.org/files/v3.16/cmake-3.16.0-Linux-x86_64.tar.gz"
if [ ! -f cmake-3.16.0-Linux-x86_64.tar.gz ]; then
  wget --no-check-certificate https://cmake.org/files/v3.16/cmake-3.16.0-Linux-x86_64.tar.gz
  tar -zxvf cmake-3.16.0-Linux-x86_64.tar.gz
  # mv cmake-3.16.0-Linux-x86_64 cmake && \
  # mkdir temp_cmake && cd temp_cmake && \
  # ../cmake/boostrap --system-curl && \
  # make -j$num_cores && sudo make install
fi
  
# used to modify ELF executables's linker and RPATH
sudo apt install patchelf

# make sure the command is issued under python3.6
pip install protobuf
pip install numpy


