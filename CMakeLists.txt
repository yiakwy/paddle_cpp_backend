cmake_minimum_required(VERSION 3.0)

project(cpp_train_demo CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

### Step 1: find sources ###

# in case that we want to refer to source codes in Paddle main line
if (NOT DEFINED PADDLE_SRC)
  set(PADDLE_SRC "/home/yiak/WorkSpace/Github/Paddle")
  message(WARNING "please set PADDLE_SRC with -DPADDLE_SRC, defaults to ${PADDLE_SRC}")
endif()

### Step 2: find headers and libraries ###

# for people who use derived products: "paddle inference", "paddle-lite", and those who event build paddle libraries from Paddle main line sources:
if(NOT DEFINED PADDLE_LIB)
  set(PADDLE_LIB "${PADDLE_SRC}/build")
  message(WARNING "please set PADDLE_LIB with -DPADDLE_LIB, defaults to ${PADDLE_LIB}")
endif()

# for whom install paddle from python side (Note this is not a reliable or easy way when you want to use source codes from Paddle main line)

execute_process(COMMAND python -c "import paddle.sysconfig as sysconfig; print(sysconfig.get_include())"
        OUTPUT_VARIABLE PADDLE_INCR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python -c "import paddle.sysconfig as sysconfig; print(sysconfig.get_lib())"
	OUTPUT_VARIABLE PADDLE_LIB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)

# find downloaded dependencies manually
execute_process(COMMAND find ${PADDLE_LIB}/third_party/threadpool
                             ${PADDLE_LIB}/third_party/dlpack
        -type f -name "*.h" -exec dirname {} \; OUTPUT_VARIABLE THIRD_INCR OUTPUT_STRIP_TRAILING_WHITESPACE)
# message(STATUS "THIRD_INCR : ${THIRD_INCR}")
string(REPLACE "\n" ";" THIRD_INCR ${THIRD_INCR})

### Step 3: add general headers ###

# add Paddle main line headers
# Make sure protoc in python is the version of c++ used (v3.1.0)
include_directories("${PADDLE_INCR}")
# help to discover generated headers:
#   1. protobuf files, e.g.: "paddle/fluid/platform/error_codes.pb.h"
#   2. installed third party headers which is not exported in installation directory, e.g.: "third_party/install/mklml/include/mkl_cblas.h"
include_directories("${PADDLE_LIB}")

# add general third party headers
include_directories("${THIRD_INCR}")

# add general third party headers search path manually
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")

include_directories("${PADDLE_LIB}/third_party/boost")

# include Eigen 3 headers not ended with typical ".h" or ".hpp" suffix
include_directories("${PADDLE_LIB}/third_party/eigen3/src/extern_eigen3/")

### Step 4: link to general dependencies ###

# link to general dependencies
link_directories("${PADDLE_LIB_DIR}")

# or link to general dependencies manually
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")

### Step 5: link optional libraries

# MKL library is used for CPU kernel by default
option(WITH_MKLDNN     "Compile PaddlePaddle with MKLDNN"                                   OFF)
option(WITH_MKL        "Compile PaddlePaddle with MKL support, default use openblas."       OFF)

if(WITH_MKLDNN)
  add_definitions(-DPADDLE_WITH_MKLDNN)
  include_directories("${PADDLE_LIB}/third_party/install/mkldnn/include")
  if(WIN32)
    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/mkldnn.lib)
  else(WIN32)
    set(MKLDNN_LIB ${PADDLE_LIB}/third_party/install/mkldnn/lib/libmkldnn.so)
  endif(WIN32)
endif(WITH_MKLDNN)

if(WITH_MKL)
  include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
  if(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib)
  else(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel.so)
  endif(WIN32)
else()
  if(APPLE)
    set(MATH_LIB cblas)
  elseif(WIN32)
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.lib)
  else()
    set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas.a)
  endif(APPLE)
endif()

### Step 6: Set CXX Flags for gcc compiler

if(APPLE)
  set(MACOS_LD_FLAGS "-undefined dynamic_lookup -Wl,-all_load -framework CoreFoundation -framework Security")
else(APPLE)
  set(ARCHIVE_START "-Wl,--whole-archive")
  set(ARCHIVE_END "-Wl,--no-whole-archive")
  set(EXTERNAL_LIB "-lrt -ldl -lpthread")
endif(APPLE)

# for debug purpose
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")

include(cmake/Utils.cmake)

set (SRCS_ROOT "${PROJECT_SOURCE_DIR}/modules")

# add searching path
include_directories(${SRCS_ROOT})

# add sub modules
add_subdirectory(${SRCS_ROOT})

#[[
### Step 7: specify local binaries and libraries

add_executable(demo_trainer demo_trainer.cc)
set_target_properties(demo_trainer PROPERTIES LINKER_LANGUAGE CXX)

# target_include_directories(demo_trainer PUBLIC ${PADDLE_LIB}/third_party/threadpool/src/extern_threadpool)
target_include_directories(demo_trainer PUBLIC ${PADDLE_LIB}/third_party/dlpack/src/extern_dlpack/include/)
target_link_libraries(demo_trainer
        ${MACOS_LD_FLAGS}
        ${ARCHIVE_START}
        ${PADDLE_LIB}/paddle/fluid/inference/libpaddle_inference.so
        ${ARCHIVE_END}
        ${MATH_LIB}
        ${MKLDNN_LIB}
        gflags glog protobuf z xxhash
        ${EXTERNAL_LIB})
]]