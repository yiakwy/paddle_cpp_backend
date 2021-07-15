# add tests
include(CTest)
include("cmake/External/GTest.cmake")

# add logs
# include("cmake/External/GLog.cmake")

# add flags
# include("cmake/External/GFlags.cmake")

# Mapping externals used for this project
echo ("Home directory : $ENV{HOME}")
if (DEFINED ENV{MAPPING_EXTERNAL_DIR})
    set (EXTERNAL_DIR "$ENV{MAPPING_EXTERNAL_DIR}")
else ()
    set (EXTERNAL_DIR "$ENV{HOME}/mapping_external")
    message (WARNING "System variable EXTERNAL_DIR is nonexist, using ${EXTERNAL_DIR} instead.")
endif ()

# used to install compiled third party libraries by other groups
# in this project, this should be empty
if (NOT IS_DIRECTORY ${EXTERNAL_DIR})
    message(FATAL_ERROR "EXTERNAL_DIR ${EXTERNAL_DIR} is not a directory")
endif ()

if (CMAKE_HOST_WIN32)
    set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/win64)
elseif (CMAKE_HOST_APPLE)
    set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/darwin)
elseif(CMAKE_HOST_UNIX)
    if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
        set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/linux)
    elseif (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
        set(EXTERNAL_LIBS_DIR ${EXTERNAL_DIR}/arm64-linux)
    endif()
endif()
echo("EXTERNAL_LIBS_DIR : ${EXTERNAL_LIBS_DIR}")

list (APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# OpenCV
# To avoid conflicts introduced by opencv4, I installed the package frm source
# see vendors/github.com/opencv and installer scripts/thirdparties/linux/deb/apt/install_opencv.sh
set (OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV 4.0 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV 3.0 QUIET)
  if (NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV >= 3.0 not found")
  endif()
endif()
if (OpenCV_FOUND)
  echo("Find Opencv (ver.${OPENCV_VERSION}) (include: ${OpenCV_INCLUDE_DIRS}, library: ${OpenCV_LIBRARIES})")
  include_directories(
    ${OpenCV_INCLUDE_DIRS}
    )
endif()

# Boost
find_package(Boost COMPONENTS system filesystem REQUIRED)
echo ("Boost Version : ${Boost_VERSION_STRING}")
echo ("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
echo ("Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
echo ("Boost_LIBRARIES :   ${Boost_LIBRARIES}")

# Pangolin
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# ceres
# set(Ceres_DIR ${MAPPING_EXTERNAL_DIR}/ceres/lib/cmake/Ceres)
# find_package(Ceres QUIET REQUIRED)
# include_directories(${CERES_INCLUDE_DIRS})

# Since we switch to gcc > 8.2, c++14, we decided to move to PCL-1.11 to replace old version of PCL-1.8
# Though we have great efforts in optimizing codes base pertaining to PCL, PCL-1.11 still have a better native support to
# CUDA and threaded algorithms in fundamental components.
# PCL-1.11
set(PCL_DIR ${MAPPING_EXTERNAL_DIR}/pcl/share/pcl-1.11)
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    find_package(PCL REQUIRED
            COMPONENTS common kdtree octree search surface io ml sample_consensus filters geometry 2d features segmentation visualization registration)
else ()
    find_package(PCL QUIET REQUIRED
            COMPONENTS common kdtree octree search surface io ml sample_consensus filters geometry 2d features segmentation registration)
endif ()
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
list(APPEND CMAKE_INSTALL_RPATH "${PCL_LIBRARY_DIRS}")

# TBB
LIST(APPEND CMAKE_MODULE_PATH ${MAPPING_EXTERNAL_DIR}/tbb/lib/cmake/TBB)
# set(TBB_ROOT_DIR ${EXTERNAL_LIBS_DIR}/tbb)
# set(TBB_INCLUDE_DIR ${EXTERNAL_LIBS_DIR}/tbb/include)
# set(TBB_LIBRARY ${EXTERNAL_LIBS_DIR}/tbb/lib)
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
message(STATUS "TBB_INCLUDE_DIR: ${TBB_INCLUDE_DIR}")
message(STATUS "TBB_INCLUDE_DIRS: ${TBB_INCLUDE_DIRS}")
message(STATUS "TBB_LIBRARY: ${TBB_LIBRARY}")
message(STATUS "TBB_LIBRARIES: ${TBB_LIBRARIES}")
message(STATUS "TBB_VERSION: ${TBB_VERSION}")
unset(TBB_ROOT_DIR)
unset(TBB_LIBRARY)

# find_package(ZLIB)
set (ZLIB_LIBRARIES "/usr/local/lib/libz.so.1.2.9")
echo ("ZLIB_LIBRARIES: ${ZLIB_LIBRARIES}")
link_libraries(${ZLIB_LIBRARIES})

# see tutorials from https://cmake.org/cmake/help/latest/module/FindCUDA.html
# CUDA/CUDNN
find_package(CUDA REQUIRED)
find_cuda_helper_libs(culibos)
list(APPEND CUDA_LIBRARIES
        ${CUDA_cublas_LIBRARY}
        ${CUDA_curand_LIBRARY}
        ${CUDA_culibos_LIBRARY}
)
echo ("CUDA_LIBRARIES : ${CUDA_LIBRARIES}")
if (CUDA_FOUND)
    echo ("Found cuda libraries : ${CUDA_LIBRARIES}")
    include_directories(${CUDA_INCLUDE_DIRS})
    message(STATUS "CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA_TOOLKIT_TARGET_DIR: ${CUDA_TOOLKIT_TARGET_DIR}")
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

    set(NVCC_EXTRA_FLAGS "-std=c++14 -Xcompiler -fPIC -use_fast_math --disable-warnings")
    if(X86 OR X86_64)
        set(NVCC_EXTRA_FLAGS "${NVCC_EXTRA_FLAGS} -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75")
    else()
        set(NVCC_EXTRA_FLAGS "${NVCC_EXTRA_FLAGS} -gencode arch=compute_72,code=sm_72")
    endif()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_EXTRA_FLAGS}")
    message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
else()
    echo ("CUDA is not found!")
endif()
