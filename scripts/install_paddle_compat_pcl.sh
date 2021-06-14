#!/usr/bin/env bash
set -ex

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# import libs
source ${ROOT}/scripts/utils.sh

VERSION="1.7.1"
PKG_NAME=pcl
PKG="pcl-${VERSION}"
PKG_SRC=${PKG}.tar.gz

DOWNLOAD_LINK="https://github.com/PointCloudLibrary/pcl/archive/${PKG_SRC}"

WORKHORSE="gpu"
# if [ -z "${WORKHORSE}" ]; then
#     WORKHORSE="cpu"
# fi

# Install system-provided pcl
# apt-get -y update && \
#   apt-get -y install \
#   libpcl-dev
# exit 0
# if ldconfig -p | grep -q libpcl_common ; then
#     info "Found existing PCL installation. Skipp re-installation."
#     exit 0
# fi

GPU_OPTIONS="-DCUDA_ARCH_BIN=\"${SUPPORTED_NVIDIA_SMS}\""
if [ "${WORKHORSE}" = "cpu" ]; then
    GPU_OPTIONS="-DWITH_CUDA=OFF"
fi

info "GPU Options for PCL:\"${GPU_OPTIONS}\""

TARGET_ARCH="$(uname -m)"
ARCH_OPTIONS=""
if [ "${TARGET_ARCH}" = "x86_64" ]; then
    ARCH_OPTIONS="-DPCL_ENABLE_SSE=ON"
else
    ARCH_OPTIONS="-DPCL_ENABLE_SSE=OFF"
fi

sudo apt-get install --no-install-recommends -y \
    libflann-dev \
    libglew-dev \
    libglfw3-dev \
    freeglut3-dev \
    libusb-1.0-0-dev \
    libdouble-conversion-dev \
    libopenni-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    liblz4-dev \
    libfreetype6-dev \
    libpcap-dev \
    libqhull-dev

# NOTE(storypku)
# libglfw3-dev depends on libglfw3,
# and libglew-dev have a dependency over libglew2.0

THREAD_NUM=$(expr `nproc` / 2 - 1)

download_if_not_cached "${PKG_NAME}" "${VERSION}" "${DOWNLOAD_LINK}" pcl-${PKG}

# Ref: https://src.fedoraproject.org/rpms/pcl.git
#  -DPCL_PKGCONFIG_SUFFIX:STRING="" \
#  -DCMAKE_SKIP_RPATH=ON \

pushd $VENDOR_DIR/${PKG_NAME}/
    if [[ ${VERSION} == "1.10.1" ]]; then
      patch -p1 < ${ROOT}/scripts/pcl-sse-fix-${VERSION}.patch
    fi
    mkdir -p build && cd build
    set +x
    cmake .. \
        "${GPU_OPTIONS}" \
        "${ARCH_OPTIONS}" \
        -DPCL_ENABLE_SSE=ON \
        -DWITH_DOCS=OFF \
        -DWITH_TUTORIALS=OFF \
        -DBUILD_global_tests=ON \
        -DOPENNI_INCLUDE_DIR:PATH=/usr/include/ni \
        -DCMAKE_INSTALL_PREFIX="${MAPPING_EXTERNAL_DIR}/${PKG_NAME}" \
        -DBOOST_ROOT=${HOME}/WorkSpace/Github/Paddle/build/third_party/boost/src/extern_boost \
        -DEIGEN_INCLUDE_DIR=${HOME}/WorkSpace/Github/Paddle/build/third_party/eigen3/src/extern_eigen3 \
        -DCMAKE_BUILD_TYPE=Release
    make -j${THREAD_NUM}
    make install
popd

ldconfig

ok "Successfully installed PCL ${VERSION}"
