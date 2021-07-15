ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD=${ROOT}/build #${ROOT}/cmake-build-debug
mkdir -p $BUILD

cd $BUILD

# apply patch to Paddle src root
#  The Paddle uses an old boost distribution (1.41.0) to build ML core software.
#  However, despite of many reasons to do that. We do not want it in this build. 
#
#  There are plenty of reasons to support us. For example, If we want to build 
#  customize operators for 3d problems, it is convenient for us to include third party
#  libraries like PCL-1.8.1 to construct binaries either for simple visualization or
#  just reduce the code base. However, PCL is shipped with latest boost which is not 
#  compatible with the oldest one.
# 
#  The current paddle built is not compatible with boost-1.65.1
#
# patch -p1 < ${ROOT}/scripts/build_update_boost-1.41.0_to_1.65.1.patch

# addtional flags to add
#  -DON_INERENCE=ON
#  -DProtobuf_INCLUDE_DIR=${BUILD}/third_party/install/protobuf/include
PATH=/usr/local/gcc-8.2/bin:$PATH cmake .. -DPYTHON_EXECUTABLE:FILEPATH=$(which python) \
	 -DPYTHON_INCLUDE_DIR:PATH=$PYTHON_INCLUDE_DIR \
	 -DPYTHON_LIBRARY:FILEPATH=$PYTHON_LIBRARY \
	 -DWITH_GPU=ON \
	 -DWITH_TESTING=OFF \
   -DWITH_NCCL=ON \
   -DProtobuf_INCLUDE_DIR=${BUILD}/third_party/install/protobuf/include \
	 -DCMAKE_BUILD_TYPE=Release
