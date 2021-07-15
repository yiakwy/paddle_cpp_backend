ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD="${ROOT}/cmake-build-debug"

GLOG_logtostderr=1 ${BUILD}/bin/farthest_point_sampling_op_test
