//
// Created by yiak on 2021/6/12.
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <string>

// GNU software
#include <glob.h>
// Linux libraries
#include <sys/stat.h>

#define _POXIS_SOURCE
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <stdio.h>

#include <boost/shared_ptr.hpp>
// the latest build pcl compatible with boost-1.41.0 (published in 2009) is PCL 1.7.0
// however, this build has many problems with existing Eigen3
/*
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_traits.h>
 */

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"


// @todo TODO add str formatter, remove hard coded file path
DEFINE_string(test_pcd_file, "/home/yiak/WorkSpace/Github/paddle_cpp_backend/modules/ops/sampling/tmp_test_small_dataset/kittidata/2011_09_26/2011_09_26_drive_0018_sync/velodyne_points/data/0000000000.bin", "test cpu/gpu kernel main computing functions");

#include "farthest_point_sampling_op.h"

int read_kitti_velodyne_points(const std::string& filename, double** points, size_t* num_returns) {
    FILE *fp = fopen(filename.c_str(), "r");

    if (fp == nullptr) {
        return -1;
    }

    int fd = fp->_fileno;

    struct stat attrib;
    // read info from linux innode metadata
    if (stat(filename.c_str(), &attrib) < 0) {
        LOG(ERROR)
                << "Could not find the meta data from the innode of the file[Linux] :" << filename.c_str();
        close(fd);
        return -1;
    }
    size_t size = (size_t)attrib.st_size;
    *num_returns = size / (sizeof(float)*4);

    CHECK(points != nullptr);
    if (*points == nullptr) {
        *points = (double*)malloc(*num_returns * sizeof(double)*4);
        memset(*points, 0, *num_returns);
    }

    int i=0;
    for (int i=0; i < *num_returns; i++) {
        float x, y, z, intensity;

        if (fread(&x, sizeof(float), 1, fp) == 0) break;
        if (fread(&y, sizeof(float), 1, fp) == 0) break;
        if (fread(&z, sizeof(float), 1, fp) == 0) break;
        if (fread(&intensity, sizeof(float), 1, fp) == 0) break;

        (*points)[4*i + 0] = x;
        (*points)[4*i + 1] = y;
        (*points)[4*i + 2] = z;
        (*points)[4*i + 3] = intensity*255;
    }

    return 0;

}

void Parse_args(int argc, char** argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::InitGoogleLogging(argv[0]);
}

using namespace pp3d::operators;

int main(int argc, char** argv) {
    Parse_args(argc, (char**)argv);

    // paddle::framework::InitDevices();

    double* points = nullptr;
    int* points_indices;
    size_t num_returns;
    if(read_kitti_velodyne_points(FLAGS_test_pcd_file, &points, &num_returns) == -1) {
        LOG(FATAL) << "Cannot read " << FLAGS_test_pcd_file;
    }

    // Shape : batch, number of velodyne returns, point dimension
    int shape[3] = {1, (int)num_returns, 4};
    int sample_size = 1024; // should be equalt to GPU batch size

    points_indices = (int*)malloc(sample_size * sizeof(int));
    memset(points_indices, 0, sample_size * sizeof(int));

    // test the operator
    int n_threads = std::thread::hardware_concurrency() - 1;
    tbb::task_scheduler_init sched(n_threads);

    pp3d::operators::farthestPointSampling<platform::CPUDeviceContext>(&shape[0], 3, sample_size, points, points_indices);

    // dump points back to PCD format for visualization
    // using PCLPoint = pcl::PointXYZI;
    auto dump_points = [=] (double* points, int size) {
        /*
        pcl::PointCloud<PCLPoint>::Ptr cloud(new pcl::PointCloud<PCLPoint>);
        PCLPoint tmp_point;
        for (size_t i=0; i < size; i++)
        {
            tmp_point.x = points[i*4+0];
            tmp_point.y = points[i*4+1];
            tmp_point.z = points[i*4+2];
            tmp_point.intensity = points[i*4+3];
            cloud->push_back(tmp_point);
        }

        // @todo TODO move generated file to log folder

        pcl::PCDWriter writer;
        if (writer.write("filtered_points_frame_0.pcd", *cloud, false) < 0) {
            LOG(FATAL) << "cloud for the test file is invalid. Exit";
        }
         */

    };
    dump_points(points, num_returns);
    LOG(INFO) << "test points is dumped to disk";

    LOG(INFO) << "filtered points is dumped to disk";

    // deallocate points
    free(points);
    points = nullptr;
    free(points_indices);
    points_indices = nullptr;
    return 0;
}