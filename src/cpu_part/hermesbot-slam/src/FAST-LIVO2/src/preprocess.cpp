/* 
This file is part of FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry.

Developer: Chunran Zheng <zhengcr@connect.hku.hk>

For commercial use, please contact me at <zhengcr@connect.hku.hk> or
Prof. Fu Zhang at <fuzhang@hku.hk>.

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "preprocess.h"

Preprocess::Preprocess()
{
  lidar_type = VELO16;
  downsample_rate = 0.0;
  blind = 0;
  blind_sqr = 0;

  rng_ = std::mt19937{std::random_device{}()};
}

Preprocess::~Preprocess() {}

void Preprocess::process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  switch (lidar_type) {
  case VELO16:
    velodyne_handler(msg);
    break;
  case L515:
    l515_handler(msg);
    break;
  default:
    printf("Error LiDAR Type: %d \n", lidar_type);
    break;
  }
  *pcl_out = pl_surf;
}

void Preprocess::l515_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
  pl_surf.clear();
  pcl::PointCloud<pcl::PointXYZRGB> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  pl_surf.reserve(plsize);

  for (int i = 0; i < plsize; ++i)
  {
    if (!keep_random(downsample_rate)) continue;

    const auto& src = pl_orig.points[i];
    PointType p;
    p.x = src.x; p.y = src.y; p.z = src.z;
    p.normal_x = src.r; p.normal_y = src.g; p.normal_z = src.b;
    p.curvature = 0.0;

    if (!pass_range(p, blind_sqr)) continue;
    pl_surf.points.push_back(p);
  }
}

void Preprocess::velodyne_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg)
{
  pl_surf.clear();
  pcl::PointCloud<velodyne_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.points.size();
  pl_surf.reserve(plsize);

  for (int i = 0; i < plsize; ++i)
  {
    if (!keep_random(downsample_rate)) continue;

    const auto& src = pl_orig.points[i];
    PointType p;
    p.x = src.x; p.y = src.y; p.z = src.z;
    p.normal_x = p.normal_y = p.normal_z = 0.0f;
    p.intensity = src.intensity;
    p.curvature = src.time;

    if (!pass_range(p, blind_sqr)) continue;
    pl_surf.points.push_back(p);
  }
}