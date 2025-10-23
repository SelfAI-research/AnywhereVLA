#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include "common_lib.h"
#include <pcl_conversions/pcl_conversions.h>
#include <random>
#include <algorithm>

using namespace std;


/*** Velodyne ***/
namespace velodyne_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D;
  float intensity;
  float time;
  std::uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
} // namespace velodyne_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
 (float, x, x)(float, y, y)(float, z, z)
 (float, intensity, intensity)
 (float, time, time)
 (std::uint16_t, ring, ring))
/****************/

class Preprocess
{
public:
  //   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();

  void process(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg, PointCloudXYZI::Ptr &pcl_out);

  PointCloudXYZI pl_surf;
  int lidar_type;
  double blind, blind_sqr, downsample_rate;

private:
  void velodyne_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);
  void l515_handler(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

  // simple random keep
  std::mt19937 rng_;
  std::uniform_real_distribution<float> uni_{0.0f, 1.0f};

  inline bool keep_random(double filter_probapility) {
    return uni_(rng_) * 100.0f > filter_probapility;
  }
  inline bool pass_range(const PointType& p, double blind_sqr) const {
    double r2 = (double)p.x*p.x + (double)p.y*p.y + (double)p.z*p.z;
    return r2 > blind_sqr;
  }

};
typedef std::shared_ptr<Preprocess> PreprocessPtr;

#endif // PREPROCESS_H_