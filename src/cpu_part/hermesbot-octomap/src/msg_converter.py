import numpy as np
from sensor_msgs.msg import PointCloud2

def _xyz_dtype_from_cloud(cloud: PointCloud2):
    """Structured dtype view over cloud.data using actual x,y,z offsets."""
    fields = {f.name: f for f in cloud.fields}
    offs = [fields["x"].offset, fields["y"].offset, fields["z"].offset]
    return np.dtype({"names": ["x", "y", "z"],
                     "formats": ["<f4", "<f4", "<f4"],
                     "offsets": offs,
                     "itemsize": cloud.point_step})

def _extract_xyz_views(cloud: PointCloud2):
    """Zero-copy structured view to x,y,z (1D arrays)."""
    dt = _xyz_dtype_from_cloud(cloud)
    npts = (cloud.width * cloud.height)
    rec = np.frombuffer(cloud.data, dtype=dt, count=npts)
    return rec["x"], rec["y"], rec["z"]

def ros_pointcloud_to_numpy_points(pointcloud: PointCloud2) -> np.ndarray:
    x,y,z = _extract_xyz_views(pointcloud)
    points = np.array([x,y,z]).T.astype("float32")
    return points