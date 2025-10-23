# lidar_synchronizer/lidar_synchronizer.py
import os, time, yaml
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32


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


class PointCloudMerger(Node):
    def __init__(self, config_path: str):
        super().__init__('lidar_synchronizer')

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)['lidar_synchronizer']

        self.output_topic = cfg['output_topic']
        self.queue_size   = cfg['queue_size']
        self.frame_id     = cfg['frame_id']
        self.lidar_cfgs   = cfg['lidars']
        self.slop_s       = cfg['slop']
        self.qos_depth    = cfg['qos_depth']

        print("[LiDAR Synchronizer] Loaded configuration:")
        print(f"  Output topic : {self.output_topic}")
        print(f"  Frame ID     : {self.frame_id}")
        print(f"  Queue size   : {self.queue_size}")
        print(f"  Slop (s)     : {self.slop_s}")
        print(f"  QoS depth    : {self.qos_depth}")
        print(f"  Lidars       : {len(self.lidar_cfgs)}\n")

        # RELIABLE + deep queue
        sensor_qos = QoSProfile(
            depth=self.qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE
        )

        self.subscribers, self.transforms, self.pp_params = [], [], []
        for i, lidar in enumerate(self.lidar_cfgs):
            topic = lidar['topic']
            t = np.asarray(lidar['transform']['translation'], dtype=np.float32)
            R = np.asarray(lidar['transform']['rotation'],    dtype=np.float32).reshape(3, 3)
            self.transforms.append((R, t))
            pp = lidar['preprocesing']
            self.pp_params.append({
                'downsample_rate': float(pp.get('downsample_rate', 0.0)),
                'blind_range':     float(pp.get('blind_range',     0.0)),
                'max_range':       float(pp.get('max_range',   np.inf))
            })
            self.subscribers.append(Subscriber(self, PointCloud2, topic, qos_profile=sensor_qos))

            print(f"  Lidar[{i}] topic: {topic}")
            print(f"    ↳ Downsample : {self.pp_params[-1]['downsample_rate']}%")
            print(f"    ↳ Blind range: {self.pp_params[-1]['blind_range']} m")
            print(f"    ↳ Max range  : {self.pp_params[-1]['max_range']} m")

        self.ts = ApproximateTimeSynchronizer(
            self.subscribers, self.queue_size, self.slop_s, allow_headerless=False
        )
        self.ts.registerCallback(self.callback)

        self.publisher = self.create_publisher(PointCloud2, self.output_topic, 1)

        self._msg_count = 0
        self._start_time = time.time()
        self._log_every = cfg.get('log_every_n', 10)
        self.times = []

        self._pool = ThreadPoolExecutor(max_workers=max(1, len(self.subscribers)))

        print("\n[LiDAR Synchronizer] Waiting for lidar mesages...\n")

    @staticmethod
    def _preprocess_xyz(x, y, z, params):
        # finite mask
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

        # distance mask (compute r^2 to avoid sqrt)
        r2 = x*x + y*y + z*z
        in_range = (r2 >= params['blind_range'] ** 2) & (r2 <= params['max_range'] ** 2)

        mask = finite & in_range
        idx = mask.nonzero()[0]

        # downsample
        rate = params['downsample_rate']
        if rate > 0.0 and idx.size:
            keep = np.random.random(size=idx.size) > (rate / 100.0)
            idx = idx[keep]

        if idx.size == 0:
            return None

        # build Nx3 (one allocation)
        pts = np.empty((idx.size, 3), dtype=np.float32)
        pts[:, 0] = x[idx]; pts[:, 1] = y[idx]; pts[:, 2] = z[idx]
        return pts

    def _process_one(self, cloud: PointCloud2, R: np.ndarray, t: np.ndarray, pp: dict):
        t0 = time.time()
        x, y, z = _extract_xyz_views(cloud)
        pts = self._preprocess_xyz(x, y, z, pp)
        if pts is None or pts.size == 0:
            return None
        out = pts @ R.T + t
        self.times.append(time.time() - t0)
        return out

    def callback(self, *clouds: PointCloud2):
        futures = []
        for cloud, (R, t), pp in zip(clouds, self.transforms, self.pp_params):
            futures.append(self._pool.submit(self._process_one, cloud, R, t, pp))

        transformed_clouds = [f.result() for f in futures if f.result() is not None]
        if not transformed_clouds:
            return

        merged_points = np.concatenate(transformed_clouds, axis=0)

        header = Header()
        header.stamp = clouds[0].header.stamp
        header.frame_id = self.frame_id
        merged_cloud = create_cloud_xyz32(header, merged_points)
        self.publisher.publish(merged_cloud)

        # stats
        self._msg_count += 1
        if (self._msg_count % self._log_every) == 0:
            elapsed = time.time() - self._start_time
            rate = self._msg_count / elapsed
            print(f"\r[{self._msg_count:5d}] merged {len(transformed_clouds)} clouds → "
              f"{merged_points.shape[0]:7d} pts ({rate:5.1f} Hz)", end='', flush=True)
            print(f"average time = {np.mean(np.asarray(self.times)):.4f}+-{np.std(np.asarray(self.times)):.4f}", end='', flush=True)


    def destroy_node(self):
        try:
            self._pool.shutdown(wait=False, cancel_futures=True)
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    config_path = os.path.join(os.path.dirname(__file__), 'config/lidar_synchronizer.yaml')
    print(f"[LiDAR Synchronizer] Starting with config file: {config_path}")
    node = PointCloudMerger(config_path)
    
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("\n[LiDAR Synchronizer] Shutdown complete.\n")
if __name__ == '__main__':
    main()
