import numpy as np
from scipy import ndimage as ndi
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid


class FrontierResult:
    """Holds frontier clusters and provides RViz markers."""

    def __init__(self, world_points: list[tuple[float, float]], yaws: list[float] | None = None) -> None:
        self.world_points = world_points
        self.yaws = yaws if yaws is not None else [None] * len(world_points)

    def to_markers_msg(self, frame_id: str) -> MarkerArray:
        marker_array = MarkerArray()

        # Clear previous markers
        clear = Marker()
        clear.action = Marker.DELETEALL
        marker_array.markers.append(clear)

        for idx, (world_x, world_y) in enumerate(self.world_points):
            # centroid sphere
            sphere_marker = Marker()
            sphere_marker.header.frame_id = frame_id
            sphere_marker.ns = "frontiers"
            sphere_marker.id = idx * 2
            sphere_marker.type = Marker.SPHERE
            sphere_marker.action = Marker.ADD
            sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.2
            sphere_marker.color.a = 1.0
            sphere_marker.color.r = 0.1
            sphere_marker.color.g = 0.8
            sphere_marker.color.b = 0.1
            sphere_marker.pose.position.x = float(world_x)
            sphere_marker.pose.position.y = float(world_y)
            sphere_marker.pose.orientation.w = 1.0
            marker_array.markers.append(sphere_marker)

            # yaw arrow (if available)
            yaw = self.yaws[idx] if idx < len(self.yaws) else None
            if yaw is not None:
                arrow_marker = Marker()
                arrow_marker.header.frame_id = frame_id
                arrow_marker.ns = "frontier_yaw"
                arrow_marker.id = idx * 2 + 1
                arrow_marker.type = Marker.ARROW
                arrow_marker.action = Marker.ADD
                # length, shaft/head thickness
                arrow_marker.scale.x = 0.6
                arrow_marker.scale.y = 0.06
                arrow_marker.scale.z = 0.06
                arrow_marker.color.a = 1.0
                arrow_marker.color.r = 0.2
                arrow_marker.color.g = 0.2
                arrow_marker.color.b = 1.0
                arrow_marker.pose.position.x = float(world_x)
                arrow_marker.pose.position.y = float(world_y)
                # yaw -> quaternion about Z
                half_yaw = yaw * 0.5
                arrow_marker.pose.orientation.z = np.sin(half_yaw)
                arrow_marker.pose.orientation.w = np.cos(half_yaw)
                marker_array.markers.append(arrow_marker)
        return marker_array


class FrontierFinder:
    """Unknown adjacent to free -> split into multiple chunks -> centroids -> FoV-opt yaw -> standoff -> NMS."""

    def __init__(
        self,
        occ_threshold_free: int,
        min_cluster_px: int,
        max_chunk_px: int,
        min_goal_separation_m: float,
        offset_m: float,
        max_candidates: int,
        angle_samples: int,
        gain_stride: int = 1,
    ) -> None:
        self.occ_free = int(occ_threshold_free)
        self.min_cluster_px = int(min_cluster_px)
        self.max_chunk_px = int(max_chunk_px)
        self.min_goal_sep = float(min_goal_separation_m)
        self.standoff_m = float(offset_m)
        self.max_candidates = int(max_candidates)
        self.angle_samples = int(angle_samples)
        self.gain_stride = max(1, int(gain_stride))

    @staticmethod
    def _world_of(pixel_x: np.ndarray, pixel_y: np.ndarray, grid: OccupancyGrid) -> tuple[np.ndarray, np.ndarray]:
        resolution = grid.info.resolution
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        world_x = origin_x + (pixel_x.astype(np.float64) + 0.5) * resolution
        world_y = origin_y + (pixel_y.astype(np.float64) + 0.5) * resolution
        return world_x, world_y

    @staticmethod
    def _circle_mask(grid: OccupancyGrid, center_xy: tuple[float, float], radius: float) -> np.ndarray:
        """Cells within circle centered at center_xy with given radius (meters)."""
        if radius is None or not np.isfinite(radius) or radius <= 0.0:
            # No restriction
            height, width = grid.info.height, grid.info.width
            return np.ones((height, width), dtype=bool)
        height, width = grid.info.height, grid.info.width
        resolution = grid.info.resolution
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        center_x, center_y = float(center_xy[0]), float(center_xy[1])

        world_x_coords = origin_x + (np.arange(width) + 0.5) * resolution
        world_y_coords = origin_y + (np.arange(height) + 0.5) * resolution
        grid_world_x, grid_world_y = np.meshgrid(world_x_coords, world_y_coords)
        return ((grid_world_x - center_x) ** 2 + (grid_world_y - center_y) ** 2) <= (radius ** 2)

    def _best_yaw_for_point(
        self,
        point_x: float,
        point_y: float,
        unknown_mask: np.ndarray,
        world_grid_x: np.ndarray,
        world_grid_y: np.ndarray,
        fov_deg: float,
        gain_radius_m: float,
    ) -> float | None:
        """Choose yaw that maximizes count of unknown cells inside FoV wedge around (px,py)."""
        if not np.isfinite(fov_deg) or fov_deg <= 0.0:
            return None, 0
        delta_x = world_grid_x - point_x
        delta_y = world_grid_y - point_y
        radius_sq = delta_x * delta_x + delta_y * delta_y
        local_mask = (radius_sq <= (gain_radius_m * gain_radius_m)) & unknown_mask
        if not local_mask.any():
            return None, 0
        bearing_angles = np.arctan2(delta_y, delta_x)
        half_fov_rad = np.radians(float(fov_deg) / 2.0)

        best_yaw, best_gain = None, -1
        for sample_idx in range(self.angle_samples):
            yaw_candidate = (2.0 * np.pi) * (sample_idx / self.angle_samples)
            angle_diff = (bearing_angles - yaw_candidate + np.pi) % (2.0 * np.pi) - np.pi
            gain = np.count_nonzero(local_mask & (np.abs(angle_diff) <= half_fov_rad))
            if gain > best_gain:
                best_yaw, best_gain = yaw_candidate, gain
        return best_yaw, best_gain

    @staticmethod
    def _chunked_centroids(pixel_x: np.ndarray, pixel_y: np.ndarray, max_chunk_px: int) -> list[tuple[float, float]]:
        """
        Split points into chunks and compute centroids of each chunk.
        """
        num_pixels = pixel_x.size
        if num_pixels <= max_chunk_px:
            return [(pixel_x.mean(), pixel_y.mean())]
        cluster_cx, cluster_cy = pixel_x.mean(), pixel_y.mean()
        angles = np.arctan2(pixel_y - cluster_cy, pixel_x - cluster_cx)
        sort_idx = np.argsort(angles)
        pixel_x, pixel_y = pixel_x[sort_idx], pixel_y[sort_idx]
        num_chunks = int(np.ceil(num_pixels / float(max_chunk_px)))
        chunk_indices = np.array_split(np.arange(num_pixels), num_chunks)
        centroids: list[tuple[float, float]] = []
        for indices in chunk_indices:
            centroids.append((pixel_x[indices].mean(), pixel_y[indices].mean()))
        return centroids

    @staticmethod
    def _nms_keep(points_xy: list[tuple[float, float]], min_separation_m: float) -> list[int]:
        """
        Non-maximum suppression: keep points that are not too close to each other.
        """
        kept_indices: list[int] = []
        for idx_i, (xi, yi) in enumerate(points_xy):
            is_far_enough = True
            for idx_j in kept_indices:
                xj, yj = points_xy[idx_j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 < (min_separation_m * min_separation_m):
                    is_far_enough = False
                    break
            if is_far_enough:
                kept_indices.append(idx_i)
        return kept_indices

    def find(
        self,
        map_msg: OccupancyGrid,
        explore_range: float,
        robot_xy: tuple[float, float],
        robot_yaw: float,
        camera_fov: float | None = None,
        camera_range: float | None = None,
        explore_center_xy: tuple[float, float] = (0.0, 0.0),
    ) -> FrontierResult:
        """
        Returns centroids of frontier clusters subject to bounds and FoV.
        Frontier = unknown (-1) & neighbor_of(free).

        - Bounds: candidates must lie within circle radius `explore_range` from `explore_center_xy`.
        - FoV yaw selection: if camera_fov is provided, each centroid's yaw is chosen to
          maximize unknown coverage inside a wedge (±FoV/2) within a small radius.
        """
        map_height, map_width = map_msg.info.height, map_msg.info.width
        occupancy = np.asarray(map_msg.data, dtype=np.int8).reshape((map_height, map_width))

        unknown_mask = occupancy == -1
        free_mask = (0 <= occupancy) & (occupancy <= self.occ_free)
        occupied_mask = occupancy > self.occ_free

        # Radial bound from origin (or provided center)
        explore_mask = self._circle_mask(map_msg, explore_center_xy, explore_range)

        # Unknown near free (8-neighborhood), restricted by explore_range
        dilated_free_mask = ndi.binary_dilation(free_mask, structure=np.ones((3, 3), dtype=bool))
        dilated_occupied_mask = ndi.binary_dilation(occupied_mask, structure=np.ones((5, 5), dtype=bool))
        frontier_mask = unknown_mask & dilated_free_mask & explore_mask & ~dilated_occupied_mask

        # [DEBUG] Return frontier_mask as points
        # fy, fx = np.where(frontier_mask)
        # if fy.size:
        #     sel = np.arange(0, fy.size, 2)
        #     fx, fy = fx[sel], fy[sel]
        #     wx, wy = self._world_of(fx, fy, map_msg)
        #     # outward yaw (toward unknown) from gradient of the unknown mask
        #     grad_y, grad_x = np.gradient(unknown_mask.astype(np.float32))
        #     yaw_list = np.arctan2(grad_y[fy, fx], grad_x[fy, fx]).tolist()
        #     return FrontierResult(list(zip(wx.tolist(), wy.tolist())), yaws=yaw_list)

        if not frontier_mask.any():
            return FrontierResult([])

        # FAST: get component slices to avoid full-image scans per label
        cc_labels, cc_count = ndi.label(frontier_mask, structure=np.ones((3, 3), dtype=bool))
        cc_slices = ndi.find_objects(cc_labels)

        # Precompute world coordinate grids once
        resolution = map_msg.info.resolution
        origin_x = map_msg.info.origin.position.x
        origin_y = map_msg.info.origin.position.y
        world_x_coords = origin_x + (np.arange(map_width) + 0.5) * resolution
        world_y_coords = origin_y + (np.arange(map_height) + 0.5) * resolution
        world_grid_x, world_grid_y = np.meshgrid(world_x_coords, world_y_coords)

        candidate_points: list[tuple[float, float]] = []
        candidate_yaws: list[float] = []

        robot_x, robot_y = float(robot_xy[0]), float(robot_xy[1])

        # pixel radius for local FoV gain window
        gain_radius_m = float(camera_range) if (camera_range is not None and np.isfinite(camera_range) and camera_range > 0.0) else 3.0
        gain_radius_px = max(1, int(np.ceil(gain_radius_m / resolution)))
        stride = self.gain_stride

        # iterate connected components via their slices
        for label_idx, comp_slice in enumerate(cc_slices, start=1):
            if comp_slice is None:
                continue
            comp_block = cc_labels[comp_slice] == label_idx
            comp_ys_rel, comp_xs_rel = np.where(comp_block)
            if comp_xs_rel.size < self.min_cluster_px:
                continue
            comp_ys = comp_ys_rel + comp_slice[0].start
            comp_xs = comp_xs_rel + comp_slice[1].start

            pixel_centroids = self._chunked_centroids(comp_xs.astype(float), comp_ys.astype(float), self.max_chunk_px)

            for centroid_px_x, centroid_px_y in pixel_centroids:
                world_x_arr, world_y_arr = self._world_of(np.array([centroid_px_x]), np.array([centroid_px_y]), map_msg)
                goal_x, goal_y = float(world_x_arr[0]), float(world_y_arr[0])

                if camera_fov is not None:
                    # ROI slice for FoV gain around the centroid (crop + optional decimation)
                    ci = int(round(centroid_px_y))  # row index
                    cj = int(round(centroid_px_x))  # col index
                    i0 = max(0, ci - gain_radius_px); i1 = min(map_height, ci + gain_radius_px + 1)
                    j0 = max(0, cj - gain_radius_px); j1 = min(map_width,  cj + gain_radius_px + 1)
                    unknown_local = unknown_mask[i0:i1:stride, j0:j1:stride]
                    X_local = world_grid_x[i0:i1:stride, j0:j1:stride]
                    Y_local = world_grid_y[i0:i1:stride, j0:j1:stride]

                    yaw, gain = self._best_yaw_for_point(
                        goal_x, goal_y,
                        unknown_mask=unknown_local,
                        world_grid_x=X_local,
                        world_grid_y=Y_local,
                        fov_deg=float(camera_fov),
                        gain_radius_m=gain_radius_m,
                    )
                    print(f"({goal_x}, {goal_y}) - {gain}")
                    if yaw is None:
                        yaw = np.atan2(goal_y - robot_y, goal_x - robot_x)
                    if gain < 50:
                        continue
                else:
                    yaw = np.atan2(goal_y - robot_y, goal_x - robot_x)

                # standoff in free side (towards robot)
                robot_to_goal_dx, robot_to_goal_dy = goal_x - robot_x, goal_y - robot_y
                distance_rg = np.hypot(robot_to_goal_dx, robot_to_goal_dy)
                if distance_rg > 1e-6:
                    unit_dx, unit_dy = robot_to_goal_dx / distance_rg, robot_to_goal_dy / distance_rg
                    standoff_x, standoff_y = goal_x - self.standoff_m * unit_dx, goal_y - self.standoff_m * unit_dy
                else:
                    standoff_x, standoff_y = goal_x, goal_y

                # keep inside explore circle
                center_x, center_y = explore_center_xy
                radial_dist2 = (standoff_x - center_x) ** 2 + (standoff_y - center_y) ** 2
                if radial_dist2 > explore_range * explore_range:
                    clamp_scale = (explore_range - 1e-3) / np.sqrt(radial_dist2)
                    standoff_x = center_x + (standoff_x - center_x) * clamp_scale
                    standoff_y = center_y + (standoff_y - center_y) * clamp_scale

                candidate_points.append((standoff_x, standoff_y))
                candidate_yaws.append(yaw)

        if not candidate_points:
            return FrontierResult([])

        order_idx = np.argsort([np.hypot(px - robot_x, py - robot_y) for (px, py) in candidate_points])
        candidate_points = [candidate_points[i] for i in order_idx]
        candidate_yaws = [candidate_yaws[i] for i in order_idx]

        kept_indices = self._nms_keep(candidate_points, self.min_goal_sep)
        candidate_points = [candidate_points[i] for i in kept_indices]
        candidate_yaws = [candidate_yaws[i] for i in kept_indices]

        if len(candidate_points) > self.max_candidates:
            candidate_points = candidate_points[: self.max_candidates]
            candidate_yaws = candidate_yaws[: self.max_candidates]

        return FrontierResult(candidate_points, yaws=candidate_yaws)
