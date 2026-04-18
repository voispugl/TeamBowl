#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener


_POINTFIELD_TO_DTYPE = {
    PointField.INT8: np.dtype(np.int8),
    PointField.UINT8: np.dtype(np.uint8),
    PointField.INT16: np.dtype(np.int16),
    PointField.UINT16: np.dtype(np.uint16),
    PointField.INT32: np.dtype(np.int32),
    PointField.UINT32: np.dtype(np.uint32),
    PointField.FLOAT32: np.dtype(np.float32),
    PointField.FLOAT64: np.dtype(np.float64),
}


def _dtype_from_fields(fields, point_step: int) -> np.dtype:
    return np.dtype(
        {
            'names': [field.name for field in sorted(fields, key=lambda item: item.offset)],
            'formats': [
                _POINTFIELD_TO_DTYPE[field.datatype]
                if field.count == 1
                else (_POINTFIELD_TO_DTYPE[field.datatype], field.count)
                for field in sorted(fields, key=lambda item: item.offset)
            ],
            'offsets': [field.offset for field in sorted(fields, key=lambda item: item.offset)],
            'itemsize': point_step,
        }
    )


def _quat_to_rot_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


class NavCloudFilter(Node):
    def __init__(self):
        super().__init__('nav_cloud_filter')

        self.declare_parameter('input_topic', '/oak/points')
        self.declare_parameter('output_topic', '/oak/nav_points')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('transform_timeout_s', 0.1)
        self.declare_parameter('min_x_m', 0.15)
        self.declare_parameter('max_x_m', 2.50)
        self.declare_parameter('min_y_m', -1.20)
        self.declare_parameter('max_y_m', 1.20)
        self.declare_parameter('min_z_m', -0.10)
        self.declare_parameter('max_z_m', 1.20)
        self.declare_parameter('voxel_leaf_size_m', 0.08)
        self.declare_parameter('min_points_per_voxel', 3)

        self.input_topic = str(self.get_parameter('input_topic').value)
        self.output_topic = str(self.get_parameter('output_topic').value)
        self.target_frame = str(self.get_parameter('target_frame').value)
        self.transform_timeout = Duration(
            seconds=float(self.get_parameter('transform_timeout_s').value)
        )
        self.min_x_m = float(self.get_parameter('min_x_m').value)
        self.max_x_m = float(self.get_parameter('max_x_m').value)
        self.min_y_m = float(self.get_parameter('min_y_m').value)
        self.max_y_m = float(self.get_parameter('max_y_m').value)
        self.min_z_m = float(self.get_parameter('min_z_m').value)
        self.max_z_m = float(self.get_parameter('max_z_m').value)
        self.voxel_leaf_size_m = max(1e-3, float(self.get_parameter('voxel_leaf_size_m').value))
        self.min_points_per_voxel = max(1, int(self.get_parameter('min_points_per_voxel').value))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_tf_warn_ns = 0

        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self._cloud_cb,
            qos_profile_sensor_data,
        )
        self.pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f'nav_cloud_filter started | in={self.input_topic} out={self.output_topic} '
            f'frame={self.target_frame}'
        )

    def _cloud_cb(self, msg: PointCloud2):
        points = self._read_xyz_points(msg)
        if points.size == 0:
            return

        transform = self._lookup_transform(msg)
        if transform is None:
            return

        points = self._transform_points(points, transform)
        points = self._crop_points(points)
        if points.size == 0:
            self.pub.publish(self._build_cloud(msg, points))
            return

        points = self._voxel_filter(points)
        self.pub.publish(self._build_cloud(msg, points))

    def _lookup_transform(self, msg: PointCloud2):
        try:
            return self.tf_buffer.lookup_transform(
                self.target_frame,
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=self.transform_timeout,
            )
        except TransformException as exc:
            now_ns = self.get_clock().now().nanoseconds
            if now_ns - self._last_tf_warn_ns > int(2e9):
                self.get_logger().warn(
                    f'Failed to transform {msg.header.frame_id} -> {self.target_frame}: {exc}'
                )
                self._last_tf_warn_ns = now_ns
            return None

    def _read_xyz_points(self, msg: PointCloud2) -> np.ndarray:
        dtype = _dtype_from_fields(msg.fields, msg.point_step)
        cloud = np.frombuffer(memoryview(msg.data), dtype=dtype, count=msg.width * msg.height)
        points = np.column_stack((cloud['x'], cloud['y'], cloud['z'])).astype(np.float32, copy=False)
        return points[np.isfinite(points).all(axis=1)]

    def _transform_points(self, points: np.ndarray, transform) -> np.ndarray:
        rot = transform.transform.rotation
        trans = transform.transform.translation
        rot_matrix = _quat_to_rot_matrix(rot.x, rot.y, rot.z, rot.w)
        points = points @ rot_matrix.T
        points[:, 0] += trans.x
        points[:, 1] += trans.y
        points[:, 2] += trans.z
        return points

    def _crop_points(self, points: np.ndarray) -> np.ndarray:
        mask = (
            (points[:, 0] >= self.min_x_m)
            & (points[:, 0] <= self.max_x_m)
            & (points[:, 1] >= self.min_y_m)
            & (points[:, 1] <= self.max_y_m)
            & (points[:, 2] >= self.min_z_m)
            & (points[:, 2] <= self.max_z_m)
        )
        return points[mask]

    def _voxel_filter(self, points: np.ndarray) -> np.ndarray:
        voxel_indices = np.floor(points / self.voxel_leaf_size_m).astype(np.int32)
        unique_voxels, inverse, counts = np.unique(
            voxel_indices, axis=0, return_inverse=True, return_counts=True
        )
        sums = np.zeros((unique_voxels.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, points)
        keep = counts >= self.min_points_per_voxel
        if not np.any(keep):
            return np.empty((0, 3), dtype=np.float32)
        centroids = sums[keep] / counts[keep, None]
        return centroids.astype(np.float32)

    def _build_cloud(self, source_msg: PointCloud2, points: np.ndarray) -> PointCloud2:
        cloud = PointCloud2()
        cloud.header.stamp = source_msg.header.stamp
        cloud.header.frame_id = self.target_frame
        cloud.height = 1
        cloud.width = int(points.shape[0])
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        cloud.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
        return cloud


def main():
    rclpy.init()
    node = NavCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
