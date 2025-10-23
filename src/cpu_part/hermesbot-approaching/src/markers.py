"""RViz markers: edge, candidates, goal."""
from typing import Iterable, Sequence, Tuple
import math
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion

class MarkersPublisher:
    """Publishes debug visuals for edge, candidates, and chosen goal."""
    def __init__(self, node: Node, edge_topic: str, candidates_topic: str, goal_topic: str) -> None:
        self.edge_pub = node.create_publisher(MarkerArray, edge_topic, 10)
        self.cand_pub = node.create_publisher(MarkerArray, candidates_topic, 10)
        self.goal_pub = node.create_publisher(MarkerArray, goal_topic, 10)

    def publish_edge(self, frame: str, pts_xy: Sequence[Tuple[float,float]]) -> None:
        ma = MarkerArray(); m = Marker()
        m.header.frame_id, m.ns, m.id = frame, "edge", 0
        m.type, m.action = Marker.POINTS, Marker.ADD
        m.scale.x = m.scale.y = 0.03; m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 0.9
        m.pose.orientation.w = 1.0
        m.points = [Point(x=x, y=y, z=0.0) for x, y in pts_xy]
        ma.markers.append(m); self.edge_pub.publish(ma)

    def publish_candidates(self, frame: str, cands: Iterable[Tuple[float,float,float]], limit: int = 60) -> None:
        ma = MarkerArray()
        for idx, (x, y, yaw) in enumerate(cands):
            if idx >= limit: break
            m = Marker(); m.header.frame_id, m.ns, m.id = frame, "candidates", idx
            m.type, m.action = Marker.ARROW, Marker.ADD
            m.scale.x, m.scale.y, m.scale.z = 0.25, 0.05, 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.9, 0.2, 0.9
            m.pose.position.x, m.pose.position.y = x, y
            qz, qw = math.sin(yaw*0.5), math.cos(yaw*0.5)
            m.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
            ma.markers.append(m)
        self.cand_pub.publish(ma)

    def publish_goal(self, frame: str, x: float, y: float, yaw: float) -> None:
        ma = MarkerArray(); m = Marker()
        m.header.frame_id, m.ns, m.id = frame, "goal", 0
        m.type, m.action = Marker.ARROW, Marker.ADD
        m.scale.x, m.scale.y, m.scale.z = 0.35, 0.08, 0.08
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 0.4, 0.95
        m.pose.position.x, m.pose.position.y = x, y
        qz, qw = math.sin(yaw*0.5), math.cos(yaw*0.5)
        m.pose.orientation = Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        ma.markers.append(m); self.goal_pub.publish(ma)
