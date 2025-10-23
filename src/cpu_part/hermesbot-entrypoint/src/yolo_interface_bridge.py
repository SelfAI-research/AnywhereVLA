import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import SetBool

class SemanticEnableBridge(Node):
    """Bridge /semantic_map/enable (Bool) to YOLO SetBool service."""
    def __init__(self) -> None:
        super().__init__("semantic_enable_bridge")
        self.topic_interface = "/semantic_map/enable"
        self.yolo_enable_service = "/yolo_node/enable"
        self._cli = self.create_client(SetBool, self.yolo_enable_service)
        self.create_subscription(Bool, self.topic_interface, self._on_bool, 10)
        self.get_logger().info(f"Bridging {self.topic_interface}  →  {self.yolo_enable_service}")

    def _on_bool(self, msg: Bool) -> None:
        if not self._cli.service_is_ready():
            self.get_logger().warn(f"Service {self.yolo_enable_service} not ready")
            return
        self.get_logger().info(f"Forwarding {self.topic_interface} → {self.yolo_enable_service}")
        req = SetBool.Request(); req.data = msg.data
        self._cli.call_async(req)

def main() -> None:
    rclpy.init()
    node = SemanticEnableBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
