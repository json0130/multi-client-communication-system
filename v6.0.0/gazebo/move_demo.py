import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class BackAndForthDemo(Node):
    def __init__(self):
        super().__init__('back_and_forth_demo')
        # Publish to Robot 1's isolated velocity topic
        self.publisher_ = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self.run_demo()

    def run_demo(self):
        msg = Twist()
        
        # Move Forward
        self.get_logger().info('Moving forward...')
        msg.linear.x = 0.3
        self.publisher_.publish(msg)
        time.sleep(2)

        # Move Backward
        self.get_logger().info('Moving backward...')
        msg.linear.x = -0.3
        self.publisher_.publish(msg)
        time.sleep(2)

        # Stop
        self.get_logger().info('Stopping...')
        msg.linear.x = 0.0
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = BackAndForthDemo()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
