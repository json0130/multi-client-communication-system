import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from std_srvs.srv import Empty
import time
import math

class TourGuideDemo(Node):
    def __init__(self):
        super().__init__('tour_guide')
        # Isolate velocity commands to Robot 1
        self.publisher_ = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        
        # Service clients for world and entity resets
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        self.state_client = self.create_client(SetEntityState, '/set_entity_state')
        
        # Reset world state and teleport robots back to base coordinates
        self.reset_gazebo_environment()
        time.sleep(1.0)
        
        self.run_tour()

    def reset_gazebo_environment(self):
        self.get_logger().info('Resetting Gazebo environment and physics...')
        if self.reset_world_client.wait_for_service(timeout_sec=2.0):
            req = Empty.Request()
            self.reset_world_client.call_async(req)
        
        # Teleport all robots to their designated starting coordinates
        self.teleport_robot('robot1', 0.0, 0.0, 0.0)      # Room 1 Start (Facing East)
        self.teleport_robot('robot2', 4.0, -3.0, 3.14)    # Room 2 Start (Facing West)
        self.teleport_robot('robot3', 2.0, -6.0, 1.57)    # Room 3 Start (Facing North)

    def teleport_robot(self, robot_name, x, y, yaw):
        if not self.state_client.wait_for_service(timeout_sec=1.0):
            return
        req = SetEntityState.Request()
        state = EntityState()
        state.name = robot_name
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = 0.01
        state.pose.orientation.z = math.sin(yaw / 2.0)
        state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state = state
        
        future = self.state_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)  # Force execution

    def publish_twist(self, linear, angular, duration):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        
        start_time = time.time()
        # Continuously stream velocity commands at 10 Hz for the target duration
        while (time.time() - start_time) < duration:
            self.publisher_.publish(msg)
            rclpy.spin_once(self, timeout_sec=0)  # Process ROS 2 network events and services
            time.sleep(0.1)        # 10 Hz rate
        
        # Explicitly publish stop command
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)
        rclpy.spin_once(self, timeout_sec=0)
        time.sleep(0.5)

    def run_tour(self):
        TURN_90 = 1.57 
        TURN_180 = 3.14

        self.get_logger().info('--- Navigating to Robot 2 ---')
        self.publish_twist(0.3, 0.0, 3.0) 
        self.publish_twist(0.0, -1.0, TURN_90)  # Turn Right (South)
        self.publish_twist(0.4, 0.0, 5.0)       # Corridor
        self.publish_twist(0.0, 1.0, TURN_90)   # Turn Left (East into Room 2)
        self.publish_twist(0.3, 0.0, 3.0)
        self.get_logger().info('Arrived at Robot 2. Pausing...')
        time.sleep(3.0)

        self.get_logger().info('--- Navigating to Robot 3 ---')
        self.publish_twist(0.0, 1.0, TURN_180)  # Turn Around 180
        self.publish_twist(0.3, 0.0, 3.0)       # Exit Room 2
        self.publish_twist(0.0, 1.0, TURN_90)   # Turn Left (South)
        self.publish_twist(0.3, 0.0, 6.0)       # Drive to Robot 3
        self.get_logger().info('Arrived at Robot 3. Pausing...')
        time.sleep(3.0)

        self.get_logger().info('--- Returning to Start ---')
        self.publish_twist(0.0, 1.0, TURN_180)  # Turn Around 180 (North)
        self.publish_twist(0.3, 0.0, 6.0)       # Drive up
        self.publish_twist(0.0, 1.0, TURN_90)   # Turn Left (West)
        self.publish_twist(0.4, 0.0, 5.0)       # Corridor
        self.publish_twist(0.0, -1.0, TURN_90)  # Turn Right (North)
        self.publish_twist(0.3, 0.0, 3.0)       # Final drive
        self.get_logger().info('Tour Complete. Ready for next run.')

def main(args=None):
    rclpy.init(args=args)
    node = TourGuideDemo()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
