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
        self.publisher_ = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        self.state_client = self.create_client(SetEntityState, '/set_entity_state')
        
        self.reset_gazebo_environment()
        time.sleep(1.0)
        self.run_tour()

    def reset_gazebo_environment(self):
        self.get_logger().info('Resetting Gazebo environment and physics...')
        if self.reset_world_client.wait_for_service(timeout_sec=2.0):
            req = Empty.Request()
            self.reset_world_client.call_async(req)
        
        # Teleport entities to designated coordinates
        self.teleport_robot('robot1', 0.092, 5.528, 0.0)      # Room 1 Fixed Start (Facing East)
        self.teleport_robot('robot2', 9.695, 1.035, 3.140)    # Corridor Right (Facing West)
        self.teleport_robot('robot3', 8.128, -5.582, 1.570)   # Lower Corridor (Facing North)

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
        rclpy.spin_until_future_complete(self, future)

    def publish_twist(self, linear, angular, duration):
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            self.publisher_.publish(msg)
            rclpy.spin_once(self, timeout_sec=0)
            time.sleep(0.1)
        
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher_.publish(msg)
        rclpy.spin_once(self, timeout_sec=0)
        time.sleep(0.5)

    def run_tour(self):
        TURN_90_LEFT = (0.0, 0.5, 3.14)
        TURN_90_RIGHT = (0.0, -0.5, 3.14)
        TURN_180 = (0.0, 0.5, 6.28)

        # --- STEP 1: APPROACH DOOR ---
        self.get_logger().info('--- Approaching Door ---')
        self.publish_twist(*TURN_90_LEFT)       # Turn Left 90° (Facing North towards door)
        self.publish_twist(0.3, 0.0, 4.3)       # Drive North to Doorway
        self.get_logger().info('Arrived at Door. Pausing...')
        time.sleep(2.0)

        # --- STEP 2: HEAD DOWN & TURN LEFT TO ROBOT 2 ---
        self.get_logger().info('--- 180 Turn and Heading Down to Corridor ---')
        self.publish_twist(*TURN_180)           # Turn 180° (Facing South)
        self.publish_twist(0.3, 0.0, 19.2)      # Drive South down hall to main corridor
        self.get_logger().info('--- Turning Left to Approach Robot 2 ---')
        self.publish_twist(*TURN_90_LEFT)       # Turn Left 90° (Facing East)
        self.publish_twist(0.3, 0.0, 32.0)      # Drive East along corridor to Robot 2
        self.get_logger().info('Arrived at Robot 2. Pausing...')
        time.sleep(3.0)

        # --- STEP 3: NAVIGATE TO ROBOT 3 ---
        self.get_logger().info('--- Navigating to Robot 3 ---')
        self.publish_twist(*TURN_180)           # Turn 180° (Facing West)
        self.publish_twist(0.3, 0.0, 5.2)       # Drive West to Robot 3 Hallway Junction
        self.publish_twist(*TURN_90_LEFT)       # Turn Left 90° (Facing South)
        self.publish_twist(0.3, 0.0, 22.1)      # Drive South to Robot 3
        self.get_logger().info('Arrived at Robot 3. Pausing...')
        time.sleep(3.0)

        # --- STEP 4: RETURN TO START ---
        self.get_logger().info('--- Returning to Start Pose ---')
        self.publish_twist(*TURN_180)           # Turn 180° (Facing North)
        self.publish_twist(0.3, 0.0, 22.1)      # Drive North to Main Corridor
        self.publish_twist(*TURN_90_LEFT)       # Turn Left 90° (Facing West)
        self.publish_twist(0.3, 0.0, 26.8)      # Drive West to Room 1 Hallway Junction
        self.publish_twist(*TURN_90_RIGHT)      # Turn Right 90° (Facing North into Room 1)
        self.publish_twist(0.3, 0.0, 15.0)      # Drive North to Start position
        self.publish_twist(*TURN_90_RIGHT)      # Turn Right 90° (Facing East to reset orientation)
        self.get_logger().info('Tour Complete. Returned to base pose.')

def main(args=None):
    rclpy.init(args=args)
    node = TourGuideDemo()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
