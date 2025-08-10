#!/usr/bin/env python3 

"""
Author : Belabed Abdelkader
Time : 22/07/2025 at 03:00 pm

"""

import rclpy
from rclpy.node import Node
import time

import math
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf_transformations import euler_from_quaternion

import point_cloud2 as pc2
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
#from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose, Point, Quaternion
from rclpy.logging import LoggingSeverity

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.2
SEVERITY = LoggingSeverity.ERROR

last_odom = None
environment_dim = 20
velodyne_data = np.ones(environment_dim) * 10


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    # List of restricted areas in (xmin, xmax, ymin, ymax) format
    forbidden_zones = [
        (-6.2, -3.8, 3.8, 6.2),
        (-2.7, -1.3, -0.2, 4.7),
        (-4.2, -0.3, 1.3, 2.7),
        (-4.2, -0.8, -4.2, -2.3),
        (-3.7, -1.3, -2.7, -0.8),
        (0.8, 4.2, -3.2, -1.8),
        (2.5, 4.0, -3.2, 0.7),
        (3.8, 6.2, -4.2, -3.3),
        (1.3, 4.2, 1.5, 3.7),
        (-7.2, -3.0, -1.5, 0.5)
    ]

    # Check if the point is inside any restricted area
    for xmin, xmax, ymin, ymax in forbidden_zones:
        if xmin < x < xmax and ymin < y < ymax:
            goal_ok = False
            break
    
    # Check world limits
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv(Node):
    """Superclass for all Gazebo environments."""

    def __init__(self):
        super().__init__('env')
        self.environment_dim = 20
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0

        self.set_self_state = EntityState()
        self.set_self_state.name = "td_robot"

        # Create a Pose object and assign position and orientation
        pose = Pose()
        pose.position = Point(x=-1.0, y=0.0, z=0.0)  # Set position
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)  # Set orientation

        # Set up the ROS_2 publishers and subscribers
        self.vel_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.set_state = self.create_publisher(EntityState, "gazebo/set_model_state", 10)

        self.unpause = self.create_client(Empty, "/unpause_physics")
        self.pause = self.create_client(Empty, "/pause_physics")
        self.reset_proxy = self.create_client(Empty, "/reset_world")
        self.req = Empty.Request

        self.publisher = self.create_publisher(MarkerArray, "goal_point", 3)
        self.publisher2 = self.create_publisher(MarkerArray, "linear_velocity", 1)
        self.publisher3 = self.create_publisher(MarkerArray, "angular_velocity", 1)

        self.set_state_client = SetModelStateClient()  # Initialize SetModelStateClient

    # Perform an action and read a new state
    def step(self, action):
        global velodyne_data
        target = False
        
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = float(action[0])
        vel_cmd.angular.z = float(action[1])
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')

        try:
            pass
            self.pause.call_async(Empty.Request())
        except (rclpy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(velodyne_data)
        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = last_odom.pose.pose.position.x
        self.odom_y = last_odom.pose.pose.position.y
        
        quaternion = [last_odom.pose.pose.orientation.x,
                      last_odom.pose.pose.orientation.y,
                      last_odom.pose.pose.orientation.z,
                      last_odom.pose.pose.orientation.w,]

        euler = euler_from_quaternion(quaternion)
        # Convert to degrees if necessary
        #euler_degrees = [math.degrees(angle) for angle in euler]
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        #print("Distance to Goal:", distance, flush=True)
        #print("Yaw Angle (Theta):", angle, flush=True)

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            print("\033[92mGOAL!!!\033[0m", flush=True)
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.

        while not self.reset_proxy.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('reset : service not available, waiting again...')

        try:
            self.reset_proxy.call_async(Empty.Request())
        except rclpy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        #quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        quaternion = R.from_euler('xyz', [0.0, 0.0, angle]).as_quat()
        # Ensure quaternion is valid
        assert np.isclose(np.linalg.norm(quaternion), 1.0), "Invalid quaternion!"
        # Create a deep copy of the current state
        object_state = EntityState()
        object_state.name = self.set_self_state.name  # Assign the same entity name

        # Initialize position variables
        #x = 0.0
        #y = 0.0
        position_ok = False

        # Generate a random valid position until check_pos confirms it's valid
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)  # Ensure position is valid according to your logic

        # Update position and orientation
        object_state.pose = Pose()
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        #object_state.pose.position.z = 0.0  # Assume the object stays on the ground plane

        # Update orientation using a valid quaternion (assume quaternion is provided or calculated)
        object_state.pose.orientation.x = quaternion[0]
        object_state.pose.orientation.y = quaternion[1]
        object_state.pose.orientation.z = quaternion[2]
        object_state.pose.orientation.w = quaternion[3]

        # Publish the updated state using the client
        self.set_state_client.set_state(object_state.name, object_state.pose)

        # Update odometry for tracking purposes
        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y


        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        self.publish_markers([0.0, 0.0])

        while not self.unpause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.unpause.call_async(Empty.Request())
        except:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        while not self.pause.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('service not available, waiting again...')

        try:
            self.pause.call_async(Empty.Request())
        except:
            print("/gazebo/pause_physics service call failed")

        v_state = []
        v_state[:] = velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

        # Delete the existing goal marker before spawning a new one
        self.delete_goal_marker()

        rclpy.spin_once(self, timeout_sec=0.2)

        # Once a valid goal is found, visualize it in Gazebo
        self.spawn_goal_marker(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = EntityState()
            box_state.name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

            # Use SetModelStateClient to set the box state
            self.set_state_client.set_state(name, box_state.pose)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0.0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = float(abs(action[0]))
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 0.0
        marker2.color.g = 1.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 6.0
        marker2.pose.position.y = 0.0
        marker2.pose.position.z = 0.0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = float(abs(action[1]))
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 6.0
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0.0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    #! Visualize the goal_point in gazebo environment
    def spawn_goal_marker(self, x, y):
        # Path to your SDF model file
        model_path = '/home/ubuntu/DRL_robot_navigation_ros2/src/td3/gazebo_goal/goal.xml'
        with open(model_path, 'r') as f:
            model_xml = f.read()

        # Define the pose for the goal marker
        goal_pose = Pose()
        goal_pose.position.x = x
        goal_pose.position.y = y
        # goal_pose.position.z = z (if needed)

        # Call the service to spawn the model
        client = self.create_client(SpawnEntity, '/spawn_entity')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("SpawnEntity service not available!")
            return

        request = SpawnEntity.Request()
        request.name = "goal_marker"
        request.xml = model_xml
        request.robot_namespace = ""
        request.initial_pose = goal_pose

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            #self.get_logger().info(f"Goal marker spawned successfully at ({x}, {y})!")
            pass
        else:
            self.get_logger().error(f"Failed to spawn goal marker: {future.exception()}")

    def delete_goal_marker(self, name="goal_marker"):
        client = self.create_client(DeleteEntity, '/delete_entity')
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("DeleteEntity service not available!")
            return

        request = DeleteEntity.Request()
        request.name = name

        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            #self.get_logger().info(f"Deleted existing goal marker: {name}")
            pass
        else:
            self.get_logger().error(f"Failed to delete goal marker: {future.exception()}")

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            print("\033[91mCollision is detected!\033[0m", flush=True)
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            #env.get_logger().info("reward 100")
            return 100.0
        elif collision:
            #env.get_logger().info("reward -100")
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

class Odom_subscriber(Node):

    def __init__(self):
        super().__init__('odom_subscriber')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.subscription

    def odom_callback(self, od_data):
        global last_odom
        last_odom = od_data

class Velodyne_subscriber(Node):

    def __init__(self):
        super().__init__('velodyne_subscriber')
        self.subscription = self.create_subscription(
            PointCloud2,
            "/velodyne_points",
            self.velodyne_callback,
            10)
        self.subscription

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / environment_dim]]
        for m in range(environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / environment_dim]
            )
        self.gaps[-1][-1] += 0.03

    def velodyne_callback(self, v):
        global velodyne_data
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        velodyne_data = np.ones(environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        velodyne_data[j] = min(velodyne_data[j], dist)
                        break

class SetModelStateClient(Node):
    def __init__(self):
        super().__init__("set_entity_state_client")
        self.get_logger().set_level(SEVERITY)
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        self.request = SetEntityState.Request()

    def set_state(self, name, new_pose):
        self.request.state.name = name
        self.request.state.pose = new_pose
        self.future = self.client.call_async(self.request)
