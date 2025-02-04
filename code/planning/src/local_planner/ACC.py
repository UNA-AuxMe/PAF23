#!/usr/bin/env python
import numpy as np
import ros_compatibility as roscomp
from carla_msgs.msg import CarlaSpeedometer  # , CarlaWorldInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from ros_compatibility.node import CompatibleNode
from rospy import Publisher, Subscriber
from std_msgs.msg import Bool, Float32, Float32MultiArray
from utils import calculate_rule_of_thumb, interpolate_speed


class ACC(CompatibleNode):
    """ACC (Adaptive Cruise Control) calculates and publishes the desired speed based on
    possible collisions, the current speed, the trajectory, and the speed limits."""

    def __init__(self):
        super(ACC, self).__init__("ACC")
        self.role_name = self.get_param("role_name", "hero")
        self.control_loop_rate = self.get_param("control_loop_rate", 1)

        # Get Unstuck flag and distance for unstuck routine
        self.unstuck_flag_sub: Subscriber = self.new_subscription(
            Bool,
            f"/paf/{self.role_name}/unstuck_flag",
            self.__get_unstuck_flag,
            qos_profile=1,
        )
        self.unstuck_distance_sub: Subscriber = self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/unstuck_distance",
            self.__get_unstuck_distance,
            qos_profile=1,
        )

        # Get current speed
        self.velocity_sub: Subscriber = self.new_subscription(
            CarlaSpeedometer,
            f"/carla/{self.role_name}/Speed",
            self.__get_current_velocity,
            qos_profile=1,
        )

        # Get initial set of speed limits from global planner
        self.speed_limit_OD_sub: Subscriber = self.new_subscription(
            Float32MultiArray,
            f"/paf/{self.role_name}/speed_limits_OpenDrive",
            self.__set_speed_limits_opendrive,
            qos_profile=1,
        )

        # Get trajectory to determine current speed limit
        self.trajectory_sub: Subscriber = self.new_subscription(
            Path,
            f"/paf/{self.role_name}/trajectory_global",
            self.__set_trajectory,
            qos_profile=1,
        )

        # Get current position to determine current waypoint
        self.pose_sub: Subscriber = self.new_subscription(
            msg_type=PoseStamped,
            topic="/paf/" + self.role_name + "/current_pos",
            callback=self.__current_position_callback,
            qos_profile=1,
        )

        # Get approximated speed from obstacle in front
        self.approx_speed_sub = self.new_subscription(
            Float32MultiArray,
            f"/paf/{self.role_name}/collision",
            self.__collision_callback,
            qos_profile=1,
        )

        # Get distance to and velocity of leading vehicle from radar sensor
        self.lead_vehicle_sub = self.new_subscription(
            Float32MultiArray,
            f"/paf/{self.role_name}/Radar/lead_vehicle/range_velocity_array",
            self.__update_radar_data,
            qos_profile=1,
        )

        # Publish desired speed to acting
        self.velocity_pub: Publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/acc_velocity", qos_profile=1
        )

        # Publish current waypoint and speed limit
        self.wp_publisher: Publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/current_wp", qos_profile=1
        )
        self.speed_limit_publisher: Publisher = self.new_publisher(
            Float32, f"/paf/{self.role_name}/speed_limit", qos_profile=1
        )

        # unstuck attributes
        self.__unstuck_flag: bool = False
        self.__unstuck_distance: float = -1

        # List of all speed limits, sorted by waypoint index
        self.__speed_limits_OD: [float] = []
        # Current Trajectory
        self.__trajectory: Path = None
        # Current index from waypoint
        self.__current_wp_index: int = 0
        # Current speed
        self.__current_velocity: float = None
        # Distance and speed from possible collsion object
        self.obstacle_speed: float = None
        # Obstacle distance
        self.obstacle_distance: float = None
        # Current speed limit
        self.speed_limit: float = None  # m/s
        # Radar data
        self.leading_vehicle_distance = None
        self.leading_vehicle_relative_speed = None
        self.leading_vehicle_speed = None

        self.logdebug("ACC initialized")

    def __update_radar_data(self, data: Float32MultiArray):
        if not data.data or len(data.data) < 2:
            # no distance and speed data of the leading vehicle is transferred
            # (leading vehicle is very far away)
            self.leading_vehicle_distance = None
            self.leading_vehicle_relative_speed = None
            self.leading_vehicle_speed = None
        else:
            self.leading_vehicle_distance = data.data[0]
            self.leading_vehicle_relative_speed = data.data[1]
            self.leading_vehicle_speed = (
                self.__current_velocity + self.leading_vehicle_relative_speed
            )

    def __collision_callback(self, data: Float32):
        """Safe approximated speed form obstacle in front together with
        timestamp when recieved.
        Timestamp is needed to check wether we still have a vehicle in front

        Args:
            data (Float32): Speed from obstacle in front
        """
        if np.isinf(data.data[0]):
            # If no obstacle is in front, we reset all values
            self.obstacle_speed = None
            self.obstacle_distance = None
            return
        self.obstacle_speed = data.data[1]
        self.obstacle_distance = data.data[0]

    def __get_unstuck_flag(self, data: Bool):
        """Set unstuck flag

        Args:
            data (Bool): Unstuck flag
        """
        self.__unstuck_flag = data.data

    def __get_unstuck_distance(self, data: Float32):
        """Set unstuck distance

        Args:
            data (Float32): Unstuck distance
        """
        self.__unstuck_distance = data.data

    def __get_current_velocity(self, data: CarlaSpeedometer):
        """Set current velocity

        Args:
            data (CarlaSpeedometer): Current velocity from carla
        """
        self.__current_velocity = float(data.speed)

    def __set_trajectory(self, data: Path):
        """Recieve trajectory from global planner

        Args:
            data (Path): Trajectory path
        """
        self.__trajectory = data

    def __set_speed_limits_opendrive(self, data: Float32MultiArray):
        """Recieve speed limits from OpenDrive via global planner

        Args:
            data (Float32MultiArray): speed limits per waypoint
        """
        self.__speed_limits_OD = data.data

    def __current_position_callback(self, data: PoseStamped):
        """Get current position and check if next waypoint is reached
            If yes -> update current waypoint and speed limit

        Args:
            data (PoseStamped): Current position from perception
        """
        if len(self.__speed_limits_OD) < 1 or self.__trajectory is None:
            return

        agent = data.pose.position
        # Get current waypoint
        current_wp = self.__trajectory.poses[self.__current_wp_index].pose.position
        # Get next waypoint
        next_wp = self.__trajectory.poses[self.__current_wp_index + 1].pose.position
        # distances from agent to current and next waypoint
        # NOTE: its possible that the "current" waypoint is behind the hero car ?
        d_old = np.sqrt((agent.x - current_wp.x) ** 2 + (agent.y - current_wp.y) ** 2)
        d_new = np.sqrt((agent.x - next_wp.x) ** 2 + (agent.y - next_wp.y) ** 2)
        if d_new < d_old:
            # If distance to next waypoint is smaller than to current
            # update current waypoint and corresponding speed limit
            self.__current_wp_index += 1
            self.wp_publisher.publish(self.__current_wp_index)
            self.speed_limit = self.__speed_limits_OD[self.__current_wp_index]
            self.speed_limit_publisher.publish(self.speed_limit)
        # in case we used the unstuck routine to drive backwards
        # we have to follow WPs that are already passed
        elif self.__unstuck_flag:
            if self.__unstuck_distance is None or self.__unstuck_distance == -1:
                return
            self.__current_wp_index -= int(self.__unstuck_distance)
            self.wp_publisher.publish(self.__current_wp_index)
            self.speed_limit = self.__speed_limits_OD[self.__current_wp_index]
            self.speed_limit_publisher.publish(self.speed_limit)

    def run(self):
        """
        Control loop
        :return:
        """

        def loop(timer_event=None):
            """
            Permanent checks if distance to a possible object is too small and
            publishes the desired speed to motion planning
            """

            if (
                # often none
                self.leading_vehicle_distance is not None
                # often none -> often does elif even if if-case is necessary
                and self.leading_vehicle_speed is not None
                and self.__current_velocity is not None
            ):
                if self.leading_vehicle_speed < 0.0:
                    acc_speed = 0.0
                    self.velocity_pub.publish(acc_speed)

                # If we have obstalce information,
                # we can calculate the safe speed
                safety_distance: float
                safety_distance = calculate_rule_of_thumb(
                    False, self.__current_velocity
                )
                if self.leading_vehicle_distance < safety_distance:
                    # If safety distance is reached, we want to reduce the
                    # speed to meet the desired distance
                    # https://encyclopediaofmath.org/index.php?title=Linear_interpolation
                    safe_speed = self.leading_vehicle_speed * (
                        self.leading_vehicle_distance / safety_distance
                    )
                    # Interpolate speed for smoother braking
                    safe_speed = interpolate_speed(safe_speed, self.__current_velocity)
                    if safe_speed < 1.0:
                        safe_speed = 0
                    self.velocity_pub.publish(safe_speed)
                else:
                    # If safety distance is reached just hold current speed
                    if (
                        self.__current_velocity < 1.0
                        and self.leading_vehicle_distance < 1.0
                    ) or self.leading_vehicle_distance < 6.0:  # Emergency stop
                        safe_speed = 0
                    else:
                        safe_speed = 10.0  # Safe cruising speed
                    self.velocity_pub.publish(safe_speed)

            elif self.speed_limit is not None:
                # If we have no obstacle, we want to drive with the current
                # speed limit
                self.velocity_pub.publish(self.speed_limit)
            else:
                # If we don't have speed limits do not drive
                # probably a problem accured in the global planner
                self.velocity_pub.publish(0)

        self.new_timer(self.control_loop_rate, loop)
        self.spin()


if __name__ == "__main__":
    """
    main function starts the ACC node
    :param args:
    """
    roscomp.init("ACC")

    try:
        node = ACC()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()
