#!/usr/bin/env python
import numpy as np
import ros_compatibility as roscomp
from carla_msgs.msg import CarlaSpeedometer  # , CarlaWorldInfo
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from ros_compatibility.node import CompatibleNode
import rospy
from rospy import Publisher, Subscriber
from std_msgs.msg import Bool, Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from typing import Optional
from typing import List
from typing import Tuple
from planning.cfg import ACCConfig
from dynamic_reconfigure.server import Server

import utils
import shapely

import mapping_common.map
import mapping_common.mask
from mapping_common.map import Map
from mapping_common.entity import Entity, FlagFilter
from mapping_common.shape import Polygon
from mapping.msg import Map as MapMsg

MARKER_NAMESPACE: str = "acc"


class ACC(CompatibleNode):
    """ACC (Adaptive Cruise Control) calculates and publishes the desired speed based on
    possible collisions, the current speed, the trajectory, and the speed limits."""

    map: Optional[Map] = None
    last_map_timestamp: Optional[rospy.Time] = None

    # unstuck attributes
    __unstuck_flag: bool = False
    __unstuck_distance: float = -1
    # List of all speed limits, sorted by waypoint index
    __speed_limits_OD: List[float] = []
    trajectory_global: Optional[Path] = None
    trajectory: Optional[Path] = None
    __current_position: Optional[Point] = None
    # Current index from waypoint
    __current_wp_index: int = 0
    __current_heading: Optional[float] = None
    speed_limit: Optional[float] = None  # m/s

    def __init__(self):
        super(ACC, self).__init__("ACC")
        self.role_name = self.get_param("role_name", "hero")

        # Map
        # # Current speed
        # self.__current_velocity: Optional[float] = None
        # Distance and speed from possible collsion object
        # self.obstacle_speed: Optional[float] = None
        # # Obstacle distance
        # self.obstacle_distance: Optional[float] = None
        # Current speed limit
        # Radar data
        # self.leading_vehicle_distance = None
        # self.leading_vehicle_relative_speed = None
        # self.leading_vehicle_speed = None

        # Get Map
        self.map_sub: Subscriber = self.new_subscription(
            MapMsg,
            f"/paf/{self.role_name}/mapping/init_data",
            self.__get_map,
            qos_profile=1,
        )

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
        # self.velocity_sub: Subscriber = self.new_subscription(
        #     CarlaSpeedometer,
        #     f"/carla/{self.role_name}/Speed",
        #     self.__get_current_velocity,
        #     qos_profile=1,
        # )

        # Get initial set of speed limits from global planner
        self.speed_limit_OD_sub: Subscriber = self.new_subscription(
            Float32MultiArray,
            f"/paf/{self.role_name}/speed_limits_OpenDrive",
            self.__set_speed_limits_opendrive,
            qos_profile=1,
        )

        # Get global trajectory to determine current speed limit
        self.trajectory_global_sub: Subscriber = self.new_subscription(
            Path,
            f"/paf/{self.role_name}/trajectory_global",
            self.__set_trajectory_global,
            qos_profile=1,
        )

        # Get trajectory to determine current speed limit
        self.trajectory_sub: Subscriber = self.new_subscription(
            Path,
            f"/paf/{self.role_name}/trajectory",
            self.__set_trajectory,
            qos_profile=1,
        )

        # Get current position to determine current waypoint
        self.pose_sub: Subscriber = self.new_subscription(
            msg_type=PoseStamped,
            topic=f"/paf/{self.role_name}/current_pos",
            callback=self.__current_position_callback,
            qos_profile=1,
        )

        # Get current_heading
        self.heading_sub: Subscriber = self.new_subscription(
            Float32,
            f"/paf/{self.role_name}/current_heading",
            self.__get_heading,
            qos_profile=1,
        )

        # Get approximated speed from obstacle in front
        # self.approx_speed_sub = self.new_subscription(
        #     Float32MultiArray,
        #     f"/paf/{self.role_name}/collision",
        #     self.__collision_callback,
        #     qos_profile=1,
        # )

        # Get distance to and velocity of leading vehicle from radar sensor
        # self.lead_vehicle_sub = self.new_subscription(
        #     Float32MultiArray,
        #     f"/paf/{self.role_name}/Radar/lead_vehicle/range_velocity_array",
        #     self.__update_radar_data,
        #     qos_profile=1,
        # )

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

        # Publish debugging marker
        self.marker_publisher: Publisher = self.new_publisher(
            MarkerArray, f"/paf/{self.role_name}/acc/debug_markers", qos_profile=1
        )

        # Tunable values for the controllers
        self.sg_Ki: float
        self.sg_T_gap: float
        self.sg_d_min: float
        self.ct_Kp: float
        self.ct_Ki: float
        self.ct_T_gap: float
        self.ct_d_min: float
        Server(ACCConfig, self.dynamic_reconfigure_callback)

        self.logdebug("ACC initialized")

    def __get_map(self, data: MapMsg):
        if self.map is not None:
            self.last_map_timestamp = self.map.timestamp
        self.map = Map.from_ros_msg(data)
        self.update_velocity()

    def dynamic_reconfigure_callback(self, config: "ACCConfig", level):
        self.sg_Ki = config["sg_Ki"]
        self.sg_T_gap = config["sg_T_gap"]
        self.sg_d_min = config["sg_d_min"]
        self.ct_Kp = config["ct_Kp"]
        self.ct_Ki = config["ct_Ki"]
        self.ct_T_gap = config["ct_T_gap"]
        self.ct_d_min = config["ct_d_min"]

        return config

    # def __update_radar_data(self, data: Float32MultiArray):
    #     if not data.data or len(data.data) < 2:
    #         # no distance and speed data of the leading vehicle is transferred
    #         # (leading vehicle is very far away)
    #         self.leading_vehicle_distance = None
    #         self.leading_vehicle_relative_speed = None
    #         self.leading_vehicle_speed = None
    #     else:
    #         self.leading_vehicle_distance = data.data[0]
    #         self.leading_vehicle_relative_speed = data.data[1]
    #         self.leading_vehicle_speed = (
    #             self.__current_velocity + self.leading_vehicle_relative_speed
    #         )

    # def __collision_callback(self, data: Float32):
    #     """Safe approximated speed form obstacle in front together with
    #     timestamp when recieved.
    #     Timestamp is needed to check wether we still have a vehicle in front

    #     Args:
    #         data (Float32): Speed from obstacle in front
    #     """
    #     if np.isinf(data.data[0]):
    #         # If no obstacle is in front, we reset all values
    #         self.obstacle_speed = None
    #         self.obstacle_distance = None
    #         return
    #     self.obstacle_speed = data.data[1]
    #     self.obstacle_distance = data.data[0]

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

    # def __get_current_velocity(self, data: CarlaSpeedometer):
    #     """Set current velocity

    #     Args:
    #         data (CarlaSpeedometer): Current velocity from carla
    #     """
    #     self.__current_velocity = float(data.speed)

    def __get_heading(self, data: Float32):
        """Recieve current heading

        Args:
            data (Float32): Current heading
        """
        self.__current_heading = float(data.data)

    def __set_trajectory_global(self, data: Path):
        """Recieve trajectory from global planner

        Args:
            data (Path): Trajectory path
        """
        self.trajectory_global = data

    def __set_trajectory(self, data: Path):
        """Recieve trajectory from motion planner

        Args:
            data (Path): Trajectory path
        """
        self.trajectory = data

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
        self.__current_position = data.pose.position
        if len(self.__speed_limits_OD) < 1 or self.trajectory_global is None:
            return

        agent = self.__current_position
        # Get current waypoint
        current_wp = self.trajectory_global.poses[self.__current_wp_index].pose.position
        # Get next waypoint
        next_wp = self.trajectory_global.poses[
            self.__current_wp_index + 1
        ].pose.position
        # distances from agent to current and next waypoint
        d_old = abs(agent.x - current_wp.x) + abs(agent.y - current_wp.y)
        d_new = abs(agent.x - next_wp.x) + abs(agent.y - next_wp.y)
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

    def publish_debug_markers(self, m: List[Marker]):
        marker_array = MarkerArray(
            markers=[Marker(ns=MARKER_NAMESPACE, action=Marker.DELETEALL)]
        )
        for id, marker in enumerate(m):
            marker.header.frame_id = "hero"
            marker.header.stamp = rospy.get_rostime()
            marker.ns = MARKER_NAMESPACE
            marker.id = id
            marker.lifetime = rospy.Duration.from_sec(0.5)
            marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def run(self):
        """
        Control loop
        :return:
        """

        # Parameters for the PI controller
        # Kp = self.ct_Kp  # 0.5
        # Ki = self.ct_Ki  # 1.5
        # T_gap = self.ct_T_gap  # 1.9
        # d_min = self.ct_d_min  # 1

        # Parameters for the stop and go system
        # Ki_sg = 1.5
        # T_gap_sg = 1.9  # unit: seconds
        # d_min_sg = 3
        self.spin()

    def update_velocity(self):
        """
        Permanent checks if distance to a possible object is too small and
        publishes the desired speed to motion planning
        """
        if (
            self.map is None
            or self.trajectory is None
            or self.__current_position is None
            or self.__current_heading is None
        ):
            # We don't have the necessary data to drive safely
            self.velocity_pub.publish(0)
            return

        hero = self.map.hero()
        if hero is None or hero.motion is None:
            # We currenly have no hero data.
            # -> cannot drive safely
            rospy.logerr("ACC: No hero with motion found in map!")
            self.velocity_pub.publish(0)
            return
        hero_width = max(1.0, hero.get_width())

        tree = self.map.build_tree(FlagFilter(is_collider=True, is_hero=False))
        hero_transform = mapping_common.map.build_global_hero_transform(
            self.__current_position.x,
            self.__current_position.y,
            self.__current_heading,
        )

        front_mask_size = 7.5
        trajectory_mask = mapping_common.mask.build_trajectory_shape(
            self.trajectory,
            hero_transform,
            start_dist_from_hero=front_mask_size,
            max_length=100.0,
            current_wp_idx=self.__current_wp_index,
            max_wp_count=200,
            centered=True,
            width=hero_width,
        )
        if trajectory_mask is None:
            # We currently have no valid path to check for collisions.
            # -> cannot drive safely
            rospy.logerr("ACC: Unable to build collision mask!")
            self.velocity_pub.publish(0)
            return

        # Add small area in front of car to the collision mask
        front_rect = mapping_common.mask.project_plane(
            front_mask_size, size_y=hero_width
        )
        collision_masks = [front_rect, trajectory_mask]
        # collision_mask = shapely.GeometryCollection(collision_masks)
        collision_mask = shapely.union_all(collision_masks)

        marker_list = []
        for mask in collision_masks:
            mask_marker = Polygon.from_shapely(mask).to_marker()
            mask_marker.scale.z = 0.2
            mask_marker.color.a = 0.5
            mask_marker.color.r = 0
            mask_marker.color.g = 1.0
            mask_marker.color.b = 1.0
            marker_list.append(mask_marker)

        entity_result = tree.get_nearest_entity(collision_mask, hero.to_shapely())

        current_velocity = hero.get_global_x_velocity() or 0.0
        desired_speed: float = float("inf")
        if entity_result is not None:
            entity, distance = entity_result
            entity_marker = entity.entity.to_marker()
            entity_marker.scale.z = 0.2
            entity_marker.color.a = 0.5
            entity_marker.color.r = 1.0
            entity_marker.color.g = 0.0
            entity_marker.color.b = 0.0
            marker_list.append(entity_marker)

            lead_delta_velocity = (
                hero.get_delta_forward_velocity_of(entity.entity) or -current_velocity
            )
            desired_speed = self.calculate_velocity_based_on_lead(
                current_velocity, distance, lead_delta_velocity
            )

            marker_text = (
                f"LeadDistance: {distance}\n"
                + f"LeadXVelocity: {entity.entity.get_global_x_velocity()}\n"
                + f"DeltaV: {lead_delta_velocity}\n"
                + f"RawACCSpeed: {desired_speed}"
            )
            text_marker = Marker(type=Marker.TEXT_VIEW_FACING, text=marker_text)
            text_marker.pose.position.x = -2.0
            text_marker.pose.position.y = 0.0
            text_marker.scale.z = 0.3
            text_marker.color.a = 1.0
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            marker_list.append(text_marker)

        self.velocity_pub.publish(desired_speed)
        self.publish_debug_markers(marker_list)

    def calculate_velocity_based_on_lead(
        self, hero_velocity: float, lead_distance: float, delta_v: float
    ) -> float:
        desired_speed: float = float("inf")
        if (
            hero_velocity < 2 and lead_distance < 2
        ):  # stop and go system for velocities between 0 m/s and 2 m/s
            # should use the P-controller below as soon as we get reasonable
            # radar data
            desired_speed = (lead_distance - 0.5) / 4

        else:  # system for velocities > 3 m/s  = 10.8 km/h
            Kp = self.ct_Kp
            Ki = self.ct_Ki
            T_gap = self.ct_T_gap
            d_min = self.ct_d_min

            desired_distance = d_min + T_gap * hero_velocity
            delta_d = lead_distance - desired_distance
            speed_adjustment = Ki * delta_d + Kp * delta_v
            desired_speed = hero_velocity + speed_adjustment

            desired_speed = max(desired_speed, 0.0)

            if self.speed_limit is None:
                desired_speed = min(5.0, desired_speed)
            else:
                # Max speed is the current speed limit
                desired_speed = min(self.speed_limit, desired_speed)

        return desired_speed


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
