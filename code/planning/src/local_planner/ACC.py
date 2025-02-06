#!/usr/bin/env python
import ros_compatibility as roscomp
from geometry_msgs.msg import PoseStamped, Point
import mapping_common.shape
from nav_msgs.msg import Path
from ros_compatibility.node import CompatibleNode
import rospy
from rospy import Publisher, Subscriber
from std_msgs.msg import Bool, Float32, Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from typing import Optional
from typing import List
from planning.cfg import ACCConfig
from dynamic_reconfigure.server import Server

import shapely

import mapping_common.map
import mapping_common.mask
import mapping_common.entity
from mapping_common.map import Map
from mapping_common.entity import FlagFilter
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
        collision_mask = shapely.union_all(collision_masks)

        shape_markers = []
        for mask in collision_masks:
            shape_markers.append((Polygon.from_shapely(mask), (0, 1.0, 1.0, 0.5)))

        entity_result = tree.get_nearest_entity(collision_mask, hero.to_shapely())

        text_markers = []
        entity_markers = []
        current_velocity = hero.get_global_x_velocity() or 0.0
        desired_speed: float = float("inf")
        if entity_result is not None:
            entity, distance = entity_result
            entity_markers.append((entity.entity, (1.0, 0.0, 0.0, 0.5)))

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
            text_markers.append((text_marker, (1.0, 1.0, 1.0, 1.0)))

        self.velocity_pub.publish(desired_speed)

        marker_array = mapping_common.entity.shape_debug_marker_array(
            MARKER_NAMESPACE,
            entities=entity_markers,
            shapes=shape_markers,
            markers=text_markers,
        )
        self.marker_publisher.publish(marker_array)

    def calculate_velocity_based_on_lead(
        self, hero_velocity: float, lead_distance: float, delta_v: float
    ) -> float:
        """Calculates the desired speed based on the distance and speed of the leading
           vehicle using a PI controller.

        Args:
            hero_velocity (float): Own velocity (of the hero vehicle)
            lead_distance (float): Distance to the leading vehicle
            delta_v (float): Difference between velocity of the leading vehicle and own
                velocity (negative if the own vehicle is faster than the leading
                vehicle)

        Returns:
            float: Desired speed
        """
        desired_speed: float = float("inf")
        if (
            hero_velocity < 2 and lead_distance < 2
        ):  # approaches the leading vehicle slowly until a distance of 0.5 m
            desired_speed = (lead_distance - 0.5) / 4

        else:
            # PI controller which chooses the desired speed
            Kp = self.ct_Kp
            Ki = self.ct_Ki
            T_gap = self.ct_T_gap
            d_min = self.ct_d_min

            desired_distance = d_min + T_gap * hero_velocity
            delta_d = lead_distance - desired_distance
            speed_adjustment = Ki * delta_d + Kp * delta_v
            desired_speed = hero_velocity + speed_adjustment

            # desired speed should not be negative, only drive forward
            desired_speed = max(desired_speed, 0.0)

            if self.speed_limit is None:
                # if no speed limit is available, drive 5 m/s max
                desired_speed = min(5.0, desired_speed)
            else:
                # max speed is the current speed limit
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
