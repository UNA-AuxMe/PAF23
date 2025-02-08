import py_trees
import sys
import os
from std_msgs.msg import String, Float32

import rospy
import numpy as np

from behaviors import behavior_speed as bs
import mapping_common.map
import mapping_common.mask
import mapping_common.entity
from mapping_common.map import Map
import shapely
from mapping_common.entity import FlagFilter
from visualization_msgs.msg import Marker, MarkerArray
from behaviors import behavior_utils

sys.path.append(os.path.abspath(sys.path[0] + "/.."))
# from behavior_utils import get_hero_width, get_marker_arr_in_front
from local_planner.utils import (  # type: ignore # noqa: E402
    NUM_WAYPOINTS,
    TARGET_DISTANCE_TO_STOP_OVERTAKE,
)  # type: ignore # noqa: E402

"""
Source: https://github.com/ll7/psaf2
"""


# Variable to determine the distance to overtake the object
OVERTAKE_EXECUTING = 0
OVERTAKE_FREE = False


class Ahead(py_trees.behaviour.Behaviour):
    """
    This behaviour checks whether an object that needs to be overtaken is
    ahead
    """

    def __init__(self, name):
        """
        Minimal one-time initialisation. A good rule of thumb is to only
        include the initialisation relevant for being able to insert this
        behaviour in a tree for offline rendering to dot graphs.

         :param name: name of the behaviour
        """
        super(Ahead, self).__init__(name)

    def setup(self, timeout):
        """
        Delayed one-time initialisation that would otherwise interfere with
        offline rendering of this behaviour in a tree to dot graph or
        validation of the behaviour's configuration.

        This initializes the blackboard to be able to access data written to it
        by the ROS topics.
        :param timeout: an initial timeout to see if the tree generation is
        successful
        :return: True, as the set up is successful.
        """
        self.blackboard = py_trees.blackboard.Blackboard()
        self.ot_distance_pub = rospy.Publisher(
            "/paf/hero/" "overtake_distance", Float32, queue_size=1
        )
        self.ot_marker_pub = rospy.Publisher(
            "/paf/hero/" "overtake_marker", Marker, queue_size=1
        )
        self.marker_publisher = rospy.Publisher(
            "/paf/hero/" "overtake/debug_markers", MarkerArray, queue_size=1
        )
        return True

    def initialise(self):
        """
        When is this called?
            The first time your behaviour is ticked and anytime the status is
            not RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.
        """
        # Counter for detecting overtake situation
        self.counter_overtake = 0
        self.old_obstacle_distance = 200
        return True

    def update(self):
        """
        When is this called?
        Every time your behaviour is ticked.
        What to do here?
            - Triggering, checking, monitoring. Anything...but do not block!
            - Set a feedback message
            - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]

        Gets the current distance and speed to object in front.
        Increases a counter to overtake if there is a obstacle.
        :return: py_trees.common.Status.SUCCESS, if the counter crosses a
                 certain threshold.
                 py_trees.common.Status.FAILURE, if there is nothing
                 to overtake.
        """

        map_data = self.blackboard.get("/paf/hero/mapping/init_data")

        if map_data is not None:
            map = Map.from_ros_msg(map_data)
        else:
            rospy.logerr("Map data not avilable in Overtake")
            return py_trees.common.Status.FAILURE

        # data preparation
        trajectory = self.blackboard.get("/paf/hero/trajectory")
        current_wp = self.blackboard.get("/paf/hero/current_wp")
        hero_pos = self.blackboard.get("/paf/hero/current_pos")
        hero_heading = self.blackboard.get("/paf/hero/current_heading")
        if (
            trajectory is not None
            and current_wp is not None
            and hero_pos is not None
            and hero_heading is not None
        ):
            tree = map.build_tree(FlagFilter(is_collider=True, is_hero=False))
            hero_transform = mapping_common.map.build_global_hero_transform(
                hero_pos.pose.position.x,
                hero_pos.pose.position.y,
                hero_heading.data,
            )
            hero_width = behavior_utils.get_hero_width(map)
            front_mask_size = 0.0
            trajectory_mask = mapping_common.mask.build_trajectory_shape(
                trajectory,
                hero_transform,
                start_dist_from_hero=front_mask_size,
                max_length=18.0,
                current_wp_idx=int(current_wp.data),
                max_wp_count=50,
                centered=True,
                width=hero_width,
            )
            if trajectory_mask is None:
                # We currently have no valid path to check for collisions.
                # -> cannot drive safely
                rospy.logerr("Overtake ahead: Unable to build collision mask!")
                return py_trees.common.Status.FAILURE
            collision_masks = [trajectory_mask]

            entity_result = tree.get_nearest_entity(
                trajectory_mask, map.hero().to_shapely(), 0.35
            )
        else:
            rospy.loginfo(
                f"Something is none in overtake ahead:"
                f"{trajectory is None} {current_wp is None},"
                f"{hero_pos is None} {hero_heading is None}"
            )
            return py_trees.common.Status.FAILURE

        if entity_result is not None:
            entity, distance = entity_result
            entity = entity.entity
            obstacle_distance = distance
            if entity.motion is not None:
                obstacle_speed = entity.motion.linear_motion.length()
            else:
                obstacle_speed = 0
            marker_arr = behavior_utils.get_marker_arr_in_front(
                entity, obstacle_distance, map.hero(), collision_masks
            )
            self.marker_publisher.publish(marker_arr)
        else:
            return py_trees.common.Status.FAILURE

        # filter out false positives due to trajectory inconsistency
        if (
            entity.transform.translation().y() < -3.0
            or entity.transform.translation().y() > 3.0
        ):
            return py_trees.common.Status.FAILURE

        # generate visualization marker for obstacle
        ot_marker = entity.to_marker()
        ot_marker.color.r = 1.0
        ot_marker.color.g = 0
        ot_marker.color.b = 0
        ot_marker.header.frame_id = "hero"
        ot_marker.header.stamp = rospy.Time.now()
        ot_marker.lifetime = rospy.Duration(0.3)
        self.ot_marker_pub.publish(ot_marker)

        # increase counter when something is blocking the path
        if obstacle_speed < 2.5 and obstacle_distance < 15:
            self.counter_overtake += 1
            rospy.loginfo(f"Obstacle distance: {obstacle_distance}")
            rospy.loginfo("Overtake counter: " + str(self.counter_overtake))
            if self.counter_overtake > 4:
                self.ot_distance_pub.publish(obstacle_distance)
                return py_trees.common.Status.SUCCESS
            self.old_obstacle_distance = obstacle_distance
            return py_trees.common.Status.RUNNING
        else:
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """
        When is this called?
        Whenever your behaviour switches to a non-running state.
            - SUCCESS || FAILURE : your behaviour's work cycle has finished
            - INVALID : a higher priority branch has interrupted, or shutting
            down
        writes a status message to the console when the behaviour terminates
        :param new_status: new state after this one is terminated
        """
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )


class Approach(py_trees.behaviour.Behaviour):
    """
    This behaviour is executed when the ego vehicle is in close proximity of
    an object which needs to be overtaken and
    overtake_ahead is triggered.
    It then handles the procedure for overtaking.
    """

    def __init__(self, name):
        """
        Minimal one-time initialisation. Other one-time initialisation
        requirements should be met via the setup() method.
         :param name: name of the behaviour
        """
        super(Approach, self).__init__(name)
        rospy.loginfo("Init -> Overtake Behavior: Approach")

    def setup(self, timeout):
        """
        Delayed one-time initialisation that would otherwise interfere with
        offline rendering of this behaviour in a tree to dot graph or
        validation of the behaviour's configuration.

        This initializes the blackboard to be able to access data written to it
        by the ROS topics and the current behavior publisher.
        :param timeout: an initial timeout to see if the tree generation is
        successful
        :return: True, as the set up is successful.
        """
        self.curr_behavior_pub = rospy.Publisher(
            "/paf/hero/" "curr_behavior", String, queue_size=1
        )
        self.ot_distance_pub = rospy.Publisher(
            "/paf/hero/" "overtake_distance", Float32, queue_size=1
        )
        self.blackboard = py_trees.blackboard.Blackboard()
        self.ot_marker_pub = rospy.Publisher(
            "/paf/hero/" "overtake_marker", Marker, queue_size=1
        )
        self.marker_publisher = rospy.Publisher(
            "/paf/hero/" "overtake/debug_markers", MarkerArray, queue_size=1
        )
        return True

    def initialise(self):
        """
        When is this called?
        The first time your behaviour is ticked and anytime the status is not
        RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.

        This initializes the overtaking distance to a default value.
        """
        rospy.loginfo("Approaching Overtake")
        global OVERTAKE_FREE
        self.ot_distance = 30
        self.ot_counter = 0
        self.clear_distance = 35
        OVERTAKE_FREE = False

    def update(self):
        """
        When is this called?
        Every time your behaviour is ticked.
        What to do here?
            - Triggering, checking, monitoring. Anything...but do not block!
            - Set a feedback message
            - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]

        Gets the current distance to overtake, the current oncoming lane status and the
        distance to collsion object. Slows down until stopped
        or oncoming clear.
        :return: py_trees.common.Status.RUNNING, while driving towards the obstacle
                 py_trees.common.Status.SUCCESS, if stopped behind the blocking
                 object or oncoming is free.
                 py_trees.common.Status.FAILURE, if the overtake is aborted
        """
        global OVERTAKE_EXECUTING
        global OVERTAKE_FREE

        # Intermediate layer map integration
        map_data = self.blackboard.get("/paf/hero/mapping/init_data")
        if map_data is not None:
            map = Map.from_ros_msg(map_data)
        else:
            rospy.logerr("Map data not available in Overtake")
            return py_trees.common.Status.FAILURE

        # data preparation
        trajectory = self.blackboard.get("/paf/hero/trajectory")
        current_wp = self.blackboard.get("/paf/hero/current_wp")
        hero_pos = self.blackboard.get("/paf/hero/current_pos")
        hero_heading = self.blackboard.get("/paf/hero/current_heading")
        if (
            trajectory is not None
            and current_wp is not None
            and hero_pos is not None
            and hero_heading is not None
        ):
            tree = map.build_tree(FlagFilter(is_collider=True, is_hero=False))
            hero_transform = mapping_common.map.build_global_hero_transform(
                hero_pos.pose.position.x,
                hero_pos.pose.position.y,
                hero_heading.data,
            )
            hero_width = behavior_utils.get_hero_width(map)
            front_mask_size = 1.5
            trajectory_mask = mapping_common.mask.build_trajectory_shape(
                trajectory,
                hero_transform,
                start_dist_from_hero=front_mask_size,
                max_length=20.0,
                current_wp_idx=int(current_wp.data),
                max_wp_count=50,
                centered=True,
                width=hero_width,
            )
            if trajectory_mask is None:
                # We currently have no valid path to check for collisions.
                # -> cannot drive safely
                rospy.logerr("Overtake ahead: Unable to build collision mask!")
                return py_trees.common.Status.FAILURE

            front_rect = mapping_common.mask.project_plane(
                front_mask_size, size_y=hero_width
            )
            collision_masks = [front_rect, trajectory_mask]
            collision_mask = shapely.union_all(collision_masks)

            entity_result = tree.get_nearest_entity(
                collision_mask, map.hero().to_shapely(), 0.5
            )
        else:
            rospy.loginfo(
                f"Something is none in overtake ahead:"
                f"{trajectory is None} {current_wp is None},"
                f"{hero_pos is None} {hero_heading is None}"
            )
            return py_trees.common.Status.FAILURE

        if entity_result is not None:
            entity, distance = entity_result
            entity = entity.entity
            self.ot_distance = distance
            rospy.loginfo(f"Overtake distance: {self.ot_distance}")
            OVERTAKE_EXECUTING = self.ot_distance
            if entity.motion is not None:
                obstacle_speed = entity.motion.linear_motion.length()
                if obstacle_speed > 3.0:
                    rospy.loginfo("Overtake entity started moving, abort")
                    return py_trees.common.Status.FAILURE
            else:
                obstacle_speed = 0

            marker_arr = behavior_utils.get_marker_arr_in_front(
                entity, distance, map.hero(), collision_masks
            )
            self.marker_publisher.publish(marker_arr)
        else:
            return py_trees.common.Status.FAILURE

        # generate visualization marker for obstacle
        ot_marker = entity.to_marker()
        ot_marker.color.r = 1.0
        ot_marker.color.g = 0
        ot_marker.color.b = 0
        ot_marker.header.frame_id = "hero"
        ot_marker.header.stamp = rospy.Time.now()
        ot_marker.lifetime = rospy.Duration(0.3)
        self.ot_marker_pub.publish(ot_marker)

        # slow down before overtake if blocked
        if self.ot_distance < 15.0:
            rospy.loginfo(f"Overtake Approach Distance: {self.ot_distance}")
            ot_free = tree.is_lane_free(False, self.clear_distance, 15)
            rospy.loginfo(f"Overtake is free: {ot_free}")
            if ot_free:
                self.ot_counter += 1
                # using a counter to account for inconsistencies
                if self.ot_counter > 3:
                    rospy.loginfo("Overtake is free not slowing down!")
                    self.ot_distance_pub.publish(self.ot_distance)
                    self.curr_behavior_pub.publish(bs.ot_app_free.name)
                    # bool to skip Wait since oncoming is free
                    OVERTAKE_FREE = True
                    return py_trees.common.Status.SUCCESS
                else:
                    self.ot_distance_pub.publish(self.ot_distance)
                    self.curr_behavior_pub.publish(bs.ot_app_blocked.name)
                    return py_trees.common.Status.RUNNING
            else:
                self.ot_counter = 0
                self.ot_distance_pub.publish(self.ot_distance)
                rospy.loginfo("Overtake Approach: oncoming blocked slowing down")
                self.curr_behavior_pub.publish(bs.ot_app_blocked.name)
        # obstacle is not relevant anymore
        elif self.ot_distance > 20.0:
            return py_trees.common.Status.FAILURE

        if self.ot_distance > 15.0:
            # too far
            rospy.loginfo(
                f"Overtake Approach: still approaching obstacle, "
                f"distance: {self.ot_distance}"
            )
            return py_trees.common.Status.RUNNING
        elif self.ot_distance < TARGET_DISTANCE_TO_STOP_OVERTAKE:
            # stopping
            rospy.loginfo("Overtake Approach: stopping behind obstacle")
            self.ot_distance_pub.publish(self.ot_distance)
            self.curr_behavior_pub.publish(bs.ot_app_blocked.name)
            return py_trees.common.Status.SUCCESS
        else:
            # still approaching
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        When is this called?
        Whenever your behaviour switches to a non-running state.
            - SUCCESS || FAILURE : your behaviour's work cycle has finished
            - INVALID : a higher priority branch has interrupted, or shutting
            down
        writes a status message to the console when the behaviour terminates
        :param new_status: new state after this one is terminated
        """
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )


class Wait(py_trees.behaviour.Behaviour):
    """
    This behavior handles the waiting in front of object,
    which is blocking the road.
    The Ego vehicle is waiting to get a clear path for overtaking.
    """

    def __init__(self, name):
        """
        Minimal one-time initialisation. Other one-time initialisation
        requirements should be met via the setup() method.
         :param name: name of the behaviour
        """
        super(Wait, self).__init__(name)

    def setup(self, timeout):
        """
        Delayed one-time initialisation that would otherwise interfere with
        offline rendering of this behaviour in a tree to dot graph or
        validation of the behaviour's configuration.

        This initializes the blackboard to be able to access data written to it
        by the ROS topics and the current behavior publisher.
        :param timeout: an initial timeout to see if the tree generation is
        successful
        :return: True, as the set up is successful.
        """
        self.curr_behavior_pub = rospy.Publisher(
            "/paf/hero/" "curr_behavior", String, queue_size=1
        )
        self.ot_distance_pub = rospy.Publisher(
            "/paf/hero/" "overtake_distance", Float32, queue_size=1
        )
        self.marker_publisher = rospy.Publisher(
            "/paf/hero/" "overtake/debug_markers", MarkerArray, queue_size=1
        )
        self.blackboard = py_trees.blackboard.Blackboard()
        return True

    def initialise(self):
        """
        When is this called?
            The first time your behaviour is ticked and anytime the status is
            not RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.
        This just prints a state status message.
        """
        rospy.loginfo("Waiting for Overtake")
        # slightly less distance since we have already stopped
        self.clear_distance = 35
        self.ot_counter = 0
        self.ot_gone = 0
        return True

    def update(self):
        """
        When is this called?
            Every time your behaviour is ticked.
        What to do here?
           - Triggering, checking, monitoring. Anything...but do not block!
           - Set a feedback message
           - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]

        Waits behind the road object until map function lane free check
        return True.

        :return: py_trees.common.Status.RUNNING, while is lane free returns False
                 py_trees.common.Status.SUCCESS, when lane free returns True
        """
        global OVERTAKE_FREE
        global OVERTAKE_EXECUTING
        if OVERTAKE_FREE:
            rospy.loginfo("Overtake is free!")
            self.curr_behavior_pub.publish(bs.ot_wait_free.name)
            return py_trees.common.Status.SUCCESS
        map_data = self.blackboard.get("/paf/hero/mapping/init_data")
        if map_data is not None:
            map = Map.from_ros_msg(map_data)
        else:
            rospy.logerr("Map data not available in Overtake")
            return py_trees.common.Status.FAILURE

        # data preparation
        trajectory = self.blackboard.get("/paf/hero/trajectory")
        current_wp = self.blackboard.get("/paf/hero/current_wp")
        hero_pos = self.blackboard.get("/paf/hero/current_pos")
        hero_heading = self.blackboard.get("/paf/hero/current_heading")
        if (
            trajectory is not None
            and current_wp is not None
            and hero_pos is not None
            and hero_heading is not None
        ):
            tree = map.build_tree(FlagFilter(is_collider=True, is_hero=False))
            hero_transform = mapping_common.map.build_global_hero_transform(
                hero_pos.pose.position.x,
                hero_pos.pose.position.y,
                hero_heading.data,
            )
            hero_width = behavior_utils.get_hero_width(map)
            front_mask_size = 1.0
            trajectory_mask = mapping_common.mask.build_trajectory_shape(
                trajectory,
                hero_transform,
                start_dist_from_hero=front_mask_size,
                max_length=20.0,
                current_wp_idx=int(current_wp.data),
                max_wp_count=50,
                centered=True,
                width=hero_width,
            )
            if trajectory_mask is None:
                # We currently have no valid path to check for collisions.
                # -> cannot drive safely
                rospy.logerr("Overtake ahead: Unable to build collision mask!")
                return py_trees.common.Status.FAILURE

            front_rect = mapping_common.mask.project_plane(
                front_mask_size, size_y=hero_width
            )
            collision_masks = [front_rect, trajectory_mask]
            collision_mask = shapely.union_all(collision_masks)

            entity_result = tree.get_nearest_entity(
                collision_mask, map.hero().to_shapely(), 0.5
            )
        else:
            rospy.loginfo(
                f"Something is none in overtake ahead:"
                f"{trajectory is None} {current_wp is None},"
                f"{hero_pos is None} {hero_heading is None}"
            )
            return py_trees.common.Status.FAILURE

        if entity_result is not None:
            self.ot_gone = 0
            entity, distance = entity_result
            entity = entity.entity
            OVERTAKE_EXECUTING = distance
            if entity.motion is not None:
                obstacle_speed = entity.motion.linear_motion.length()
                rospy.loginfo(f"Obstacle speed: {obstacle_speed}")
                if obstacle_speed > 3.0:
                    rospy.loginfo("Overtake entity started moving, abort")
                    return py_trees.common.Status.FAILURE
            else:
                obstacle_speed = 0
            marker_arr = behavior_utils.get_marker_arr_in_front(
                entity, distance, map.hero(), collision_masks
            )
            self.marker_publisher.publish(marker_arr)
        else:
            # using a counter to account for data inconsistencies
            self.ot_gone += 1
            if self.ot_gone > 3:
                rospy.loginfo("Overtake osbtacle is gone, abort")
                return py_trees.common.Status.FAILURE
            return py_trees.common.Status.RUNNING
        # bicycle handling
        if obstacle_speed > 1.7 and obstacle_speed < 2.4:
            ot_free = tree.is_lane_free(False, 15.0, -4.0)
        else:
            ot_free = tree.is_lane_free(False, self.clear_distance, 15.0)
        if ot_free:
            self.ot_counter += 1
            if self.ot_counter > 3:
                rospy.loginfo("Overtake is free!")
                self.curr_behavior_pub.publish(bs.ot_wait_free.name)
                return py_trees.common.Status.SUCCESS
            else:
                self.curr_behavior_pub.publish(bs.ot_wait_stopped.name)
                return py_trees.common.Status.RUNNING
        else:
            rospy.loginfo("Overtake still blocked")
            self.curr_behavior_pub.publish(bs.ot_wait_stopped.name)
            self.ot_counter = 0
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        When is this called?
            Whenever your behaviour switches to a non-running state.
           - SUCCESS || FAILURE : your behaviour's work cycle has finished
           - INVALID : a higher priority branch has interrupted, or shutting
           down
        writes a status message to the console when the behaviour terminates
        :param new_status: new state after this one is terminated
        """
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )


class Enter(py_trees.behaviour.Behaviour):
    """
    This behavior handles the switching to a new lane in the
    overtaking procedure.
    """

    def __init__(self, name):
        """
        Minimal one-time initialisation. Other one-time initialisation
        requirements should be met via the setup() method.
        :param name: name of the behaviour
        """
        super(Enter, self).__init__(name)

    def setup(self, timeout):
        """
        Delayed one-time initialisation that would otherwise interfere with
        offline rendering of this behaviour in a tree to dot graph or
        validation of the behaviour's configuration.

        This initializes the blackboard to be able to access data written to it
        by the ROS topics and the current behavior publisher.
        :param timeout: an initial timeout to see if the tree generation is
        successful
        :return: True, as the set up is successful.
        """
        self.curr_behavior_pub = rospy.Publisher(
            "/paf/hero/" "curr_behavior", String, queue_size=1
        )
        self.blackboard = py_trees.blackboard.Blackboard()
        return True

    def initialise(self):
        """
        When is this called?
            The first time your behaviour is ticked and anytime the status is
            not RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.

        This prints a state status message and publishes the behavior to
        trigger the replanning
        """
        rospy.loginfo("Enter Overtake")
        self.curr_behavior_pub.publish(bs.ot_enter_init.name)

    def update(self):
        """
        When is this called?
            Every time your behaviour is ticked.
        What to do here?
           - Triggering, checking, monitoring. Anything...but do not block!
           - Set a feedback message
           - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]

        Waits for motion_planner to finish the new trajectory.
        :return: py_trees.common.Status.RUNNING,
                 py_trees.common.Status.SUCCESS,
                 py_trees.common.Status.FAILURE,
        """
        status = self.blackboard.get("/paf/hero/overtake_success")
        if status is not None:
            if status.data == 1:
                rospy.loginfo("Overtake Enter: Trajectory planned")
                return py_trees.common.Status.SUCCESS
            elif status.data == 0:
                self.curr_behavior_pub.publish(bs.ot_enter_slow.name)
                rospy.loginfo("Overtake Enter: Slowing down")
                return py_trees.common.Status.RUNNING
            else:
                rospy.loginfo("Overtake Enter: Abort")
                return py_trees.common.Status.FAILURE
        else:
            rospy.loginfo("Overtake Enter: Waiting for status update")
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        When is this called?
           Whenever your behaviour switches to a non-running state.
          - SUCCESS || FAILURE : your behaviour's work cycle has finished
          - INVALID : a higher priority branch has interrupted, or shutting
          down
        writes a status message to the console when the behaviour terminates
        :param new_status: new state after this one is terminated
        """
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )


class Leave(py_trees.behaviour.Behaviour):
    """
    This behaviour defines the leaf of this subtree, if this behavior is
    reached, the vehicle peformed the overtake.
    """

    def __init__(self, name):
        """
        Minimal one-time initialisation. Other one-time initialisation
        requirements should be met via the setup() method.
        :param name: name of the behaviour
        """
        super(Leave, self).__init__(name)

    def setup(self, timeout):
        """
        Delayed one-time initialisation that would otherwise interfere with
        offline rendering of this behaviour in a tree to dot graph or
        validation of the behaviour's configuration.

        This initializes the blackboard to be able to access data written to it
        by the ROS topics and the current behavior publisher.
        :param timeout: an initial timeout to see if the tree generation is
        successful
        :return: True, as the set up is successful.
        """
        self.curr_behavior_pub = rospy.Publisher(
            "/paf/hero/" "curr_behavior", String, queue_size=1
        )
        self.blackboard = py_trees.blackboard.Blackboard()
        return True

    def initialise(self):
        """
        When is this called?
            The first time your behaviour is ticked and anytime the status is
            not RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.
        This prints a state status message and publishes the behavior
        """
        self.curr_behavior_pub.publish(bs.ot_leave.name)
        data = self.blackboard.get("/paf/hero/current_pos")
        self.first_pos = np.array([data.pose.position.x, data.pose.position.y])
        rospy.loginfo(f"Init Leave Overtake: {self.first_pos}")
        return True

    def update(self):
        """
        When is this called?
            Every time your behaviour is ticked.
        What to do here?
           - Triggering, checking, monitoring. Anything...but do not block!
           - Set a feedback message
           - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]
        Abort this subtree, if overtake distance is big enough
        :return: py_trees.common.Status.FAILURE, to exit this subtree
        """
        global OVERTAKE_EXECUTING
        data = self.blackboard.get("/paf/hero/current_pos")
        self.current_pos = np.array([data.pose.position.x, data.pose.position.y])
        distance = np.linalg.norm(self.first_pos - self.current_pos)
        if distance > OVERTAKE_EXECUTING + NUM_WAYPOINTS:
            rospy.loginfo(f"Overtake executed: {self.current_pos}")
            return py_trees.common.Status.FAILURE
        else:
            return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        """
        When is this called?
           Whenever your behaviour switches to a non-running state.
          - SUCCESS || FAILURE : your behaviour's work cycle has finished
          - INVALID : a higher priority branch has interrupted, or shutting
          down
        writes a status message to the console when the behaviour terminates
        :param new_status: new state after this one is terminated
        """
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
