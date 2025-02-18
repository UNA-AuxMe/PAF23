import py_trees
import numpy as np
from std_msgs.msg import String
from typing import Optional

import rospy
from mapping_common.map import Map
from mapping_common.entity import FlagFilter
from mapping_common.transform import Transform2D, Point2
from mapping_common.markers import debug_marker

from . import behavior_speed as bs
from .debug_markers import add_debug_marker, add_debug_entry, debug_status
from .topics2blackboard import BLACKBOARD_MAP_ID

from local_planner.utils import (
    TARGET_DISTANCE_TO_STOP_INTERSECTION,
)

INTERSECTION_MARKER_COLOR = (222 / 255, 23 / 255, 214 / 255, 1.0)

"""
Source: https://github.com/ll7/psaf2
"""


# 1: Green, 2: Red, 4: Yellow, 0: Unknown
def get_color(state):
    if state == 1:
        return "green"
    elif state == 2:
        return "red"
    elif state == 4:
        return "yellow"
    else:
        return ""


class Ahead(py_trees.behaviour.Behaviour):
    """
    This behaviour checks whether there is an intersection in front of the
    ego vehicle or not and triggers the rest of the decision tree handling the
     intersection.
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
        return True

    def initialise(self):
        """
        When is this called?
            The first time your behaviour is ticked and anytime the status is
            not RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.
        This initializes the variables needed to save information about the
        stop line.
        """
        self.dist = 0
        return True

    def update(self):
        """
        When is this called?
        Every time your behaviour is ticked.
        What to do here?
            - Triggering, checking, monitoring. Anything...but do not block!
            - Set a feedback message
            - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]
        Gets the current distance to the next intersection.
        :return: py_trees.common.Status.SUCCESS, if the vehicle is within range
                    of the intersection
                 py_trees.common.Status.FAILURE, if we are too far away from
                 the intersection
        """

        bb = self.blackboard.get("/paf/hero/waypoint_distance")
        if bb is None:
            return debug_status(
                self.name, py_trees.common.Status.FAILURE, "No waypoint_distance"
            )
        else:
            dist = bb.distance
            isIntersection = bb.isStopLine
            add_debug_entry(self.name, f"Dist: {dist}")
            add_debug_entry(self.name, f"Is intersection: {isIntersection}")
        if dist < 40 and isIntersection:
            return debug_status(self.name, py_trees.common.Status.SUCCESS)
        else:
            return debug_status(self.name, py_trees.common.Status.FAILURE)

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
    an intersection and intersection_ahead is
    triggered. It than handles the approaching the intersection, slowing the
    vehicle down appropriately.
    """

    def __init__(self, name):
        """
        Minimal one-time initialisation. Other one-time initialisation
        requirements should be met via the setup() method.
         :param name: name of the behaviour
        """
        super(Approach, self).__init__(name)
        rospy.loginfo("Approach started")

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
        The first time your behaviour is ticked and anytime the status is not
        RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.
        This initializes the variables needed to save information about the
        stop line, stop signs and the traffic light.
        """
        rospy.loginfo("Approaching Intersection")
        self.stop_sign_detected = False
        self.stop_distance = np.inf

        self.traffic_light_detected = False
        self.traffic_light_distance = np.inf
        self.traffic_light_status = ""

        self.virtual_stopline_distance = np.inf

        self.stopping = True

        self.curr_behavior_pub.publish(bs.int_app_init.name)

    def update(self):
        """
        When is this called?
        Every time your behaviour is ticked.
        What to do here?
            - Triggering, checking, monitoring. Anything...but do not block!
            - Set a feedback message
            - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]
        Gets the current traffic light status, stop sign status
        and the stop line distance. Calcualtes a virtual stop line and
        publishes a distance to it. Slows down car until virtual stop line
        is reached when there is a red traffic light or a stop sign.
        :return: py_trees.common.Status.RUNNING, if too far from intersection
                 py_trees.common.Status.SUCCESS, if stopped in front of inter-
                 section or entered the intersection
                 py_trees.common.Status.FAILURE, if no next path point can be
                 detected.
        """
        # Update Light Info
        light_status_msg = self.blackboard.get("/paf/hero/Center/traffic_light_state")
        light_distance_y_msg = self.blackboard.get(
            "/paf/hero/Center/traffic_light_y_distance"
        )
        if light_status_msg is not None:
            self.traffic_light_status = get_color(light_status_msg.state)
            self.traffic_light_detected = True
            if light_distance_y_msg is not None:
                self.traffic_light_distance = light_distance_y_msg.data

        add_debug_entry(self.name, f"Traffic light: {self.traffic_light_status}")
        add_debug_entry(
            self.name, f"Traffic light detected: {self.traffic_light_detected}"
        )

        # Update stopline Info
        _dis = self.blackboard.get("/paf/hero/waypoint_distance")
        if _dis is not None:
            self.stopline_distance = _dis.distance
            self.stopline_detected = _dis.isStopLine

        add_debug_entry(self.name, f"Stopline dist: {self.stopline_distance}")
        add_debug_entry(self.name, f"Stopline detected: {self.stopline_detected}")

        # Update stop sign Info
        stop_sign_msg = self.blackboard.get("/paf/hero/stop_sign")
        if stop_sign_msg is not None:
            self.stop_sign_detected = stop_sign_msg.isStop
            self.stop_distance = stop_sign_msg.distance

        add_debug_entry(self.name, f"Stopsign dist: {self.stop_distance}")
        add_debug_entry(self.name, f"Stopsign detected: {self.stop_sign_detected}")

        # calculate virtual stopline
        if self.stopline_distance != np.inf and self.stopline_detected:
            self.virtual_stopline_distance = self.stopline_distance
        elif self.stop_sign_detected:
            self.virtual_stopline_distance = self.stop_distance
        else:
            self.virtual_stopline_distance = 0.0
        target_distance = TARGET_DISTANCE_TO_STOP_INTERSECTION
        add_debug_entry(
            self.name, f"Virtual stopline dist: {self.virtual_stopline_distance}"
        )
        # stop when there is no or red/yellow traffic light or a stop sign is
        # detected
        if (
            self.traffic_light_status == "red"
            or self.traffic_light_status == "yellow"
            or (self.stop_sign_detected and not self.traffic_light_detected)
            or self.traffic_light_status == ""
        ):
            self.stopping = True
            # still far
            if self.virtual_stopline_distance > 17.0:
                self.curr_behavior_pub.publish(bs.int_app_init.name)
            else:
                self.curr_behavior_pub.publish(bs.int_app_to_stop.name)

        # approach slowly when traffic light is green as traffic lights are
        # higher priority than traffic signs this behavior is desired
        if self.traffic_light_status == "green":
            self.stopping = False
            self.curr_behavior_pub.publish(bs.int_app_green.name)

        add_debug_entry(self.name, f"Stopping: {self.stopping}")

        # get speed
        speedometer = self.blackboard.get("/carla/hero/Speed")
        if speedometer is not None:
            speed = speedometer.speed
        else:
            return debug_status(
                self.name, py_trees.common.Status.RUNNING, "No speedometer connected"
            )
        if self.virtual_stopline_distance >= target_distance:
            return debug_status(
                self.name,
                py_trees.common.Status.RUNNING,
                "Intersection still approaching",
            )
        elif ((self.virtual_stopline_distance < 2.5)) and (self.stopping):
            # stopped
            self.curr_behavior_pub.publish(bs.int_wait.name)
            if speed < 0.5:
                return debug_status(
                    self.name, py_trees.common.Status.SUCCESS, "Stopped"
                )
            else:
                return debug_status(
                    self.name, py_trees.common.Status.RUNNING, "Stopping..."
                )
        elif (
            self.virtual_stopline_distance < target_distance * 1.5
        ) and not self.stopping:
            self.curr_behavior_pub.publish(bs.int_wait_to_stop.name)
            # drive through intersection even if traffic light turns yellow
            return debug_status(
                self.name, py_trees.common.Status.SUCCESS, "Driving through"
            )

        if (
            self.virtual_stopline_distance < target_distance
            and not self.stopline_detected
        ):
            return debug_status(
                self.name,
                py_trees.common.Status.SUCCESS,
                "Over stopline, driving through",
            )
        else:
            return debug_status(self.name, py_trees.common.Status.RUNNING)

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
    This behavior handles the waiting in front of the stop line at the inter-
    section until there either is no traffic light, the traffic light is
    green or the intersection is clear.
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
        self.blackboard = py_trees.blackboard.Blackboard()
        self.red_light_flag = False
        self.green_light_time = None
        return True

    def initialise(self):
        """
        When is this called?
            The first time your behaviour is ticked and anytime the status is
            not RUNNING thereafter.
        What to do here?
            Any initialisation you need before putting your behaviour to work.
        :return: True
        """
        rospy.loginfo("Wait Intersection")

        self.red_light_flag = False
        self.green_light_time = rospy.get_rostime()
        self.over_stop_line = False
        hero_pos = self.blackboard.get("/paf/hero/current_pos")
        hero_heading = self.blackboard.get("/paf/hero/current_heading")
        self.start_pos = (hero_pos.pose.position.x, hero_pos.pose.position.y)
        hero_heading = hero_heading.data
        trajectory = self.blackboard.get("/paf/hero/trajectory")
        current_wp = self.blackboard.get("/paf/hero/current_wp")
        current_wp = current_wp.data
        trajectory_point = trajectory.poses[int(current_wp) + 10].pose.position
        self.oncoming_distance = 45.0

        x_direction = trajectory_point.x - hero_pos.pose.position.x
        y_direction = trajectory_point.y - hero_pos.pose.position.y

        self.oncoming_counter = 0
        self.red_counter = 0

        rotation_matrix = Transform2D.new_rotation(-hero_heading)
        intersection_point = rotation_matrix * Point2.new(x_direction, y_direction)

        # intersection type to determine a left turn, right turn or driving straight
        # 0 = straight, 1 = left, 2 = right

        if intersection_point.y() > 1.0:
            self.intersection_type = 1
        elif intersection_point.y() < -1.0:
            self.intersection_type = 2
        else:
            self.intersection_type = 0
        return True

    def update(self):
        """
        When is this called?
            Every time your behaviour is ticked.
        What to do here?
           - Triggering, checking, monitoring. Anything...but do not block!
           - Set a feedback message
           - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]
        Waits in front of the intersection until there is a green light.
        In case of turning left, oncoming traffic is checked bevor proceeding.
        :return: py_trees.common.Status.RUNNING, while traffic light is yellow
                 or red
                 py_trees.common.Status.SUCCESS, if the traffic light switched
                 to green or no traffic light is detected
        """
        map: Optional[Map] = self.blackboard.get(BLACKBOARD_MAP_ID)
        if map is None:
            return debug_status(
                self.name, py_trees.common.Status.FAILURE, "Map is None"
            )
        tree = map.build_tree(FlagFilter(is_collider=True, is_hero=False))

        hero = map.hero()
        if hero is None:
            return debug_status(
                self.name, py_trees.common.Status.FAILURE, "No hero in map"
            )

        light_status_msg = self.blackboard.get("/paf/hero/Center/traffic_light_state")

        if light_status_msg is not None:
            traffic_light_status = get_color(light_status_msg.state)
        else:
            traffic_light_status = "No traffic light message"

        add_debug_entry(self.name, f"Traffic light status: {traffic_light_status}")

        if light_status_msg is not None and self.over_stop_line is False:
            traffic_light_status = get_color(light_status_msg.state)
            if traffic_light_status == "red" or traffic_light_status == "yellow":
                # Wait at traffic light
                self.red_light_flag = True
                self.red_counter = 0
                self.green_light_time = rospy.get_rostime()
                self.curr_behavior_pub.publish(bs.int_wait.name)
                return debug_status(
                    self.name,
                    py_trees.common.Status.RUNNING,
                    "Waiting for traffic light",
                )
            elif (
                rospy.get_rostime() - self.green_light_time < rospy.Duration(0.5)
                and traffic_light_status == "green"
            ):
                # Wait approx 0.5s for confirmation
                return debug_status(
                    self.name,
                    py_trees.common.Status.RUNNING,
                    "Wait Confirm green light!",
                )
            elif (
                rospy.get_rostime() - self.green_light_time > rospy.Duration(0.5)
                and traffic_light_status == "green"
            ):
                # Drive through intersection
                self.over_stop_line = True
                if self.intersection_type != 1:
                    self.curr_behavior_pub.publish(bs.int_app_green.name)
                else:
                    # drive a bit over the stopline
                    self.curr_behavior_pub.publish(bs.int_wait_to_stop.name)
                return debug_status(
                    self.name, py_trees.common.Status.RUNNING, "Driving through..."
                )
            else:
                self.over_stop_line = True
                self.curr_behavior_pub.publish(bs.int_wait_to_stop.name)
                return debug_status(
                    self.name,
                    py_trees.common.Status.RUNNING,
                    "No traffic light detected",
                )
        if self.intersection_type != 1:
            self.curr_behavior_pub.publish(bs.int_enter.name)
            return debug_status(
                self.name, py_trees.common.Status.SUCCESS, "No left turn -> continue"
            )

        self.curr_behavior_pub.publish(bs.int_wait.name)
        intersection_clear, intersection_masks = tree.is_lane_free_intersection(
            hero, self.oncoming_distance, 35.0, 0.0
        )
        for mask in intersection_masks:
            add_debug_marker(debug_marker(mask, color=INTERSECTION_MARKER_COLOR))
        if intersection_clear:
            self.oncoming_counter += 1
            if self.oncoming_counter > 2:
                self.curr_behavior_pub.publish(bs.int_enter.name)
                return debug_status(
                    self.name, py_trees.common.Status.SUCCESS, "Intersection clear"
                )
            else:
                return debug_status(
                    self.name,
                    py_trees.common.Status.RUNNING,
                    f"Intersection clear\n\tCounter wait: {self.oncoming_counter}",
                )
        else:
            self.oncoming_counter = 0
            return debug_status(
                self.name, py_trees.common.Status.RUNNING, "Intersection blocked"
            )

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
    This behavior handles the driving through an intersection, it initially
    sets a speed and finishes if the ego vehicle is close to the end of the
    intersection.
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
        This prints a state status message and changes the driving speed for
        the intersection.
        """
        rospy.loginfo("Enter Intersection")

        self.curr_behavior_pub.publish(bs.int_enter.name)

    def update(self):
        """
        When is this called?
            Every time your behaviour is ticked.
        What to do here?
           - Triggering, checking, monitoring. Anything...but do not block!
           - Set a feedback message
           - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]
        Continues driving through the intersection until the vehicle gets
        close enough to the next global way point.
        :return: py_trees.common.Status.RUNNING, if too far from the end of
                 the intersection
                 py_trees.common.Status.SUCCESS, if close to the end of the
                 intersection
                 py_trees.common.Status.FAILURE, if no next path point can be
                 detected.
        """
        next_waypoint_msg = self.blackboard.get("/paf/hero/waypoint_distance")

        if next_waypoint_msg is None:
            return debug_status(
                self.name, py_trees.common.Status.FAILURE, "No next waypoint"
            )
        add_debug_entry(self.name, f"Next waypoint dist: {next_waypoint_msg.distance}")
        if next_waypoint_msg.distance > 8 and next_waypoint_msg.distance < 35:
            self.curr_behavior_pub.publish(bs.int_enter.name)
            return debug_status(
                self.name, py_trees.common.Status.RUNNING, "Driving through..."
            )
        else:
            return debug_status(
                self.name, py_trees.common.Status.SUCCESS, "Done driving through"
            )

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
    reached, the vehicle left the intersection.
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
        by the ROS topics and the target speed publisher.
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
        This prints a state status message and changes the driving speed to
        the street speed limit.
        :return: True
        """
        rospy.loginfo("Leave Intersection")
        self.curr_behavior_pub.publish(bs.int_exit.name)
        return True

    def update(self):
        """
        When is this called?
            Every time your behaviour is ticked.
        What to do here?
           - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]
        Abort this subtree
        :return: py_trees.common.Status.FAILURE, to exit this subtree
        """
        return debug_status(
            self.name, py_trees.common.Status.FAILURE, "Left intersection"
        )

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
