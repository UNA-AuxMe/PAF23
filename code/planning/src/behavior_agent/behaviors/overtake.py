import py_trees
from std_msgs.msg import String

import rospy
import numpy as np

from scipy.spatial.transform import Rotation

from behaviors import behavior_speed as bs
from local_planner.utils import NUM_WAYPOINTS, TARGET_DISTANCE_TO_STOP, convert_to_ms
from mapping_common.map import Map

"""
Source: https://github.com/ll7/psaf2
"""


# Varaible to determine the distance to overtake the object
OVERTAKE_EXECUTING = 0


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
        :return: py_trees.common.Status.SUCCESS, if the vehicle is within range
                    for the overtaking procedure
                 py_trees.common.Status.FAILURE, if we are too far away for
                 the overtaking procedure
        """

        obstacle_msg = self.blackboard.get("/paf/hero/collision")
        current_position = self.blackboard.get("/paf/hero/current_pos")
        current_heading = self.blackboard.get("/paf/hero/current_heading").data

        if obstacle_msg is None or current_position is None or current_heading is None:
            return py_trees.common.Status.FAILURE
        current_position = [
            current_position.pose.position.x,
            current_position.pose.position.y,
            current_position.pose.position.z,
        ]

        obstacle_distance = obstacle_msg.data[0]
        obstacle_speed = obstacle_msg.data[1]

        if obstacle_distance == np.Inf:
            return py_trees.common.Status.FAILURE
        # calculate approx collision position in global coords
        rotation_matrix = Rotation.from_euler("z", current_heading)
        # Apply current heading to absolute distance vector
        # and add to current position
        pos_moved_in_x_direction = current_position + rotation_matrix.apply(
            np.array([obstacle_distance, 0, 0])
        )

        if np.linalg.norm(pos_moved_in_x_direction - current_position) < 1:
            # current collision is not near trajectory lane
            rospy.logerr("Obstacle is not near trajectory lane")
            return py_trees.common.Status.FAILURE

        if obstacle_speed < 2 and obstacle_distance < 30:
            self.counter_overtake += 1
            rospy.loginfo("Overtake counter: " + str(self.counter_overtake))
            if self.counter_overtake > 3:
                return py_trees.common.Status.SUCCESS
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
    behaviours.road_features.overtake_ahead is triggered.
    It than handles the procedure for overtaking.
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
        self.blackboard = py_trees.blackboard.Blackboard()
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
        self.ot_distance = 30

        self.clear_distance = 30

    def update(self):
        """
        When is this called?
        Every time your behaviour is ticked.
        What to do here?
            - Triggering, checking, monitoring. Anything...but do not block!
            - Set a feedback message
            - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]

        Gets the current distance to overtake, the current oncoming lane status and the
        distance to collsion object. Slows down while oncoming blocked until stopped
        or oncoming clear.
        :return: py_trees.common.Status.RUNNING, if too far from overtaking
                 py_trees.common.Status.SUCCESS, if stopped behind the blocking
                 object or entered the process.
                 py_trees.common.Status.FAILURE, if the overtake is aborted
        """
        global OVERTAKE_EXECUTING
        # Update distance to collision object
        _dis = self.blackboard.get("/paf/hero/collision")

        # Intermediate layer map integration
        map_data = self.blackboard.get("/paf/hero/mapping/init_data")
        map = Map.from_ros_msg(map_data)

        entity = map.get_entity_in_front_or_back(True)

        left_free = map.is_lane_free(right_lane=False)
        rospy.loginfo(f"OT: Left lane free: {left_free}")

        if entity is not None:
            rospy.loginfo(
                f"Translation to car in front: {entity.transform.translation().x()},"
                f"{entity.transform.translation().y()}"
            )
        if _dis is not None:
            self.ot_distance = _dis.data[0]
            rospy.loginfo(f"Overtake distance: {self.ot_distance}")
            OVERTAKE_EXECUTING = self.ot_distance

        if np.isinf(self.ot_distance):
            rospy.loginfo("OvertakeApproach: Abort")
            return py_trees.common.Status.FAILURE

        # slow down before overtake if blocked
        if self.ot_distance < 20.0:
            data = self.blackboard.get("/paf/hero/oncoming")
            if data is not None:
                distance_oncoming = data.data
            else:
                distance_oncoming = 35

            if (
                distance_oncoming is not None
                and distance_oncoming > self.clear_distance
            ):
                rospy.loginfo("Overtake is free not slowing down!")
                self.curr_behavior_pub.publish(bs.ot_app_free.name)
                return py_trees.common.Status.SUCCESS
            else:
                rospy.loginfo("Overtake Approach: oncoming blocked slowing down")
                self.curr_behavior_pub.publish(bs.ot_app_blocked.name)

        # get speed
        speedometer = self.blackboard.get("/carla/hero/Speed")
        if speedometer is not None:
            speed = speedometer.speed
        else:
            rospy.logwarn("no speedometer connected")
            return py_trees.common.Status.RUNNING

        if self.ot_distance > 20.0:
            # too far
            rospy.loginfo(
                f"Overtake Approach: still approaching obstacle, "
                f"distance: {self.ot_distance}"
            )
            return py_trees.common.Status.RUNNING
        elif speed < convert_to_ms(2.0) and self.ot_distance < TARGET_DISTANCE_TO_STOP:
            # stopped
            rospy.loginfo("Overtake Approach: stopped behind obstacle")
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
        return True

    def update(self):
        """
        When is this called?
            Every time your behaviour is ticked.
        What to do here?
           - Triggering, checking, monitoring. Anything...but do not block!
           - Set a feedback message
           - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]

        Waits behind the road object until the lidar distance meets the
        requirement for the clear distance.

        :return: py_trees.common.Status.RUNNING, while clear distance is bigger
                    than lidar_distance
                 py_trees.common.Status.SUCCESS,if no lidar distance is present
                    or the lidar distance is bigger than the clear distance
        """

        # Update distance to collison and distance for clear
        clear_distance = 30
        obstacle_msg = self.blackboard.get("/paf/hero/collision")
        if obstacle_msg is None:
            rospy.logerr("No OBSTACLE in overtake wait")
            return py_trees.common.Status.FAILURE

        data = self.blackboard.get("/paf/hero/oncoming")
        if data is not None:
            distance_oncoming = data.data
        else:
            distance_oncoming = 35

        if distance_oncoming is not None:
            if distance_oncoming > clear_distance:
                rospy.loginfo("Overtake is free!")
                self.curr_behavior_pub.publish(bs.ot_wait_free.name)
                return py_trees.common.Status.SUCCESS
            else:
                rospy.loginfo(
                    f"Overtake still blocked, distance to oncoming: {distance_oncoming}"
                )
                self.curr_behavior_pub.publish(bs.ot_wait_stopped.name)
                return py_trees.common.Status.RUNNING
        elif obstacle_msg.data[0] == np.inf:
            rospy.loginf("No OBSTACLE in overtake wait")
            return py_trees.common.Status.FAILURE
        else:
            rospy.loginfo("No Lidar Distance in overtake wait")
            return py_trees.common.Status.SUCCESS

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
