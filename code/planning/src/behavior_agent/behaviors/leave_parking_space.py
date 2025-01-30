import py_trees
import rospy
from std_msgs.msg import String
import numpy as np
from behaviors import behavior_speed as bs

from mapping_common.map import Map

# from mapping import msg


class LeaveParkingSpace(py_trees.behaviour.Behaviour):
    """
    This behavior is triggered in the beginning when the vehicle needs
    to leave the parking space.
    """

    def __init__(self, name):
        """
        Minimal one-time initialisation. A good rule of thumb is to only
        include the initialisation relevant for being able to insert this
        behaviour in a tree for offline rendering to dot graphs.

         :param name: name of the behaviour
        """
        super(LeaveParkingSpace, self).__init__(name)
        rospy.loginfo("LeaveParkingSpace started")
        self.started = False
        self.finished = False

    def setup(self, timeout):
        """
        Delayed one-time initialisation that would otherwise interfere with
        offline rendering of this behaviour in a tree to dot graph or
        validation of the behaviour's configuration.

        This initializes the blackboard to be able to access data written to it
        by the ROS topics and gathers the time to check how much time has
        passed.
        :param timeout: an initial timeout to see if the tree generation is
        successful
        :return: True, as there is nothing to set up.
        """
        self.curr_behavior_pub = rospy.Publisher(
            "/paf/hero/" "curr_behavior", String, queue_size=1
        )
        self.blackboard = py_trees.blackboard.Blackboard()
        self.initPosition = None
        return True

    def initialise(self):
        """
        When is this called?
        The first time your behaviour is ticked and anytime the status is not
        RUNNING thereafter.

        What to do here?
            Any initialisation you need before putting your behaviour to work.
        Get initial position to check how far vehicle has moved during
        execution
        """
        self.initPosition = self.blackboard.get("/paf/hero/current_pos")

    def update(self):
        """
        When is this called?
        Every time your behaviour is ticked.

        pose:
            position:
                x: 294.43757083094295
                y: -1614.961812061094
                z: 211.1994649671884
        What to do here?
            - Triggering, checking, monitoring. Anything...but do not block!
            - Set a feedback message
            - return a py_trees.common.Status.[RUNNING, SUCCESS, FAILURE]

        This behaviour runs until the agent has left the parking space.
        This is checked by calculating the euclidian distance thath the agent
        has moved since the start

        :return: py_trees.common.Status.RUNNING, while the agent is leaving
                                            the parking space
                 py_trees.common.Status.SUCCESS, never to continue with
                                            intersection
                 py_trees.common.Status.FAILURE, if not in parking
                 lane
        """
        position = self.blackboard.get("/paf/hero/current_pos")
        speed = self.blackboard.get("/carla/hero/Speed")
        map_data = self.blackboard.get("/paf/hero/mapping/init_data")

        if not self.finished:
            # calculate distance between start and current position
            if (
                position is not None
                and self.initPosition is not None
                and speed is not None
                and map_data is not None
            ):
                startPos = np.array(
                    [position.pose.position.x, position.pose.position.y]
                )
                endPos = np.array(
                    [
                        self.initPosition.pose.position.x,
                        self.initPosition.pose.position.y,
                    ]
                )
                distance = np.linalg.norm(startPos - endPos)

                # checks if routine hasn't started yet
                if not self.started:
                    map = Map.from_ros_msg(map_data)
                    # checks if the left lane of the car is free,
                    # otherwise pause unparking
                    if map.entities and map.is_lane_free(
                        right_lane=False, lane_length=22.5, lane_transform=-5.0
                    ):
                        rospy.loginfo("Left lane is now free. Starting unparking.")
                        self.started = True
                    else:
                        rospy.logerr_throttle(
                            3, "Left lane is blocked. Paused unparking."
                        )

                # starting unparking if allowed (after left lane is free)
                if self.started:
                    if distance < 1 or speed.speed < 2:
                        self.curr_behavior_pub.publish(bs.parking.name)
                        self.initPosition = position
                        return py_trees.common.Status.RUNNING
                    else:
                        self.finished = True
                        return py_trees.common.Status.FAILURE
                else:
                    return py_trees.common.Status.RUNNING

            else:
                self.initPosition = position
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
        """
        self.logger.debug(
            "  %s [Foo::terminate().terminate()][%s->%s]"
            % (self.name, self.status, new_status)
        )
