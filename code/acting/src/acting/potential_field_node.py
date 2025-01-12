#!/usr/bin/env python
from ros_compatibility.node import CompatibleNode

from mapping.msg import Map as MapMsg
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

from acting.entity import Entity
from acting.map import Map

from rospy import Publisher, Subscriber
import ros_compatibility as roscomp
import rospy
from visualization_msgs.msg import Marker

from scipy.ndimage import distance_transform_edt
import numpy as np
from PIL import Image

from acting.helper_functions import interpolate_route, generate_path_from_trajectory


# PARAMETERS
DISTANCE_THRESHOLD = 10
RESOLUTION_SCALE = 10
FORCE_FACTOR = 25
SLOPE = 255
K_VALUE = 0.009
MAX_GRADIENT_DESCENT_STEPS = 20
GRADIENT_FACTOR = 10


class Potential_field_node(CompatibleNode):

    def __init__(self):
        self.entities: list[Entity] = []
        # self.trajectory: Path = self.__generate_default_path()
        self.potential_field_trajectory = Path()
        self.role_name = self.get_param("role_name", "ego_vehicle")

        self.last_pub_time = rospy.get_time()

        self.entity_matrix = np.zeros(
            (
                2 * DISTANCE_THRESHOLD * RESOLUTION_SCALE,
                2 * DISTANCE_THRESHOLD * RESOLUTION_SCALE,
            )
        )

        self.loginfo(f"entity matrix shape {self.entity_matrix.shape}")

        self.entity_matrix_midpoint = (
            DISTANCE_THRESHOLD * RESOLUTION_SCALE,
            DISTANCE_THRESHOLD * RESOLUTION_SCALE,
        )

        # ROS SETUP ###
        self.entities_sub: Subscriber = self.new_subscription(
            msg_type=MapMsg,
            topic="/paf/hero/mapping/init_data",
            callback=self.__get_entities,
            qos_profile=1,
        )

        self.entities_plot_pub: Publisher = self.new_publisher(
            Marker, "/paf/hero/mapping/entities_plot", 1
        )

        self.potential_field_trajectory_pub: Publisher = self.new_publisher(
            Path, "/paf/hero/potential_field_trajectory", 1
        )

        # END ROS SETUP ###

    # CALLBACKS ###
    def __get_entities(self, data: MapMsg):
        self.map = Map.from_ros_msg(data)
        self.entities = self.map.entities_without_hero()

    def __get_trajectory(self, data: Path):
        self.trajectory = data

    # END CALLBACKS ###

    def __filter_entities(self):
        self.entity_matrix = np.zeros(
            (
                2 * DISTANCE_THRESHOLD * RESOLUTION_SCALE,
                2 * DISTANCE_THRESHOLD * RESOLUTION_SCALE,
            )
        )
        # fill in the entities into a matrix
        for entity in self.entities:
            # try to filter out the car entities
            if (
                abs(entity.transform.translation().x()) < 2
                and abs(entity.transform.translation().y()) < 4
            ):
                continue
            x = (
                int(entity.transform.translation().x() * RESOLUTION_SCALE)
                + self.entity_matrix_midpoint[0]
            )
            y = (
                int(entity.transform.translation().y() * RESOLUTION_SCALE)
                + self.entity_matrix_midpoint[1]
            )

            try:
                if self.entity_matrix[x][y] == 0:
                    self.entity_matrix[x][y] += 1
            except IndexError:
                # if the index not in the map horizon, skip the entity
                continue

    def generate_new_trajectory(self, trajectory: Path) -> Path:
        """
        Generates a new trajectory based on the current trajectory
        and the potential field
        :param trajectory: The current trajectory
        :return: The new trajectory
        """
        new_trajectory = Path()
        new_trajectory.header = trajectory.header

        return new_trajectory

    def __calculate_field(self):
        if self.entity_matrix is None:
            return

        # normalize the entity matrix to 0-255
        self.entity_matrix = self.entity_matrix / np.max(self.entity_matrix) * 255

        # flip image verically and smooth out the values
        self.entity_grayscale = np.flipud(self.entity_matrix)
        distances = distance_transform_edt(self.entity_grayscale == 0)

        # INTRODUCE A FORCE POINTING TO THE TOP OF THE IMAGE (TO MAKE THE CAR DRIVE)
        # slope distances matrix to the top of the image
        slope_array: np.ndarray = np.linspace(SLOPE, 0, self.entity_matrix.shape[0])
        # make slope array same shape as distances
        slope_array = np.tile(slope_array, (self.entity_matrix.shape[1], 1)).T
        distances += slope_array

        # plot the updated trajectory
        # smooth the values with exponential decay
        smoothed_values = np.exp(-K_VALUE * distances)
        # normalize smoothed values to 0-255
        smoothed_values = smoothed_values / np.max(smoothed_values) * 255
        distances[distances == 0] = smoothed_values[distances == 0]

        gradient_x, gradient_y = np.gradient(smoothed_values)

        # GRADIENT DESCENT
        finished = False
        num_steps = 0
        points: list[tuple[float]] = []
        plot_points: list[tuple[int]] = []

        x, y = self.entity_matrix_midpoint
        while not finished and num_steps < MAX_GRADIENT_DESCENT_STEPS:
            # move along the gradient
            try:
                dx = gradient_x[x, y] * GRADIENT_FACTOR
                dy = gradient_y[x, y] * GRADIENT_FACTOR
                x -= int(dx)
                y -= int(dy)
                plot_points.append((x, y))
                points.append(
                    (
                        # back to the original coordinates
                        -(x - self.entity_matrix_midpoint[0]) / RESOLUTION_SCALE,
                        (y - self.entity_matrix_midpoint[1]) / RESOLUTION_SCALE,
                    )
                )
                if int(dx) <= 0:
                    self.loginfo("x is smaller than previous x, car going backwards")
                    finished = True
                num_steps += 1
            except IndexError:
                self.loginfo(f"index out of bounds after {num_steps} steps, way found")
                finished = True

        # generate a path from the trajectory and publish
        self.potential_field_trajectory = generate_path_from_trajectory(points)
        self.potential_field_trajectory_pub.publish(self.potential_field_trajectory)
        self.loginfo(
            f"Potential field trajectory published after {str(rospy.get_time() - self.last_pub_time)}"
        )
        self.last_pub_time = rospy.get_time()

        # EVERYTHING HAPPENING WITH ENTYITY MATRIX FOR PLOTTING
        self.entity_matrix_for_plotting = np.zeros(
            (
                self.entity_matrix.shape[0],
                self.entity_matrix.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        self.entity_matrix_for_plotting[:, :, 0] = np.flipud(self.entity_matrix)
        self.entity_matrix_for_plotting[:, :, 2] = smoothed_values
        for point in plot_points:
            try:
                self.entity_matrix_for_plotting[point[0], point[1], 1] = 255
            except IndexError:
                continue

    def save_image(self, matrix: np.ndarray, path: str):
        image = Image.fromarray(matrix)
        image = image.convert("RGB")
        image.save(path)
        # self.loginfo(f"Image saved at {path}")

    def run(self):
        self.loginfo("Potential Field Node Running")

        def loop(timerevent=None, logged=False):
            starttime = rospy.get_time()
            if logged:
                self.loginfo(f"TIMER LOOP START {starttime}")
            plotting_start = rospy.get_time()
            self.__filter_entities()
            if logged:
                self.loginfo(f"TIME TAKEN FOR FILTERING {rospy.get_time()-starttime}")
            self.__calculate_field()
            # PLOTTING
            self.save_image(self.entity_matrix_for_plotting, "entity_matrix.png")
            if logged:
                self.loginfo(
                    f"TIME TAKEN FOR PLOTTING {rospy.get_time()-plotting_start}"
                )
                self.loginfo(f"TIMER TAKEN FOR LOOP {rospy.get_time()-starttime}")

        self.new_timer(0.5, loop)
        self.spin()


def main(args=None):
    roscomp.init("potential_field_node", args=args)

    try:
        node = Potential_field_node()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()
