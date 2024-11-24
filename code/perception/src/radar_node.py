#!/usr/bin/env python
import rospy
import ros_numpy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from sklearn.cluster import DBSCAN
import json


class RadarNode:
    """See doc/perception/radar_node.md on
    how to configure this node
    """

    def callback(self, data):
        """Process radar Point2Cloud data and publish clustered data points.

        Extracts information from radar data
        and publishes the clustered radar data
        points as a String message

        Args:
            data: Point2Cloud message containing radar data
        """

        clustered_points = cluster_radar_data_from_pointcloud(data, 10)
        clustered_points_json = json.dumps(clustered_points)
        self.dist_array_radar_publisher.publish(clustered_points_json)

    def listener(self):
        """
        Initializes the node and it's publishers
        """
        # run simultaneously.
        rospy.init_node("radar_node")

        # publisher for radar dist_array
        self.dist_array_radar_publisher = rospy.Publisher(
            rospy.get_param(
                "~image_distance_topic", "/paf/hero/Radar/dist_array_unsegmented"
            ),
            String,
            queue_size=10,
        )

        rospy.Subscriber(
            rospy.get_param("~source_topic", "/carla/hero/RADAR"),
            PointCloud2,
            self.callback,
        )

        rospy.spin()


def pointcloud2_to_array(pointcloud_msg):
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(pointcloud_msg)
    distances = np.sqrt(
        cloud_array["x"] ** 2 + cloud_array["y"] ** 2 + cloud_array["z"] ** 2
    )
    return np.column_stack(
        (cloud_array["x"], cloud_array["y"], cloud_array["z"], distances)
    )


def cluster_radar_data_from_pointcloud(
    pointcloud_msg, max_distance, eps=1.0, min_samples=2
):
    data = pointcloud2_to_array(pointcloud_msg)
    filtered_data = data[data[:, 3] < max_distance]
    filtered_data = filtered_data[
        (filtered_data[:, 1] >= -1)
        & (filtered_data[:, 1] <= 1)
        & (filtered_data[:, 2] <= 1.3)
        & (filtered_data[:, 2] >= -0.7)
    ]
    if len(filtered_data) == 0:
        return {}
    coordinates = filtered_data[:, :2]
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = clustering.labels_
    clustered_points = {label: list(labels).count(label) for label in set(labels)}
    clustered_points = {int(label): count for label, count in clustered_points.items()}
    return clustered_points


if __name__ == "__main__":
    radar_node = RadarNode()
    radar_node.listener()
