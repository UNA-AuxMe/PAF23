#!/usr/bin/env python


from ros_compatibility.node import CompatibleNode
import ros_compatibility as roscomp
import ros_numpy
import rospy
from visualization_msgs.msg import Marker
import numpy as np
from typing import List, Optional

from mapping_common.entity import Entity, Flags, Car, Motion2D
from mapping_common.transform import Transform2D, Vector2
from mapping_common.shape import Circle, Polygon, Rectangle
from mapping_common.map import Map
from mapping_common.filter import MapFilter, GrowthMergingFilter
from mapping.msg import Map as MapMsg
from mapping.msg import ClusteredPointsArray
from sensor_msgs.msg import PointCloud2, PointField
from carla_msgs.msg import CarlaSpeedometer
from shapely.geometry import MultiPoint


# from shapely.validation import orient


class MappingDataIntegrationNode(CompatibleNode):
    """Creates the initial map data frame based on all kinds of sensor data

    It applies several filters to the map and
    then sends it off to other consumers (planning, acting)

    This node sends the maps off at a fixed rate.
    (-> It buffers incoming sensor data slightly)
    """

    lidar_data: Optional[PointCloud2] = None
    hero_speed: Optional[CarlaSpeedometer] = None
    lidar_clustered_points_data: Optional[ClusteredPointsArray] = None
    radar_clustered_points_data: Optional[ClusteredPointsArray] = None
    vision_clustered_points_data: Optional[ClusteredPointsArray] = None

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.new_subscription(
            topic=self.get_param("~lidar_topic", "/carla/hero/LIDAR"),
            msg_type=PointCloud2,
            callback=self.lidar_callback,
            qos_profile=1,
        )
        self.lanemarkings = None
        self.new_subscription(
            topic=self.get_param(
                "~lanemarkings_init_topic", "/paf/hero/mapping/init_lanemarkings"
            ),
            msg_type=MapMsg,
            callback=self.lanemarkings_callback,
            qos_profile=1,
        )
        self.new_subscription(
            topic=self.get_param("~hero_speed_topic", "/carla/hero/Speed"),
            msg_type=CarlaSpeedometer,
            callback=self.hero_speed_callback,
            qos_profile=1,
        )
        self.new_subscription(
            topic=self.get_param(
                "~clustered_points_lidar_topic", "/paf/hero/Lidar/clustered_points"
            ),
            msg_type=ClusteredPointsArray,
            callback=self.lidar_clustered_points_callback,
            qos_profile=1,
        )
        self.new_subscription(
            topic=self.get_param(
                "~clustered_points_vision_topic", "/paf/hero/visualization_pointcloud"
            ),
            msg_type=ClusteredPointsArray,
            callback=self.vision_clustered_points_callback,
            qos_profile=1,
        )
        self.new_subscription(
            topic=self.get_param(
                "~clustered_points_radar_topic", "/paf/hero/Radar/clustered_points"
            ),
            msg_type=ClusteredPointsArray,
            callback=self.radar_clustered_points_callback,
            qos_profile=1,
        )

        self.map_publisher = self.new_publisher(
            msg_type=MapMsg,
            topic=self.get_param("~map_init_topic", "/paf/hero/mapping/init_data"),
            qos_profile=1,
        )

        self.cluster_points_publisher = self.new_publisher(
            msg_type=PointCloud2,
            topic=self.get_param(
                "~cluster_points_topic", "/paf/hero/mapping/clusterpoints"
            ),
            qos_profile=1,
        )

        self.rate = self.get_param("~map_publish_rate", 20)
        self.new_timer(1.0 / self.rate, self.publish_new_map)

    def hero_speed_callback(self, data: CarlaSpeedometer):
        self.hero_speed = data

    def lidar_clustered_points_callback(self, data: ClusteredPointsArray):
        self.lidar_clustered_points_data = data

    def radar_clustered_points_callback(self, data: ClusteredPointsArray):
        rospy.loginfo(f"radar data received in intermediate layer callback: {data}")
        self.radar_clustered_points_data = data

    def vision_clustered_points_callback(self, data: ClusteredPointsArray):
        self.vision_clustered_points_data = data

    def lidar_callback(self, data: PointCloud2):
        self.lidar_data = data

    def entities_from_lidar_marker(self) -> List[Entity]:
        data = self.lidar_marker_data
        if data is None or not hasattr(data, "markers") or data.markers is None:
            rospy.logwarn("No valid marker data received.")
            return []

        lidar_entities = []
        for marker in data.markers:
            if marker.type != Marker.CUBE:
                rospy.logwarn(f"Skipping non-CUBE marker with ID: {marker.id}")
                continue
            # Extract position (center of the cube)
            x_center = marker.pose.position.x
            y_center = marker.pose.position.y

            # Extract dimensions (scale gives the size of the cube)
            width = marker.scale.x
            length = marker.scale.y

            # Create a shape and transform using the cube's data
            shape = Rectangle(width, length)  # 2D rectangle for lidar data
            v = Vector2.new(x_center, y_center)  # 2D position in x-y plane
            transform = Transform2D.new_translation(v)

            # Add entity to the list
            flags = Flags(is_collider=True)
            e = Entity(
                confidence=1,
                priority=0.25,
                shape=shape,
                transform=transform,
                timestamp=marker.header.stamp,
                flags=flags,
            )
            lidar_entities.append(e)

        return lidar_entities

    def entities_from_radar_marker(self) -> List[Entity]:
        data = self.radar_marker_data
        if data is None or not hasattr(data, "markers") or data.markers is None:
            # Handle cases where data or markers are invalid
            rospy.logwarn("No valid marker data received.")
            return []

        radar_entities = []
        for marker in data.markers:
            if marker.type != Marker.CUBE:
                rospy.logwarn(f"Skipping non-CUBE marker with ID: {marker.id}")
                continue
            # Extract position (center of the cube)
            x_center = marker.pose.position.x
            y_center = marker.pose.position.y

            # Extract dimensions (scale gives the size of the cube)
            width = marker.scale.x
            length = marker.scale.y

            # Create a shape and transform using the cube's data
            shape = Rectangle(width, length)  # 2D rectangle for lidar data
            v = Vector2.new(x_center, y_center)  # 2D position in x-y plane
            transform = Transform2D.new_translation(v)

            # Add entity to the list
            flags = Flags(is_collider=True)
            e = Entity(
                confidence=1,
                priority=0.25,
                shape=shape,
                transform=transform,
                timestamp=marker.header.stamp,
                flags=flags,
            )
            radar_entities.append(e)

        return radar_entities

    def lanemarkings_callback(self, data: MapMsg):
        map = Map.from_ros_msg(data)
        self.lanemarkings = map.entities_without_hero()

    def entities_from_lidar(self) -> List[Entity]:
        if self.lidar_data is None:
            return []

        data = self.lidar_data
        coordinates = ros_numpy.point_cloud2.pointcloud2_to_array(data)
        coordinates = coordinates.view(
            (coordinates.dtype[0], len(coordinates.dtype.names))
        )
        shape = Circle(self.get_param("~lidar_shape_radius", 0.15))
        z_min = self.get_param("~lidar_z_min", -1.5)
        z_max = self.get_param("~lidar_z_max", 1.0)
        priority = self.get_param("~lidar_priority", 0.25)

        # Ignore street level lidar points and stuff above
        filtered_coordinates = coordinates[
            np.bitwise_and(coordinates[:, 2] >= z_min, coordinates[:, 2] <= z_max)
        ]
        # Get rid of points because performance
        coordinate_count = filtered_coordinates.shape[0]
        sampled_coordinates = filtered_coordinates[
            np.random.choice(
                coordinate_count,
                int(
                    coordinate_count
                    * (1.0 - self.get_param("~lidar_discard_probability", 0.9))
                ),
                replace=False,
            ),
            :,
        ]
        lidar_entities = []
        for x, y, z, intensity in sampled_coordinates:
            v = Vector2.new(x, y)
            transform = Transform2D.new_translation(v)
            flags = Flags(is_collider=True)
            e = Entity(
                confidence=0.5 * intensity,
                priority=priority,
                shape=shape,
                transform=transform,
                timestamp=data.header.stamp,
                flags=flags,
            )
            lidar_entities.append(e)

        return lidar_entities

    def publish_cluster_points(self, cluster_points, frame_id="hero"):
        """
        Publishes the cluster points as a PointCloud2 message.
        Args:
            cluster_points: numpy array of shape (N, 3), where each row is [x, y, z]
            frame_id: coordinate frame ID for the points
        """
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.frame_id = frame_id
        point_cloud_msg.header.stamp = rospy.Time.now()
        point_cloud_msg.height = 1  # Unstructured point cloud
        point_cloud_msg.width = len(cluster_points)
        point_cloud_msg.is_dense = True
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud_msg.point_step = 12  # 4 bytes * 3 fields
        point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width
        point_cloud_msg.data = np.array(cluster_points, dtype=np.float32).tobytes()

        self.cluster_points_publisher.publish(point_cloud_msg)

    def create_entities_from_clusters(self, sensortype="") -> List[Entity]:
        rospy.loginfo("Create entities in intermediate layer")
        data = None
        if sensortype == "radar":
            data = self.radar_clustered_points_data
            self.radar_clustered_points_data = None
            rospy.loginfo("radar data received in intermediate layer")
        elif sensortype == "lidar":
            data = self.lidar_clustered_points_data
            self.lidar_clustered_points_data = None
        elif sensortype == "vision":
            pass
        else:
            raise ValueError(f"Unbekannter Sensortyp: {sensortype}")

        if data is None:
            return []

        clusterpointsarray = np.array(data.clusterPointsArray)

        # Überprüfen, ob die Länge des Arrays durch 3 teilbar ist
        if len(clusterpointsarray) % 3 != 0:
            raise ValueError(
                "Die Länge von clusterPointsArray ist nicht durch 3 teilbar. Überprüfe \
                die Datenquelle."
            )

        # Umformen in (n, 3)
        clusterpointsarray = clusterpointsarray.reshape(-1, 3)

        indexarray = np.array(data.indexArray)
        motionarray = np.array(data.motionArray) if data.motionArray else None
        motion_array_converted = None

        if motionarray is not None:
            motion_array_converted = [
                Motion2D.from_ros_msg(motion_msg) for motion_msg in motionarray
            ]

            motion_array_converted = np.array(motion_array_converted)

        unique_labels = np.unique(indexarray)
        entities = []

        for label in unique_labels:
            if label == -1:
                # -1 kann für Rauschen oder ungültige Cluster stehen
                rospy.logwarn("label -1")
                continue

            # Filtere Punkte für den aktuellen Cluster
            cluster_mask = indexarray == label
            cluster_points = clusterpointsarray[cluster_mask]

            # Prüfe, ob genügend Punkte für ein Polygon vorhanden sind
            if cluster_points.shape[0] < 3:
                continue

            # Extrahiere nur die XY-Koordinaten
            cluster_points_xy = cluster_points[:, :2]
            if not np.array_equal(cluster_points_xy[0], cluster_points_xy[-1]):
                # Füge den Startpunkt am Ende hinzu, um das Polygon zu schließen
                cluster_points_xy = np.vstack([cluster_points_xy, cluster_points_xy[0]])

            # cluster_polygon = ShapelyPolygon(cluster_points_xy).convex_hull
            cluster_polygon = MultiPoint(cluster_points_xy)
            cluster_polygon_hull = cluster_polygon.convex_hull
            if cluster_polygon_hull.is_empty or not cluster_polygon_hull.is_valid:
                rospy.loginfo("Empty hull")
                continue
            if cluster_polygon_hull.geom_type == "LineString":
                rospy.loginfo("LineString detected, skipping this cluster")
                continue

            shape = Polygon.from_shapely(
                cluster_polygon_hull, make_centered=True  # type: ignore
            )

            # Optional: Berechne die Bewegung (Motion)
            motion = None
            if motion_array_converted is not None:
                cluster_motion = motion_array_converted[indexarray == label]
                motion = cluster_motion[0]
                if self.hero_speed is not None:
                    motion_vector_hero = Vector2.forward() * self.hero_speed.speed
                    relative_motion = motion.linear_motion - motion_vector_hero
                    motion = Motion2D(relative_motion, angular_velocity=0.0)

                # rospy.loginfo(
                #     f"Relative motion: x={motion.linear_motion.x()}, y={motion.linear_motion.y()}, angular={motion.angular_velocity}"
                # )

            # Optional: Füge die Objektklasse hinzu
            # object_class = None
            # if objectclassarray is not None:
            #     cluster_class = objectclassarray[indexarray == label]
            #     object_class = np.unique(cluster_class)[0]  # Nimm die häufigste
            # Klasse

            transform = shape.offset
            shape.offset = Transform2D.identity()

            flags = Flags(is_collider=True)

            # Erstelle die Entity
            entity = Entity(
                confidence=1,
                priority=0.25,
                shape=shape,
                transform=transform,
                timestamp=rospy.Time.now(),
                flags=flags,
                motion=motion,
            )
            entities.append(entity)

        self.publish_cluster_points(clusterpointsarray)

        return entities

    def create_hero_entity(self) -> Optional[Car]:
        if self.hero_speed is None:
            return None

        motion = Motion2D(Vector2.forward() * self.hero_speed.speed)
        timestamp = self.hero_speed.header.stamp
        # Shape based on https://www.motortrend.com/cars/
        # lincoln/mkz/2020/specs/?trim=Base+Sedan
        shape = Rectangle(
            length=4.92506,
            width=1.86436,
            offset=Transform2D.new_translation(Vector2.new(0.0, 0.0)),
        )
        transform = Transform2D.identity()
        flags = Flags(is_collider=True, is_hero=True)
        hero = Car(
            confidence=1.0,
            priority=1.0,
            shape=shape,
            transform=transform,
            timestamp=timestamp,
            flags=flags,
            motion=motion,
        )
        return hero

    def publish_new_map(self, timer_event=None):
        rospy.loginfo("Publish new map() is running")
        hero_car = self.create_hero_entity()
        if hero_car is None:
            return

        entities = []
        entities.append(hero_car)

        if self.lidar_clustered_points_data is not None and self.get_param(
            "~enable_lidar_cluster"
        ):
            entities.extend(self.create_entities_from_clusters(sensortype="lidar"))
        rospy.loginfo(f"radar data: {self.radar_clustered_points_data}")
        if self.radar_clustered_points_data is not None and self.get_param(
            "~enable_radar_cluster"
        ):
            rospy.loginfo("Publish new map radar")
            entities.extend(self.create_entities_from_clusters(sensortype="radar"))
        if self.vision_clustered_points_data is not None and self.get_param(
            "enable_vision_cluster"
        ):
            entities.extend(self.create_entities_from_clusters(sensortype="vision"))

        if self.lanemarkings is not None and self.get_param("~enable_lane_marker"):
            entities.extend(self.lanemarkings)
        if self.lidar_data is not None and self.get_param("~enable_raw_lidar_points"):
            entities.extend(self.entities_from_lidar())

        # lane_box_entities visualizes the shape and position of the lane box
        # which is used for lane_free function
        # lane_box_entities = [
        #    Entity(
        #        confidence=100.0,
        #        priority=100.0,
        #        shape=Rectangle(
        #            length=22.5,
        #            width=1.5,
        #            offset=Transform2D.new_translation(Vector2.new(-2.5, 2.2)),
        #        ),
        #        transform=Transform2D.identity(),
        #        flags=Flags(is_ignored=True),
        #    )
        # ]
        # entities.extend(lane_box_entities)

        # Will be used when the new function for entity creation is implemented
        # if self.get_param("enable_vision_points"):
        #    entities.extend(self.entities_from_vision_points())

        stamp = rospy.get_rostime()
        map = Map(timestamp=stamp, entities=entities)

        for filter in self.get_current_map_filters():
            map = filter.filter(map)
        msg = map.to_ros_msg()
        self.map_publisher.publish(msg)

    def get_current_map_filters(self) -> List[MapFilter]:
        map_filters: List[MapFilter] = []

        if self.get_param("~enable_merge_filter"):
            map_filters.append(
                GrowthMergingFilter(
                    growth_distance=self.get_param("~merge_growth_distance"),
                    min_merging_overlap_percent=self.get_param(
                        "~min_merging_overlap_percent"
                    ),
                    min_merging_overlap_area=self.get_param(
                        "~min_merging_overlap_area"
                    ),
                    simplify_tolerance=self.get_param("~polygon_simplify_tolerance"),
                )
            )

        return map_filters


if __name__ == "__main__":
    name = "mapping_data_integration"
    roscomp.init(name)
    node = MappingDataIntegrationNode(name)
    node.spin()
