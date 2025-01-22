#!/usr/bin/env python


from ros_compatibility.node import CompatibleNode
import ros_compatibility as roscomp
import ros_numpy
import rospy
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
from typing import List, Optional

from mapping_common.entity import Entity, Flags, Car, Motion2D
from mapping_common.transform import Transform2D, Vector2
from mapping_common.shape import Circle, Polygon, Rectangle
from mapping_common.map import Map
from mapping.msg import Map as MapMsg
from mapping.msg import ClusteredPointsArray
from sensor_msgs.msg import PointCloud2
from carla_msgs.msg import CarlaSpeedometer
import sensor_msgs.point_cloud2 as pc2
from shapely.geometry import Polygon as ShapelyPolygon


class MappingDataIntegrationNode(CompatibleNode):
    """Creates the initial map data frame based on all kinds of sensor data

    Sends this map off to Filtering and other consumers (planning, acting)

    This node sends the maps off at a fixed rate.
    (-> It buffers incoming sensor data slightly)
    """

    lidar_data: Optional[PointCloud2] = None
    hero_speed: Optional[CarlaSpeedometer] = None
    lidar_marker_data: Optional[MarkerArray] = None
    lidar_cluster_entities_data: Optional[List[Entity]] = None
    radar_clustered_points_data: Optional[ClusteredPointsArray] = None
    radar_marker_data: Optional[MarkerArray] = None

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

        self.new_subscription(
            topic=self.get_param("~lidar_topic", "/carla/hero/LIDAR"),
            msg_type=PointCloud2,
            callback=self.lidar_callback,
            qos_profile=1,
        )
        self.new_subscription(
            topic=self.get_param("~hero_speed_topic", "/carla/hero/Speed"),
            msg_type=CarlaSpeedometer,
            callback=self.hero_speed_callback,
            qos_profile=1,
        )
        self.new_subscription(
            topic=self.get_param("~marker_topic", "/paf/hero/Lidar/Marker"),
            msg_type=MarkerArray,
            callback=self.lidar_marker_callback,
            qos_profile=1,
        )
        self.new_subscription(
            topic=self.get_param("~entity_topic", "/paf/hero/Lidar/cluster_entities"),
            msg_type=MapMsg,
            callback=self.lidar_cluster_entities_callback,
            qos_profile=1,
        )
        # self.new_subscription(
        #     topic=self.get_param("~entity_topic", "/paf/hero/Radar/cluster_entities"),
        #     msg_type=MapMsg,
        #     callback=self.radar_cluster_entities_callback,
        #     qos_profile=1,
        # )
        self.new_subscription(
            topic=self.get_param(
                "~clustered_points_radar_topic", "/paf/hero/Radar/clustered_points"
            ),
            msg_type=ClusteredPointsArray,
            callback=self.radar_clustered_points_callback,
            qos_profile=1,
        )

        self.new_subscription(
            topic=self.get_param("~marker_topic", "/paf/hero/Radar/Marker"),
            msg_type=MarkerArray,
            callback=self.radar_marker_callback,
            qos_profile=1,
        )

        self.map_publisher = self.new_publisher(
            msg_type=MapMsg,
            topic=self.get_param("~map_init_topic", "/paf/hero/mapping/init_data"),
            qos_profile=1,
        )
        # Will be removed when the new function for entity creation is implemented
        self.vision_node_pointcloud_publisher = self.new_publisher(
            msg_type=PointCloud2,
            topic="/paf/hero/mapping/temporary_pointcloud",
            qos_profile=1,
        )
        self.rate = self.get_param("~map_publish_rate", 20)
        self.new_timer(1.0 / self.rate, self.publish_new_map)

    def hero_speed_callback(self, data: CarlaSpeedometer):
        self.hero_speed = data

    def lidar_marker_callback(self, data: MarkerArray):
        self.lidar_marker_data = data

    def lidar_cluster_entities_callback(self, data: MapMsg):
        self.lidar_cluster_entities_data = data

    def radar_clustered_points_callback(self, data: ClusteredPointsArray):
        self.radar_clustered_points_data = data

    def radar_marker_callback(self, data: MarkerArray):
        self.radar_marker_data = data

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
            # Extract position (center of the cube) and calculate 2 meter offset
            # because of radar positioning
            x_center = marker.pose.position.x + 2
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

    def create_entities_from_clusters(self) -> List[Entity]:
        clusterpointsarray = self.radar_clustered_points_data.clusterPointsArray
        indexarray = self.radar_clustered_points_data.indexArray
        motionarray = self.radar_clustered_points_data.motionArray
        # objectclassarray = self.radar_clustered_points_data.object_class
        self.radar_clustered_points_data = None

        unique_labels = np.unique(indexarray)
        entities = []

        for label in unique_labels:
            if label == -1:
                # -1 kann für Rauschen oder ungültige Cluster stehen
                continue

            # Filtere Punkte für den aktuellen Cluster
            cluster_points = clusterpointsarray[indexarray == label]

            # Prüfe, ob genügend Punkte für ein Polygon vorhanden sind
            if cluster_points.shape[0] < 3:
                continue

            # Erstelle ein Shapely-Polygon (nur x und y werden verwendet)
            polygon = ShapelyPolygon(cluster_points[:, :2])

            # Optional: Berechne die Bewegung (Motion)
            motion = None
            if motionarray is not None:
                cluster_motion = motionarray[indexarray == label][0]
                # avg_motion = np.mean(cluster_motion, axis=0)
                motion = Motion2D(
                    Vector2.new(cluster_motion[0], cluster_motion[1])
                )  # Geschwindigkeit (x, y)

            # Optional: Füge die Objektklasse hinzu
            # object_class = None
            # if objectclassarray is not None:
            #     cluster_class = objectclassarray[indexarray == label]
            #     object_class = np.unique(cluster_class)[0]  # Nimm die häufigste Klasse

            if not polygon.is_valid:
                rospy.logwarn("Skipping non-Polygon entity.")
                continue

            # Extrahiere das Zentrum des Polygons (Mittelpunktskoordinaten)
            centroid = polygon.centroid
            x_center = centroid.x
            y_center = centroid.y

            v = Vector2.new(x_center, y_center)  # 2D position in x-y plane
            transform = Transform2D.new_translation(v)

            flags = Flags(is_collider=True)
            # Erstelle die Entity
            entity = Entity(
                confidence=1,
                priority=0.25,
                shape=Polygon(polygon.coords),
                transform=transform,
                timestamp=rospy.Time.now(),
                flags=flags,
                motion=motion,
            )
            entities.append(entity)

        return entities

    def create_shapely_polygons_from_ClusteredPoints(
        self, sensor
    ) -> List[ShapelyPolygon]:
        """
        Erstellt ein Shapely-Polygon aus einer ClusterPoints-Nachricht.

        Args:
            pointcloud_msg (PointCloud2): Die ROS PointCloud2 Nachricht.

        Returns:
            ShapelyPolygon: Das erstellte Shapely Polygon.
        """

        polygons = []
        # unique_labels = np.unique(points_with_labels[:, -1])
        pointclouds = self.cluster_data
        for pointcloud2 in pointclouds:

            points = []
            for p in PointCloud2.read_points(
                pointclouds, skip_nans=True, field_names=("x", "y")
            ):
                points.append((p[0], p[1]))  # Nimmt nur die x- und y-Koordinaten

            # Überprüfen, ob genügend Punkte für ein Polygon vorhanden sind
            if len(points) < 3:
                raise ValueError("Ein Polygon benötigt mindestens 3 Punkte.")

            # Optional: Erstelle den konvexen Hüllraum der Punkte
            # from scipy.spatial import ConvexHull
            # hull = ConvexHull(points)
            # hull_points = [points[i] for i in hull.vertices]

            # Erstelle das Shapely-Polygon
            polygon = ShapelyPolygon(points)
            polygons.append(polygon)
        return polygons

    # def entities_from_shapelyPolygons(self) -> List[Entity]:
    #     data = self.sensor_data
    #     if data is None or not hasattr(data, "markers") or data.markers is None:
    #         # Handle cases where data or markers are invalid
    #         rospy.logwarn("No valid marker data received.")
    #         return []

    #     radar_entities = []
    #     for pointcloudcluster in data:
    #         if marker.type != Marker.CUBE:
    #             rospy.logwarn(f"Skipping non-CUBE marker with ID: {marker.id}")
    #             continue
    #         # Extract position (center of the cube) and calculate 2 meter offset
    #         # because of radar positioning
    #         x_center = marker.pose.position.x + 2
    #         y_center = marker.pose.position.y

    #         # Extract dimensions (scale gives the size of the cube)
    #         width = marker.scale.x
    #         length = marker.scale.y

    #         # Create a shape and transform using the cube's data
    #         shape = Rectangle(width, length)  # 2D rectangle for lidar data
    #         v = Vector2.new(x_center, y_center)  # 2D position in x-y plane
    #         transform = Transform2D.new_translation(v)

    #         # Add entity to the list
    #         flags = Flags(is_collider=True)
    #         e = Entity(
    #             confidence=1,
    #             priority=0.25,
    #             shape=shape,
    #             transform=transform,
    #             timestamp=marker.header.stamp,
    #             flags=flags,
    #         )
    #         radar_entities.append(e)

    #     return radar_entities

    # def create_shapely_polygons_from_pointclouds(self, sensor) -> List[ShapelyPolygon]:
    #     """
    #     Erstellt ein Shapely-Polygon aus einer PointCloud2-Nachricht.

    #     Args:
    #         pointcloud_msg (PointCloud2): Die ROS PointCloud2 Nachricht.

    #     Returns:
    #         ShapelyPolygon: Das erstellte Shapely Polygon.
    #     """

    #     polygons = []
    #     # unique_labels = np.unique(points_with_labels[:, -1])
    #     pointclouds = self.cluster_data
    #     for pointcloud2 in pointclouds:

    #         points = []
    #         for p in PointCloud2.read_points(
    #             pointclouds, skip_nans=True, field_names=("x", "y")
    #         ):
    #             points.append((p[0], p[1]))  # Nimmt nur die x- und y-Koordinaten

    #         # Überprüfen, ob genügend Punkte für ein Polygon vorhanden sind
    #         if len(points) < 3:
    #             raise ValueError("Ein Polygon benötigt mindestens 3 Punkte.")

    #         # Optional: Erstelle den konvexen Hüllraum der Punkte
    #         # from scipy.spatial import ConvexHull
    #         # hull = ConvexHull(points)
    #         # hull_points = [points[i] for i in hull.vertices]

    #         # Erstelle das Shapely-Polygon
    #         polygon = ShapelyPolygon(points)
    #         polygons.append(polygon)
    #     return polygons

    #

    # def create_entities(polygons, velocities):
    # if polygons is None or len(polygons) == 0:
    #     # Handle cases where data is invalid or empty
    #     rospy.logwarn("No valid polygon data received.")
    #     return []

    # radar_entities = []
    # for polygon, velocity in zip(polygons, velocities):
    #     if not polygon.is_valid:
    #         rospy.logwarn("Skipping non-Polygon entity.")
    #         continue

    #     # Extrahiere das Zentrum des Polygons (Mittelpunktskoordinaten)
    #     centroid = polygon.centroid
    #     x_center = centroid.x
    #     y_center = centroid.y

    #     v = Vector2.new(x_center, y_center)  # 2D position in x-y plane
    #     transform = Transform2D.new_translation(v)

    #     # motion = Motion2D()
    #     flags = Flags(is_collider=True)
    #     e = Entity(
    #         confidence=1,
    #         priority=0.25,
    #         shape=Polygon(polygon.coords),
    #         transform=transform,
    #         timestamp=rospy.Time.now(),
    #         flags=flags,
    #         motion=velocity,
    #     )
    #     radar_entities.append(e)
    # return radar_entities

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
        hero_car = self.create_hero_entity()
        if hero_car is None:
            return

        entities = []
        entities.append(hero_car)

        # if self.lidar_marker_data is not None and self.get_param(
        #     "~enable_lidar_marker"
        # ):
        #     entities.extend(self.entities_from_lidar_marker())
        # if self.radar_marker_data is not None and self.get_param(
        #     "~enable_radar_marker"
        # ):
        #     entities.extend(self.entities_from_radar_marker())
        # if self.lidar_cluster_entities_data is not None and self.get_param(
        #     "~enable_lidar_cluster"
        # ):
        # entities.extend(self.create_entities_from_clusters())
        if self.radar_clustered_points_data is not None and self.get_param(
            "~enable_radar_cluster"
        ):
            entities.extend(self.create_entities_from_clusters())
        # if self.lidar_data is not None and self.get_param("~enable_raw_lidar_points"):
        #     entities.extend(self.entities_from_lidar())
        # Will be used when the new function for entity creation is implemented
        # if self.get_param("enable_vision_points"):
        #    entities.extend(self.entities_from_vision_points())
        stamp = rospy.get_rostime()
        map = Map(timestamp=stamp, entities=entities)
        msg = map.to_ros_msg()
        self.map_publisher.publish(msg)


if __name__ == "__main__":
    name = "mapping_data_integration"
    roscomp.init(name)
    node = MappingDataIntegrationNode(name)
    node.spin()
