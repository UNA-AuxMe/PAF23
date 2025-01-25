from dataclasses import dataclass, field
from typing import List, Optional, Callable, Literal, Tuple

from ros_compatibility.node import CompatibleNode

import rasterio.features
import shapely
from shapely import STRtree
import numpy as np
import numpy.typing as npt

from genpy.rostime import Time
from std_msgs.msg import Header
from mapping_common import entity

from mapping_common.entity import Entity, FlagFilter, ShapelyEntity

from shapely.geometry import Polygon, LineString
from shapely import MultiPolygon
from cairosvg import svg2png
import rasterio
import cv2
from cv2 import line

from mapping import msg


@dataclass
class Map:
    """2 dimensional map for the intermediate layer

    General information:
    - 2D top-down map. The height(z) dimension is mostly useless
      for collision detection and path planning
    - The map is based on the local car position and is only built using sensor data.
      No global positioning via GPS or similar is used.
    - The map (0/0) is the center of the hero car
    - All transformations to entities are relative to
      the hero car's coordinate system (position/heading)
    - The map's x-axis is aligned with the heading of the hero car
    - The map's y-axis points to the left of the hero car
    - Coordinate system is a right-hand system like tf2 (can be visualized in RViz)
    - The map might include the hero car as the first entity in entities
    """

    timestamp: Time = Time()
    """The timestamp this map was created at.

    Should be the time when this map was initially sent off
    from the mapping_data_integration node.

    This timestamp is also the "freshest" compared to
    the timestamps of all entities included in the map
    """
    entities: List[Entity] = field(default_factory=list)
    """The entities this map consists out of

    Note that this list might also include the hero car (as first element of this list)
    """

    def hero(self) -> Optional[Entity]:
        """Returns the entity of the hero car if it is the first element of the map

        Returns:
            Optional[Entity]: Entity of the hero car
        """
        if len(self.entities) <= 0:
            return None
        hero = self.entities[0]
        if not hero.flags._is_hero:
            return None
        return hero

    def entities_without_hero(self) -> List[Entity]:
        """Returns the entities without the hero car

        Only checks if the first entity is_hero

        Returns:
            List[Entity]: Entities without the hero car
        """
        if self.hero() is not None:
            return self.entities[1:]
        return self.entities

    def get_entity_in_front_or_back(self, in_front=True) -> Optional[Entity]:
        """Returns the first entity in front or back based on in_front

        Projects a polygon to simulate the road
        Calculates the nearest entity on that polygon
        Returns:
            Optional[Entity]: Entity in front
        This could be extended with a curved polygon
        if curved roads become a problem in the future
        """
        length = 80 if in_front else -80
        tree = self.build_tree(f=entity.FlagFilter(is_collider=True, is_hero=False))
        # a coverage width of 1.4 meters should cover all cars and obstacles in our way
        road_area = self.project_plane((0, -0.7), length, 1.4)
        road_entities = tree.query(road_area)

        if len(road_entities) > 0:
            if in_front:
                return min(
                    road_entities,
                    key=lambda e: e.entity.transform.translation().x(),
                ).entity
            else:
                return max(
                    road_entities,
                    key=lambda e: e.entity.transform.translation().x(),
                ).entity
        else:
            return None

    def project_plane(self, start_point, size_x, size_y):
        """Projects a rectangular plane starting from start point
        forward in the x-direction.

        Parameters:
        - start_point(float, float): Starting point tuple from which
        the rectangle is constructed
        - size_x (float): Length of the plane along the x-axis.
        - size_y (float): Width of the plane along the y-axis.

        Returns:
        - Polygon: A Shapely Polygon representing the plane."""

        x, y = start_point

        points = [
            (x, y),
            (x + size_x, y),
            (x + size_x, y + size_y),
            (x, y + size_y),
            (x, y),
        ]

        return Polygon(points)

    def curve_to_polygon(self, points, width):
        """Creates a polygon with a specified width around a given curve.

        Parameters:
        - points (list of tuple): A list of (x, y) coordinates representing the curve.
        - width (float): The width of the polygon along the curve.

        Returns:
        - Polygon: A Shapely Polygon representing the widened curve."""

        if len(points) < 2:
            raise ValueError("At least two points are required to define a curve.")
        if width <= 0:
            raise ValueError("Width must be a positive value.")

        # Create a LineString from the given points
        curve = LineString(points)

        # Create a buffer around the curve to form a polygon with the given width
        polygon = curve.buffer(width / 2, cap_style="round", join_style="round")

        return polygon

    def get_entities_with_coverage(self, polygon, entities: List[Entity], coverage):
        """Returns a list of entities that have at least coverage % in the
        given polygon.

        Parameters:
        - polygon (Polygon): A Shapely Polygon object representing the target area.
        - entities (list): A list of entities each having a Shapely shape.

        Returns:
        - list: A list of entities that have at least coverage % in the polygon."""

        collision_entities = []

        for ent in entities:
            shape = ent.to_shapely().poly

            # Calculate intersection area
            intersection = polygon.intersection(shape)
            if intersection.area / shape.area >= coverage:
                collision_entities.append(ent)

        return collision_entities

    def build_tree(
        self,
        f: Optional[FlagFilter] = None,
        filter_fn: Optional[Callable[[Entity], bool]] = None,
    ) -> "MapTree":
        """Creates a filtered MapTree

        **IMPORTANT**: The map
        MUST NOT BE MODIFIED WHILE USING THE TREE,
        otherwise results will be invalid or crash

         Useful for for quickly calculating which entities of a map are
        the nearest or (intersect, touch, etc.) with a given geometry

        Args:
            f (Optional[FlagFilter], optional): Filtering with FlagFilter.
                Defaults to None.
            filter_fn (Optional[Callable[[Entity], bool]], optional):
                Filtering with function/lambda. Defaults to None.

        Returns:
            MapTree
        """
        return MapTree(self, f, filter_fn)

    def filtered(
        self,
        f: Optional[FlagFilter] = None,
        filter_fn: Optional[Callable[[Entity], bool]] = None,
    ) -> List[Entity]:
        """Filters self.entities

        Args:
            f (Optional[FlagFilter], optional): Filtering with FlagFilter.
                Defaults to None.
            filter_fn (Optional[Callable[[Entity], bool]], optional):
                Filtering with function/lambda. Defaults to None.

        Returns:
            List[Entity]: List of entities both filters were matching for
        """
        return [e for e in self.entities if _entity_matches_filter(e, f, filter_fn)]

    def to_multi_poly_array(
        self,
        array_shape: Tuple[int, int],
        resolution_scale: int,
        parent_node: CompatibleNode,
    ):
        poly_array: np.ndarray = np.zeros(array_shape, dtype=np.uint8)
        for entity_obj in self.entities_without_hero():
            coords = [coords for coords in entity_obj.to_shapely().poly.exterior.coords]
            for i in range(len(coords) - 1):
                pt1: Tuple[int, int] = (
                    int((array_shape[0] - coords[i][0]) * resolution_scale),
                    int((array_shape[1] - coords[i][1]) * resolution_scale)
                    + array_shape[1] // 2,
                )
                pt2: Tuple[int, int] = (
                    int((array_shape[0] - coords[i + 1][0]) * resolution_scale),
                    int((array_shape[1] - coords[i + 1][1]) * resolution_scale)
                    + array_shape[1] // 2,  # shift 0 to the middle of the image
                )

                poly_array = line(
                    img=poly_array, pt1=pt1, pt2=pt2, color=255, thickness=1
                )

        cv2.imwrite("/workspace/code/contourplot.png", poly_array)
        parent_node.loginfo("contourplot saved")
        return

    @staticmethod
    def from_ros_msg(m: msg.Map) -> "Map":
        entities = list(map(lambda e: Entity.from_ros_msg(e), m.entities))
        return Map(timestamp=m.header.stamp, entities=entities)

    def to_ros_msg(self) -> msg.Map:
        entities = list(map(lambda e: e.to_ros_msg(), self.entities))
        header = Header(stamp=self.timestamp)
        return msg.Map(header=header, entities=entities)


@dataclass(init=False)
class MapTree:
    """An acceleration structure around the shapely.STRtree

    **IMPORTANT**: The map this tree was created with
    MUST NOT BE MODIFIED WHILE USING THE TREE,
    otherwise results will be invalid or crash

    Useful for for quickly calculating which entities of a map are
    the nearest or (intersect, touch, etc.) with a given geometry

    The map's entities can optionally be filtered upon tree creation
    """

    _str_tree: STRtree
    filtered_entities: List[ShapelyEntity]
    """Only the entities of this tree that weren't filtered out from the map.

    Also includes their shapely.Polygon
    """
    map: Map
    """The map this tree was created with
    """

    def __init__(
        self,
        map: Map,
        f: Optional[FlagFilter] = None,
        filter_fn: Optional[Callable[[Entity], bool]] = None,
    ):
        """Creates a shapely.STRtree based on the given map and filtering

        Both filtering methods can be combined.
        Both filters need to match for the entity to match.

        Args:
            map (Map)
            f (Optional[FlagFilter], optional): Filtering with FlagFilter.
                Defaults to None.
            filter_fn (Optional[Callable[[Entity], bool]], optional):
                Filtering with function/lambda. Defaults to None.
        """
        self.map = map
        self.filtered_entities = [e.to_shapely() for e in map.filtered(f, filter_fn)]
        self._str_tree = STRtree(geoms=[e.poly for e in self.filtered_entities])

    def _idxs_to_entity(self, idxs: npt.NDArray) -> List[ShapelyEntity]:
        return [self.filtered_entities[i] for i in idxs]

    def nearest(self, geo: shapely.Geometry) -> Optional[ShapelyEntity]:
        """Returns the nearest Entity inside the tree based on geo

        More information here:
        https://shapely.readthedocs.io/en/stable/strtree.html#shapely.STRtree.nearest

        Args:
            geo (shapely.Geometry): Geometry to calculate the nearest entity to

        Returns:
            Optional[ShapelyEntity]: If the tree is empty, will return None.
                Otherwise will return the nearest Entity
        """
        idx = self._str_tree.nearest(geo)
        if idx is None:
            return None
        return self.filtered_entities[idx]

    def query(
        self,
        geo: shapely.Geometry,
        predicate: Optional[
            Literal[
                "intersects",
                "within",
                "contains",
                "overlaps",
                "crosses",
                "touches",
                "covers",
                "covered_by",
                "contains_properly",
                "dwithin",
            ]
        ] = None,
        distance: Optional[float] = None,
    ) -> List[ShapelyEntity]:
        """Calculates which entities interact with *geo*

        More information here:
        https://shapely.readthedocs.io/en/stable/strtree.html#shapely.STRtree.query

        Args:
            geo (shapely.Geometry): The geometry to query with
            predicate (Optional[ Literal[ &quot;intersects&quot;, &quot;within&quot;,
            &quot;contains&quot;, &quot;overlaps&quot;, &quot;crosses&quot;,
            &quot;touches&quot;, &quot;covers&quot;, &quot;covered_by&quot;,
            &quot;contains_properly&quot;, &quot;dwithin&quot;, ] ], optional):
            Which interaction to filter for. Defaults to None.
            distance (Optional[float], optional):
            Must only be set for the &quot;dwithin&quot; predicate
            and controls its distance. Defaults to None.

        Returns:
            List[ShapelyEntity]: The List of queried entities in the tree
        """
        idxs: npt.NDArray[np.int64] = self._str_tree.query(
            geo, predicate=predicate, distance=distance
        )
        return self._idxs_to_entity(idxs)

    def query_nearest(
        self,
        geo: shapely.Geometry,
        max_distance: Optional[float] = None,
        exclusive: bool = False,
        all_matches: bool = True,
    ) -> List[Tuple[ShapelyEntity, float]]:
        """Queries the distance from *geo* to its nearest entities in the tree

        More information here:
        https://shapely.readthedocs.io/en/stable/strtree.html#shapely.STRtree.query_nearest

        Args:
            geo (shapely.Geometry): The geometry to query with
            max_distance (Optional[float], optional): Maximum distance for the query.
                Defaults to None.
            exclusive (bool, optional): If True, ignores entities
            with a shape equal to geo. Defaults to False.
            all_matches (bool, optional): If True, all equidistant and intersected
            geometries will be returned. If False only the nearest. Defaults to True.

        Returns:
            List[Tuple[ShapelyEntity, float]]: A List of Tuples.
            Each contains a queried entity and its distance to geo
        """
        query: Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]] = (
            self._str_tree.query_nearest(
                geo,
                max_distance=max_distance,
                return_distance=True,
                exclusive=exclusive,
                all_matches=all_matches,
            )
        )
        result: List[Tuple[ShapelyEntity, float]] = []
        for idx, distance in zip(query[0], query[1]):
            result.append((self.filtered_entities[idx], distance))
        return result


def _entity_matches_filter(
    e: Entity,
    f: Optional[FlagFilter] = None,
    filter_fn: Optional[Callable[[Entity], bool]] = None,
) -> bool:
    if f is not None:
        if not e.matches_filter(f):
            return False
    if filter_fn is not None:
        if not filter_fn(e):
            return False
    return True
