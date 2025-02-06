from dataclasses import dataclass, field
from typing import List, Optional, Callable, Literal, Tuple

import shapely
from shapely import STRtree, LineString
import numpy as np
import numpy.typing as npt

from genpy.rostime import Time
from std_msgs.msg import Header
from mapping_common.transform import Transform2D, Vector2
from mapping_common.entity import Entity, FlagFilter, ShapelyEntity
from mapping_common.shape import Rectangle
import mapping_common.mask

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

    def get_lane_y_axis_intersections(self, direction: str = "left") -> dict:
        """calculates the intersections of the lanemarks in lane_pos direction

        Args:
            direction (str): lanemarks on "left", "right" or "both" will be checked.
            Other inputs will be ignored

        Returns:
            dict{uuid, coordinate}: dictionary with uuid of lanemark as keys
            and coordinates as according entries
        """
        if direction == "left":
            y_axis_line = LineString([[0, 0], [0, 8]])
        elif direction == "right":
            y_axis_line = LineString([[0, 0], [0, -8]])
        elif direction == "both":
            y_axis_line = LineString([[0, -8], [0, 8]])
        else:
            return {}
        intersection = []
        lanemark_filter = FlagFilter(is_lanemark=True)
        lanemarks = self.filtered(lanemark_filter)
        intersections = {}
        for lanemark in lanemarks:
            intersection = lanemark.shape.to_shapely(lanemark.transform).intersection(
                y_axis_line
            )
            if not intersection.is_empty:
                intersections[lanemark.uuid] = [
                    intersection.centroid.x,
                    intersection.centroid.y,
                ]
        return intersections

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
    _tree_polys: List[shapely.Polygon]
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
        self._tree_polys = [e.poly for e in self.filtered_entities]
        self._str_tree = STRtree(geoms=self._tree_polys)

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

        **IMPORTANT: The query might include false positives,
        because the STRtree seems to only roughly check for intersections.
        It includes all entities that MIGHT intersect with geo, but they
        still need to be checked manually with shapely.intersects(a, b) afterwards!**

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

    def query_self(
        self,
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
    ) -> List[Tuple[ShapelyEntity, ShapelyEntity]]:
        """Queries interactions between the shapes inside this tree.

        Removes any self intersections and duplicate interaction pairs.

        Args:
            predicate (Optional[ Literal[ &quot;intersects&quot;, &quot;within&quot;,
                &quot;contains&quot;, &quot;overlaps&quot;, &quot;crosses&quot;,
                &quot;touches&quot;, &quot;covers&quot;, &quot;covered_by&quot;,
                &quot;contains_properly&quot;, &quot;dwithin&quot;, ] ], optional):
                Which interaction to filter for. Defaults to None.
            distance (Optional[float], optional):
                Must only be set for the &quot;dwithin&quot; predicate
                and controls its distance. Defaults to None.

        Returns:
            List[Tuple[ShapelyEntity, ShapelyEntity]]:
                Tuples of interacting entity pairs
        """
        query: npt.NDArray[np.int64] = self._str_tree.query(
            self._tree_polys, predicate=predicate, distance=distance
        )
        # Remove invalid pairs like [0, 0] and duplicates like [1, 4]<->[4, 1]
        filter = query[0] < query[1]
        transposed = np.transpose(query)
        deduplicated = transposed[filter]

        results: List[Tuple[ShapelyEntity, ShapelyEntity]] = []
        for pair in deduplicated:
            entity_pair = (
                self.filtered_entities[pair[0]],
                self.filtered_entities[pair[1]],
            )
            results.append(entity_pair)
        return results

    def get_entity_in_front_or_back(self, in_front=True) -> Optional[ShapelyEntity]:
        """Returns the first entity in front or back based on in_front

        Projects a polygon to simulate the road
        Calculates the nearest entity on that polygon
        Returns:
            Optional[Entity]: Entity in front
        This could be extended with a curved polygon
        if curved roads become a problem in the future
        """
        length = 80 if in_front else -80
        # a coverage width of 1.4 meters should cover all cars and obstacles in our way
        road_area = mapping_common.mask.project_plane(length, 1.4)
        road_entities = self.query(road_area)

        if len(road_entities) > 0:
            if in_front:
                return min(
                    road_entities,
                    key=lambda e: e.entity.transform.translation().x(),
                )
            else:
                return max(
                    road_entities,
                    key=lambda e: e.entity.transform.translation().x(),
                )
        else:
            return None

    def is_lane_free(
        self,
        right_lane: bool = False,
        lane_length: float = 20.0,
        lane_transform: float = 0.0,
    ) -> bool:
        """Returns if a lane left or right of our car is free.
        Right now, a rectangle shape of length lane_length placed
        on the left or right side of the car with a transformation of lane_transform
        in front or back. Checks if this rectangle lane box intersects with any
        relevant entities.

        Idea for later: using lanemark detection and if data is realiable form a
        polygon for the lane within the detected lanes.

        Parameters:
        - right_lane (bool): If true, checks the right lane instead of the left lane
        - lane_length (float): Sets the lane length that should be checked, in meters.
          Default value is 20 meters.
        - lane_transform (float): Transforms the checked lane box to the front (>0) or
          back (<0) of the car, in meters. Default is 0 meter so the lane box originates
           from the car position -> same distance to the front and rear get checked
        Returns:
            bool: lane is free / not free
        """
        # checks which lane should be checked and set the multiplier for
        # the lane entity translation(>0 = left from car)
        lane_pos = 1
        if right_lane:
            lane_pos = -1

        # lane length cannot be negative, as no rectangle with negative dimension exists
        if lane_length < 0:
            raise ValueError("Lane length cannot take a negative value.")

        lane_box_shape = Rectangle(
            length=lane_length,
            width=1.5,
            offset=Transform2D.new_translation(
                Vector2.new(lane_transform, lane_pos * 2.5)
            ),
        )

        # converts lane box Rectangle to a shapely Polygon
        lane_box_shapely = lane_box_shape.to_shapely(Transform2D.identity())

        # creates intersection list of lane box with map entities
        lane_box_intersection_entities = self.query(
            geo=lane_box_shapely, predicate="intersects"
        )

        # if list with lane box intersection is empty --> lane is free
        if not lane_box_intersection_entities:
            return True
        return False

    def get_nearest_entity(
        self,
        mask: shapely.Geometry,
        reference: ShapelyEntity,
        min_coverage_percent: float = 0.0,
        min_coverage_area: float = 0.0,
    ) -> Optional[Tuple[ShapelyEntity, float]]:
        """Returns the nearest entity to *reference* that have
        at least coverage % or area in the mask geometry.

        Args:
            mask (shapely.Geometry): A Shapely Geometry object representing
                the target area.
            reference (ShapelyEntity): Entity for the nearest distance calculation
            min_coverage_percent (float, optional): Defaults to 0.0.
            min_coverage_area (float, optional): Defaults to 0.0.

        Returns:
            Optional[Tuple[ShapelyEntity, float]]:
                A Tuple of (Nearest Entity, Distance to reference)
        """

        query = self.get_overlapping_entities(
            mask,
            min_coverage_percent=min_coverage_percent,
            min_coverage_area=min_coverage_area,
        )

        if len(query) <= 0:
            return None

        query_distances = map(lambda e: (e, reference.get_distance_to(e)), query)
        return min(query_distances, key=lambda e: e[1])

    def get_overlapping_entities(
        self,
        mask: shapely.Geometry,
        min_coverage_percent: float = 0.0,
        min_coverage_area: float = 0.0,
    ) -> List[ShapelyEntity]:
        """Returns a list of entities that have at least coverage % or area in the
        mask geometry.

        Args:
            mask (shapely.Geometry): A Shapely Geometry object representing
                the target area.
            min_coverage_percent (float, optional): Defaults to 0.0.
            min_coverage_area (float, optional): Defaults to 0.0.

        Returns:
            List[ShapelyEntity]: A list of entities that have at least
                coverage % or area in the polygon
        """

        query = self.query(mask)

        collision_entities = []

        for ent in query:
            shape = ent.poly
            shape_area: float = shape.area

            if min_coverage_percent <= 0.0 and min_coverage_area <= 0.0:
                # Check for intersection, because query only does it roughly
                if shapely.intersects(mask, shape):
                    collision_entities.append(ent)
                continue

            # Calculate intersection area
            shapely_intersection = shapely.intersection(mask, shape)
            intersection_area: float = shapely_intersection.area
            if (
                intersection_area / shape_area >= min_coverage_percent
                or intersection_area >= min_coverage_area
            ):
                collision_entities.append(ent)

        return collision_entities


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


def build_global_hero_transform(x: float, y: float, heading: float) -> Transform2D:
    """Builds a Transform2D representing the global position of the hero
    based on its coordinates and heading

    Args:
        x (float): Global hero x coordinate
        y (float): Global hero y coordinate
        heading (float): hero heading

    Returns:
        Transform2D: Global hero Transform2D
    """
    translation = Vector2.new(x, y)
    return Transform2D.new_rotation_translation(heading, translation)


def lane_free_filter() -> FlagFilter:
    """Creates the default flag filter for the lane free check

    Returns:
        FlagFilter
    """
    filter = FlagFilter()
    filter.is_collider = True
    filter.is_hero = False
    filter.is_lanemark = False
    filter.is_ignored = False
    return filter
