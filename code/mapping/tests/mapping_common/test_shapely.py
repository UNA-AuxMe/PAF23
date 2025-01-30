from typing import List

from mapping_common import shape, entity
from mapping_common.map import Map
from mapping_common.transform import Transform2D, Vector2

import test_entity
import test_shape


def test_circle_shapely():
    offset_transl = Vector2.new(1.0, 0.0)
    offset = Transform2D.new_translation(offset_transl)
    s = shape.Circle(3.0, offset)
    poly = s.to_shapely(Transform2D.identity())
    bounds = poly.bounds
    assert bounds == (-2.0, -3.0, 4.0, 3.0)


def test_rectangle_shapely():
    s = shape.Rectangle(4.0, 1.5)
    poly = s.to_shapely(Transform2D.identity())
    bounds = poly.bounds
    assert bounds == (-2.0, -0.75, 2.0, 0.75)


def test_polygon_to_shapely():
    s = test_shape.get_polygon()
    poly = s.to_shapely(Transform2D.identity())
    bounds = poly.bounds
    assert bounds == (1.0, 2.0, 5.0, 6.0)


def test_polygon_from_shapely():
    s = test_shape.get_polygon()
    poly = s.to_shapely(Transform2D.identity())
    s_conv = shape.Polygon.from_shapely(poly)

    assert s == s_conv


def test_polygon_to_from_shapely_centered():
    s = test_shape.get_polygon()
    poly = s.to_shapely(Transform2D.identity())
    s_conv = shape.Polygon.from_shapely(poly, make_centered=True)
    poly_conv = s_conv.to_shapely(Transform2D.identity())

    assert poly.equals_exact(poly_conv, tolerance=0.0)
    assert s_conv.offset.translation() == Vector2.new(3.0, 4.0)


def test_car_shapely():
    car = test_entity.get_car()
    transl = Vector2.new(2.0, -3.0)
    transform = Transform2D.new_translation(transl)
    car.transform = transform
    s = car.to_shapely()
    bounds = s.poly.bounds
    assert bounds == (1.5, -4.0, 2.5, -2.0)


def get_test_entities() -> List[entity.Entity]:
    translations = [
        Vector2.new(2.0, -3.0),
        Vector2.new(1.0, 5.0),
        Vector2.new(2.0, 1.0),
    ]
    entities = []
    for t in translations:
        car = test_entity.get_car()
        transform = Transform2D.new_translation(t)
        car.transform = transform
        entities.append(car)

    lane = entity.Lanemarking(
        style=entity.Lanemarking.Style.SOLID,
        confidence=1.0,
        priority=1.0,
        shape=shape.Rectangle(4.0, 0.2),
        transform=Transform2D.identity(),
        flags=entity.Flags(is_lanemark=True),
    )
    entities.append(lane)
    return entities


def test_map_tree_nearest():
    entities = get_test_entities()

    map = Map(entities=entities)
    tree = map.build_tree(f=entity.FlagFilter(is_collider=True))
    test_shape = shape.Circle(0.5).to_shapely(Transform2D.identity())
    nearest = tree.nearest(test_shape)

    assert nearest is not None
    assert nearest.entity == entities[2]


def test_map_tree_query():
    entities = get_test_entities()

    map = Map(entities=entities)
    tree = map.build_tree(f=entity.FlagFilter(is_collider=True))
    test_shape = shape.Rectangle(3.0, 1.0).to_shapely(Transform2D.identity())
    query = tree.query(test_shape)

    assert len(query) == 1
    assert query[0].entity == entities[2]


def test_map_tree_query_nearest():
    entities = get_test_entities()

    map = Map(entities=entities)
    tree = map.build_tree(f=entity.FlagFilter(is_collider=True))
    test_shape = shape.Rectangle(1.0, 1.0).to_shapely(Transform2D.identity())
    query = tree.query_nearest(test_shape)

    assert len(query) == 1
    assert query[0][0].entity == entities[2]
    assert query[0][1] == 1.0


def test_map_tree_query_self():
    car = test_entity.get_car()
    car_transform = Transform2D.identity()
    car_shape = shape.Rectangle(3.0, 3.0)
    car.transform = car_transform
    car.shape = car_shape
    entities = [car]
    entities += get_test_entities()

    map = Map(entities=entities)
    tree = map.build_tree(f=entity.FlagFilter(is_collider=True))

    query = tree.query_self()

    assert len(query) == 1
    assert query[0][0].entity == entities[0]
    assert query[0][1].entity == entities[3]
