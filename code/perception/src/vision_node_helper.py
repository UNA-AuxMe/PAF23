from typing import List, Union


# Carla-Farben
carla_colors = [
    [0, 0, 0],  # 0: None
    [70, 70, 70],  # 1: Buildings
    [190, 153, 153],  # 2: Fences
    [72, 0, 90],  # 3: Other
    [220, 20, 60],  # 4: Pedestrians
    [153, 153, 153],  # 5: Poles
    [157, 234, 50],  # 6: RoadLines
    [128, 64, 128],  # 7: Roads
    [244, 35, 232],  # 8: Sidewalks
    [107, 142, 35],  # 9: Vegetation
    [0, 0, 255],  # 10: Vehicles
    [102, 102, 156],  # 11: Walls
    [220, 220, 0],  # 12: TrafficSigns
]

carla_colors_map = {
    0: [0, 0, 0],  # 0: None
    1: [70, 70, 70],  # 1: Buildings
    2: [190, 153, 153],  # 2: Fences
    3: [72, 0, 90],  # 3: Other
    4: [220, 20, 60],  # 4: Pedestrians
    5: [153, 153, 153],  # 5: Poles
    6: [157, 234, 50],  # 6: RoadLines
    7: [128, 64, 128],  # 7: Roads
    8: [244, 35, 232],  # 8: Sidewalks
    9: [107, 142, 35],  # 9: Vegetation
    10: [0, 0, 255],  # 10: Vehicles
    11: [102, 102, 156],  # 11: Walls
    12: [220, 220, 0],  # 12: TrafficSigns
}

# COCO-Klassen → Carla-Klassen Mapping
coco_to_carla = [
    4,  # 0: Person -> Pedestrians
    10,  # 1: Bicycle -> Vehicles
    10,  # 2: Car -> Vehicles
    10,  # 3: Motorbike -> Vehicles
    10,  # 4: Airplane -> Vehicles
    10,  # 5: Bus -> Vehicles
    10,  # 6: Train -> Vehicles
    10,  # 7: Truck -> Vehicles
    10,  # 8: Boat -> Vehicles
    12,  # 9: Traffic Light -> TrafficSigns
    3,  # 10: Fire Hydrant -> Other
    12,  # 11: Stop Sign -> TrafficSigns
    3,  # 12: Parking Meter -> Other
    3,  # 13: Bench -> Other
    3,  # 14: Bird -> Other
    3,  # 15: Cat -> Other
    3,  # 16: Dog -> Other
    3,  # 17: Horse -> Other
    3,  # 18: Sheep -> Other
    3,  # 19: Cow -> Other
    3,  # 20: Elephant -> Other
    3,  # 21: Bear -> Other
    3,  # 22: Zebra -> Other
    3,  # 23: Giraffe -> Other
    3,  # 24: Backpack -> Other
    3,  # 25: Umbrella -> Other
    3,  # 26: Handbag -> Other
    3,  # 27: Tie -> Other
    3,  # 28: Suitcase -> Other
    3,  # 29: Frisbee -> Other
    3,  # 30: Skis -> Other
    3,  # 31: Snowboard -> Other
    3,  # 32: Sports Ball -> Other
    3,  # 33: Kite -> Other
    3,  # 34: Baseball Bat -> Other
    3,  # 35: Baseball Glove -> Other
    3,  # 36: Skateboard -> Other
    3,  # 37: Surfboard -> Other
    3,  # 38: Tennis Racket -> Other
    3,  # 39: Bottle -> Other
    3,  # 40: Wine Glass -> Other
    3,  # 41: Cup -> Other
    3,  # 42: Fork -> Other
    3,  # 43: Knife -> Other
    3,  # 44: Spoon -> Other
    3,  # 45: Bowl -> Other
    3,  # 46: Banana -> Other
    3,  # 47: Apple -> Other
    3,  # 48: Sandwich -> Other
    3,  # 49: Orange -> Other
    3,  # 50: Broccoli -> Other
    3,  # 51: Carrot -> Other
    3,  # 52: Hot Dog -> Other
    3,  # 53: Pizza -> Other
    3,  # 54: Donut -> Other
    3,  # 55: Cake -> Other
    3,  # 56: Chair -> Other
    3,  # 57: Couch -> Other
    3,  # 58: Potted Plant -> Other
    3,  # 59: Bed -> Other
    3,  # 60: Dining Table -> Other
    3,  # 61: Toilet -> Other
    3,  # 62: TV -> Other
    3,  # 63: Laptop -> Other
    3,  # 64: Mouse -> Other
    3,  # 65: Remote -> Other
    3,  # 66: Keyboard -> Other
    3,  # 67: Cell Phone -> Other
    3,  # 68: Microwave -> Other
    3,  # 69: Oven -> Other
    3,  # 70: Toaster -> Other
    3,  # 71: Sink -> Other
    3,  # 72: Refrigerator -> Other
    3,  # 73: Book -> Other
    3,  # 74: Clock -> Other
    3,  # 75: Vase -> Other
    3,  # 76: Scissors -> Other
    3,  # 77: Teddy Bear -> Other
    3,  # 78: Hair Drier -> Other
    3,  # 79: Toothbrush -> Other
]

coco_to_carla_map = {
    0: 4,  # 0: Person -> Pedestrians
    1: 10,  # 1: Bicycle -> Vehicles
    2: 10,  # 2: Car -> Vehicles
    3: 10,  # 3: Motorbike -> Vehicles
    4: 10,  # 4: Airplane -> Vehicles
    5: 10,  # 5: Bus -> Vehicles
    6: 10,  # 6: Train -> Vehicles
    7: 10,  # 7: Truck -> Vehicles
    8: 10,  # 8: Boat -> Vehicles
    9: 12,  # 9: Traffic Light -> TrafficSigns
    10: 3,  # 10: Fire Hydrant -> Other
    11: 12,  # 11: Stop Sign -> TrafficSigns
    12: 3,  # 12: Parking Meter -> Other
    13: 3,  # 13: Bench -> Other
    14: 3,  # 14: Bird -> Other
    15: 3,  # 15: Cat -> Other
    16: 3,  # 16: Dog -> Other
    17: 3,  # 17: Horse -> Other
    18: 3,  # 18: Sheep -> Other
    19: 3,  # 19: Cow -> Other
    20: 3,  # 20: Elephant -> Other
    21: 3,  # 21: Bear -> Other
    22: 3,  # 22: Zebra -> Other
    23: 3,  # 23: Giraffe -> Other
    24: 3,  # 24: Backpack -> Other
    25: 3,  # 25: Umbrella -> Other
    26: 3,  # 26: Handbag -> Other
    27: 3,  # 27: Tie -> Other
    28: 3,  # 28: Suitcase -> Other
    29: 3,  # 29: Frisbee -> Other
    30: 3,  # 30: Skis -> Other
    31: 3,  # 31: Snowboard -> Other
    32: 3,  # 32: Sports Ball -> Other
    33: 3,  # 33: Kite -> Other
    34: 3,  # 34: Baseball Bat -> Other
    35: 3,  # 35: Baseball Glove -> Other
    36: 3,  # 36: Skateboard -> Other
    37: 3,  # 37: Surfboard -> Other
    38: 3,  # 38: Tennis Racket -> Other
    39: 3,  # 39: Bottle -> Other
    40: 3,  # 40: Wine Glass -> Other
    41: 3,  # 41: Cup -> Other
    42: 3,  # 42: Fork -> Other
    43: 3,  # 43: Knife -> Other
    44: 3,  # 44: Spoon -> Other
    45: 3,  # 45: Bowl -> Other
    46: 3,  # 46: Banana -> Other
    47: 3,  # 47: Apple -> Other
    48: 3,  # 48: Sandwich -> Other
    49: 3,  # 49: Orange -> Other
    50: 3,  # 50: Broccoli -> Other
    51: 3,  # 51: Carrot -> Other
    52: 3,  # 52: Hot Dog -> Other
    53: 3,  # 53: Pizza -> Other
    54: 3,  # 54: Donut -> Other
    55: 3,  # 55: Cake -> Other
    56: 3,  # 56: Chair -> Other
    57: 3,  # 57: Couch -> Other
    58: 3,  # 58: Potted Plant -> Other
    59: 3,  # 59: Bed -> Other
    60: 3,  # 60: Dining Table -> Other
    61: 3,  # 61: Toilet -> Other
    62: 3,  # 62: TV -> Other
    63: 3,  # 63: Laptop -> Other
    64: 3,  # 64: Mouse -> Other
    65: 3,  # 65: Remote -> Other
    66: 3,  # 66: Keyboard -> Other
    67: 3,  # 67: Cell Phone -> Other
    68: 3,  # 68: Microwave -> Other
    69: 3,  # 69: Oven -> Other
    70: 3,  # 70: Toaster -> Other
    71: 3,  # 71: Sink -> Other
    72: 3,  # 72: Refrigerator -> Other
    73: 3,  # 73: Book -> Other
    74: 3,  # 74: Clock -> Other
    75: 3,  # 75: Vase -> Other
    76: 3,  # 76: Scissors -> Other
    77: 3,  # 77: Teddy Bear -> Other
    78: 3,  # 78: Hair Drier -> Other
    79: 3,  # 79: Toothbrush -> Other
}

COCO_CLASS_COUNT = 80


def get_carla_color(coco_class: Union[int, float]) -> List[int]:
    """Get the Carla color for a given COCO class.
    Args:
    coco_class: COCO class index (0-79)

    Returns:
    RGB color values for the corresponding Carla class

    Raises:
    ValueError: If coco_class is out of valid range
    """
    coco_idx = int(coco_class)
    if not 0 <= coco_idx < COCO_CLASS_COUNT:
        raise ValueError(f"Invalid COCO class index: {coco_idx}")
    carla_class = coco_to_carla[coco_idx]
    return carla_colors[carla_class]
