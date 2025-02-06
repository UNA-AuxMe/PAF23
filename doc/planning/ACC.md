# ACC (Adaptive Cruise Control)

**Summary:** The ACC module is a ROS node responsible for adaptive speed control in an autonomous vehicle. It receives information about possible collisions, the current speed, the trajectory, and the speed limits. Based on this information, it calculates the desired speed and publishes it.

- [ROS Data Interface](#ros-data-interface)
  - [Published Topics](#published-topics)
  - [Subscribed Topics](#subscribed-topics)
- [Node Creation + Running Tests](#node-creation--running-tests)
- [Functionality](#functionality)

## ROS Data Interface

### Published Topics

This module publishes the following topics:

- `/paf/hero/acc_velocity`: The desired speed for the vehicle.
- `/paf/hero/current_wp`: The current waypoint.
- `/paf/hero/speed_limit`: The current speed limit.
- `/paf/hero/acc/debug_markers`: Markers that show debugging data e.g. which vehicle is recognized as leading vehicle.

### Subscribed Topics

This module subscribes to the following topics:

- `/paf/hero/mapping/init_data`: The map published by the intermediate layer.
- `/paf/hero/unstuck_flag`: A flag indicating whether the vehicle is stuck.
- `/paf/hero/unstuck_distance`: The distance the vehicle needs to travel to get unstuck.
- `/paf/hero/speed_limits_OpenDrive`: The speed limits from OpenDrive.
- `/paf/hero/trajectory_global`: The global trajectory of the vehicle (without overtakes etc).
- `/paf/hero/trajectory`: The trajectory of the vehicle including overtakes etc.
- `/paf/hero/current_pos`: The current position of the vehicle.
- `/paf/hero/current_heading`: The current heading of the vehicle.

## Node Creation + Running Tests

To run this node insert the following statement in the [planning.launch](../../code/planning/launch/planning.launch) file:

```xml
<node pkg="planning" type="ACC.py" name="ACC" output="screen">
    <param name="role_name" value="hero" />
    <param name="control_loop_rate" value="0.3" />
</node>
```

## Functionality

Each time the map from the intermediate layer is received, the ACC triggers the function `update_velocity` where the main logic of the ACC can be found. The most important steps are:

- Definition of a trajectory mask and a rectangle in front of the car that is used to set the area in which a leading vehicle is searched. The rectangle is able to detect possible collisions that are close to the car while the trajectory mask checks the area at a higher distance from the car.
- Calculation of the leading vehicle based on the trajectory mask and the rectangle.
- Calculation of the desired speed using the function `calculate_velocity_based_on_lead`.
- Publishing of the markers to visualize the masks and the chosen leading vehicle.
  