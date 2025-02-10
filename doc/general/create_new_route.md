# Create a new route

**Summary:** This page shows how to create a new route with scenarios that can be loaded later as a checkpoint or used for an automated test.

- [Scenario types](#scenario-types)
- [Create new route](#create-new-route)

## Scenario types

The carla simulator has different scenario types you can work with, which are listed down below. The corresponding python files can be found [here](https://github.com/carla-simulator/scenario_runner/tree/master/srunner/scenarios).

|                             Name |                                                                                                                                                                                                                                                                                                                                                                             Description                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| actor_flow | Scenarios in which another (opposite) vehicle 'illegally' takes priority, e.g. by running a red traffic light. |
|blocked_intersection | Scenario with low visibility, the ego performs a turn only to find out that the end is blocked by another vehicle. |
| change_lane | Change lane scenario: The scenario realizes a driving behavior, in which the user-controlled ego vehicle follows a fast driving car on the highway. There's a slow car driving in great distance to the fast vehicle. At one point the fast vehicle is changing the lane to overtake a slow car, which is driving on the same lane. The ego vehicle doesn't "see" the slow car before the lane change of the fast car, therefore it hast to react fast to avoid an collision. There are two options to avoid an accident: The ego vehicle adjusts its velocity or changes the lane as well. |
| construction_crash_vehicle | Object crash without prior vehicle action scenario: The scenario realizes the user controlled ego vehicle moving along the road and encountering a construction setup. |
| controll_loss | Control Loss Vehicle scenario: The scenario realizes that the vehicle looses control due to bad road conditions, etc. and checks to see if the vehicle regains control and corrects it's course. |
| cut_in | Cut in scenario: The scenario realizes a driving behavior on the highway. The user-controlled ego vehicle is driving traight and keeping its velocity at a constant level. Another car is cutting just in front, coming from left or right lane. The ego vehicle may need to brake to avoid a collision. |
| cut_in_with_static_vehicle |Cut in(with static vehicle) scenario synchronizes a vehicle that is parked at a side lane to cut in in front of the ego vehicle, forcing it to break. |
| follow_leading_vehicle | Follow leading vehicle scenario: The scenario realizes a common driving behavior, in which the user-controlled ego vehicle follows a leading car driving down a given road. At some point the leading car has to slow down and finally stop. The ego vehicle has to react accordingly to avoid a collision. The scenario ends either via a timeout, or if the ego vehicle stopped close enough to the leading vehicle. |
| hard_break | Hard break scenario: The scenario spawn a vehicle in front of the ego that drives for a while before suddenly hard breaking, forcing the ego to avoid the collision. |
| invading_turn | Scenario in which the ego is about to turn right when a vehicle coming from the opposite lane invades the ego's lane, forcing the ego to move right to avoid a possible collision. |
| maneuver_opposite_direction | Vehicle Maneuvering In Opposite Direction: Vehicle is passing another vehicle in a rural area, in daylight, under clear weather conditions, at a non-junction and encroaches into another vehicle traveling in the opposite direction. |
| no_signal_junction_crossing | Non-signalized junctions: crossing negotiation: The hero vehicle is passing through a junction without traffic lights And encounters another vehicle passing across the junction. |
| object_crash_intersection | Object crash with prior vehicle action scenario: The scenario realizes the user controlled ego vehicle moving along the road and encounters a cyclist ahead after taking a right or left turn. |
| opposite_vehicle_taking_priority | Scenarios in which another (opposite) vehicle 'illegally' takes priority, e.g. by running a red traffic light. |
| parking_cut_in | Parking cut in scenario synchronizes a vehicle that is parked at a side lane to cut in in front of the ego vehicle, forcing it to break. |
| parking_exit | Scenario in which the ego is parked between two vehicles and has to maneuver to start the route. |
| pedestrian_crossing | Pedestrians crossing through the middle of the lane. |
|route_obstacles | This class holds everything required for a scenario in which there is an accident in front of the ego, forcing it to lane change. A police vehicle is located before two other cars that have been in an accident. |
|vehicle_opens_door | This class holds everything required for a scenario in which another vehicle parked at the side lane opens the door, forcing the ego to lane change, invading the opposite lane. |

## Create new route

For creating a new route take this blueprint of a route:

```xml
<routes>
    <route id="0" town="Town12">
        <weathers>
            <weather route_percentage="0"
                cloudiness="5.0" precipitation="0.0" precipitation_deposits="0.0" wetness="0.0"
                wind_intensity="10.0" sun_azimuth_angle="-1.0" sun_altitude_angle="90.0" fog_density="2.0"/>
            <weather route_percentage="100"
                cloudiness="5.0" precipitation="0.0" precipitation_deposits="0.0" wetness="0.0"
                wind_intensity="10.0" sun_azimuth_angle="-1.0" sun_altitude_angle="15.0" fog_density="2.0"/>
        </weathers>
        <waypoints>
            <position x="780.7" y="5574.7" z="369"/>
            <position x="159.5" y="5573.9" z="369.5"/>
            <position x="-204.7" y="5286.5" z="370"/>
        </waypoints>
        <scenarios>
            <scenario name="HazardAtSideLane_2" type="HazardAtSideLane">
                <trigger_point x="778.7" y="5574.7" z="369" yaw="179.9"/>
                <distance value="50"/>
                <bicycle_drive_distance value="80"/>
                <bicycle_speed value="8"/>
            </scenario>
            </scenarios>
    </route>
</routes>
```

This example is from [routes_bicycle.xml](/code/routes/routes_bicycle.xml)

To change it for your purpose you have to change the following:

1. The id number at line 2 (must be unique). For easy handling increment the id number if you have multiple routes.
2. The waypoints (They discribe where the agent has to go. The first one is the spawnpoint of the agent) Tip: Place always only one waypoint at one road and place it on the right side of the road. To find the coordinates use this interactive [map](https://carla.readthedocs.io/en/latest/map_town12/).
3. The scenario (You can change the scenario with one mentioned in the list) If the route is long enough you can add more than one scenario to a route.
