# Intersection Behavior

**Summary:** This file explains the Intersection behavior.

- [General](#general)
- [Intersection ahead](#intersection-ahead)
- [Approach](#approach)
- [Wait](#wait)
- [Enter](#enter)
- [Leave](#leave)
- [Intersection behavior speed tuning results](#intersection-behavior-speed-tuning-results)

## General

The Intersection behaviour is used to control the vehicle when it encounters a intersection. It handles stop signs as well as traffic lights.
The Intersection sequence consists of the sub-behaviours "Approach", "Wait", "Enter" and "Leave".

To enter the Intersection sequence "Intersection ahead" must firstly be successful.

## Intersection ahead

Successful when there is a stop line within a set distance.

## Approach

Handles approaching the intersection by slowing down the vehicle. Returns RUNNING while still far from the intersection, SUCCESS when the vehicle has stopped or has already entered the intersection and FAILURE when the path is faulty.

Calculates a virtual stopline based on whether a stopline or only a stop sign has been detected and publishes a distance to it. While the vehicle is not stopped at the virtual stopline nor has entered the intersection, int_app_to_stop is published as the current behaviour.
This is used inside motion_planning to calculate a stopping velocity.

A green light is approached with 30 km/h.

## Wait

Handles wating at the stop line until the vehicle is allowed to drive.

If the light is green or when there isn't a traffic light returns SUCCESS otherwise RUNNING.

## Enter

Handles driving through the intersection. Uses 50 km/h as target speed.

Returns SUCCESS once the next waypoint is not this intersection anymore.

## Leave

Signifies that the vehicle has left the intersection and simply returns FAILURE to leave the Intersection sequence.

## Intersection behavior speed tuning results

Left turn with pedestrian:
Faster (50km/h) results in crash, Slower(15,10) results in crash aswell
Right turn:
Seems to be working fine with 50 km/h, is likely controlled by ACC and not behavior though
Firetruck:
50 km/h seems to be fast enough 2/2
