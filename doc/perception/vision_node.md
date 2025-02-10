# Vision Node

**Summary:** The Visison Node provides an adaptive interface that is able to perform object-detection and/or
image-segmentation.
With the information of the object segmentation and the subscribed depth values the vision node calculates which
lidar points are inside the segmented objects. Also with the segmented image parts of the object detection the traffic
light detection is implemented.

- [Model overview](#model-overview)
- [How it works](#how-it-works)
- [1. Object-Detection](#1-object-detection)
- [2. Distance Mapping](#2-distance-mapping)
- [3. Clustering](#3-clustering)
- [4. Collision Check](#4-collision-check)
- [5. Traffic Light Detection](#5-traffic-light-detection)
- [6. Intermediate Layer](#6-intermediate-layer)
- [7. Dynamic Reconfiguration](#7-dynamic-reconfiguration)

## Model overview

The Vision-Node implements an interface for a lot of different models which can be specified in the perception launch
file.

| Model                                 | Type         | Distance Calculation |
|---------------------------------------|--------------|--------|
| yolov8x-seg                           | segmentation | yes    |
| yolo11n-seg                           | segmentation | yes    |
| yolo11s-seg                           | segmentation | yes    |
| yolo11m-seg                           | segmentation | yes    |
| yolo11l-seg                           | segmentation | yes    |

As only with segmentation masks the correct clusters of lidar points can be done, only segmentation models are
implemented.

## How it works

The Vision-Node basically operates in multiple stages.

1. Object-Detection
2. Mapping the detected objects to the lidar points
3. Clustering the LIDAR points to filter out points which are not part of the object (e.g. segmentation mask does not
match 100 percent to the car and a background lidar point was included in the cluster)
4. Publishing the detected points to the distance_output for the collision check
5. Traffic light detection if a traffic light was found in the image
6. Publishing the object clusters to the intermediate layer

## 1. Object-Detection

The objects are detected with an ultralytics YOLO model.
The detection would include the bounding boxes, the segmentation masks, the class confidence and also a tracking id.
The tracking id could be used in the future to track objects in the intermediate layer.

## 2. Distance Mapping

With the segmentation masks and the corresponding lidar data the vision node calculates which lidar points are inside
the segmented objects.

## 3. Clustering

As it is possible that the segmentation mask does not match 100 percent to the object, the vision node clusters the
lidar points to filter out points which are not part of the object.

## 4. Collision Check

The collision check receives all found points of all objects and uses this information for the collision check.

## 5. Traffic Light Detection

The traffic light detection uses the cropped image of the traffic light.

## 6. Intermediate Layer

All detected object clusters are then published to the intermediate layer.

## 7. Dynamic Reconfiguration

It is currently possible to change the following parameters during runtime:

- `min_x`: most left value of traffic light in pixel
- `max_x`: most right value of traffic light in pixel
- `max_y`: highest value of traffic light in pixel
- `min_prob`: minimal confidence needed
