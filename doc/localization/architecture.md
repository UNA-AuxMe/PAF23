# Localization: Overview and architecture

**Summary:**
The localization package fuses IMU, GPS as well as Odometry Data into an accurate position estimation.

- [robot\_localization](#robot_localization)
- [Architecture](#architecture)
- [Summary of components](#summary-of-components)
  - [localization.launch](#localizationlaunch)
  - [sensor\_covariance\_fusion.py](#sensor_covariance_fusionpy)
  - [odometry\_fusion.py](#odometry_fusionpy)
  - [gps\_transform.py](#gps_transformpy)
  - [current\_state\_publisher.py](#current_state_publisherpy)
  - [Configuration files](#configuration-files)
    - [ekf\_config.yaml](#ekf_configyaml)
    - [odometry\_covariances.yaml](#odometry_covariancesyaml)
    - [sensor\_covariances.yaml](#sensor_covariancesyaml)

## [robot_localization](http://docs.ros.org/en/melodic/api/robot_localization/html/index.html)

The robot_localization package is a ROS-package providing state_estimation_nodes such as EKF and UKF.
Because inputs can be defined and added easily we decided on using this as a framework of our localization algorithm.

## Architecture

There are two EKF Nodes running.
The first (local_ekf) fuses Odometry data with IMU data in order to track the cars position relative to the odom frame. The odom frame itself is irrelevant for us, but this follows not only the suggestion provided in the robot localization package but also with [REP 105](https://www.ros.org/reps/rep-0105.html).
The second (global_ekf) fuses the gps data with some IMU data in order to calculate the global position.
This node publishes the global to odom frame.

## Summary of components

There are several files in the localization package. The following are a list of the most relevant.

### [localization.launch](../../code/localization/launch/localization.launch)

This launch file launches all appropriate nodes.
Apart from our custom nodes (see below) this also launches the two ekf nodes.

### [sensor_covariance_fusion.py](../../code/localization/src/sensor_covariance_fusion.py)

As the IMU and GPS data is provided by carla without covariance but the ekf nodes rely on the covariances beeing part of the message, this node catches the IMU and GPS messages, adds the appropriate covariances and republishes them.

### [odometry_fusion.py](../../code/localization/src/odometry_fusion.py)

This node calculates the odometry message from the current speed and the current steering angle of the vehicle.
It then publishes this together with appropriate covariances.

### [gps_transform.py](../../code/localization/src/gps_transform.py)

The gps_transform node is a substitude for the [navsat_transform_node](http://docs.ros.org/en/melodic/api/robot_localization/html/integrating_gps.html#using-navsat-transform-node) provided in the robot_localization package.
This needed to be custom made because the navsat_transform_node converts gps data into UTM-coordinates.
Carla assumes however wgs coordinates.

### [current_state_publisher.py](../../code/localization/src/current_state_publisher.py)

As most of the current PAF implementation relies not on the tf-data but on custom messages (namely current_position, current_heading) this node listens to the tf-graph and converts it to this data.

### Configuration files

#### [ekf_config.yaml](../../code/localization/config/ekf_config.yaml)

With this yaml the ekf nodes can be configured.
The first section configures the local ekf. The second one the global ekf.

#### [odometry_covariances.yaml](../../code/localization/config/odometry_covariances.yaml)

This file has the covariance matrices for the odometry data.

#### [sensor_covariances.yaml](../../code/localization/config/sensor_covariances.yaml)

This file contains the covariance matrices for GPS and IMU data.
