# Real-World Sensor Applications in Autonomous Driving

This research explores the types of sensors which are used in real-world autonomous vehicles by leading companies, specifically Waymo, Tesla, and Cruise. Understanding their sensor configurations can offer insights beneficial for developing simulations of autonomous vehicles.

## Waymo

![alt text](../../../assets/TheWaymoDriverSensors.png)

- *"There are 29 cameras on our Jaguar I-PACEs."* [[waymo.com](https://waymo.com/waymo-driver/)]
- *"designed with high dynamic range and thermal stability, to see in both daylight and low-light conditions"* [waymo.com]
- The newest (6th) generation of the waymo driver has *"13 cameras, 4 lidar, 6 radar, and an array of external audio receivers (EARs)"* [[waymo.com](https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/)]
  
## Tesla

- Tesla Cars use neither Lidar nor Radar [[teslawissen](https://teslawissen.ch/tesla-autopilot-unterschiede-der-hardware-generationen/)]
- *"12 sensors and 8 cameras, for a total of 20 sensors"* [[Ohio state University news](https://u.osu.edu/engr2367selfdrivingcars/how-does-a-self-driving-car-work-1-false/)]
- *"Tesla Vision Update: Replacing Ultrasonic Sensors with Tesla Vision"* [[Tesla Website](https://www.tesla.com/support/transitioning-tesla-vision)]
  - it is speculated that they do that, to safe cost and because their self driving philosophy is heavily computer vision and deep learning guided.
- For the newest Hardware of the highest level of autonomy for their car HW4, it was found out that it will have 11 cameras instead of previously 8, some places on the rear bumper, some in the front. [[Tesla magazin](https://teslamag.de/news/neue-autopilot-hardware-tesla-anschluesse-radar-mehr-kameras-56719)]

## Cruise

*"Cruise vehicles are equipped with 40+ sensors, 360° vision, and tested for millions of miles."*[[Cruise](https://www.getcruise.com/)]

## Key Observations

Autonomous vehicle manufacturers generally maintain confidentiality regarding their technological implementations. Most available information comes from third-party analysis and observations.
Key findings include:

1. Vehicles are equipped with a higher number of sensors than initially anticipated
2. Leading manufacturers are trending toward sensor count optimization for cost efficiency while maintaining safety standards [[the verge](https://www.theverge.com/2018/3/28/17172666/uber-self-driving-crash-sensor-lidar-email-ducey)]

Additionally, there is an industry-wide shift toward computer vision technologies, driven by significant technological advancements in the field.

## Future research

It would be interesting to know how these companies connect the information they receive, to create objects to avoid.

## Note

There are restrictions for the Sensors by the Carla leaderboard administration that we have to keep in mind:

*"You can use any of these sensors to configure your sensor stack. However, in order to keep a moderate computational load we have set the following limits to the number of sensors that can be added to an agent:*

    sensor.camera.rgb: 8
    sensor.lidar.ray_cast: 2
    sensor.other.radar: 4
    sensor.other.gnss: 1
    sensor.other.imu: 1
    sensor.opendrive_map: 1
    sensor.speedometer: 1"
