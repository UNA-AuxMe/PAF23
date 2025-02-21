from leaderboard.autoagents.ros1_agent import ROS1Agent
from leaderboard.autoagents.autonomous_agent import Track


def get_entry_point():
    return "PAFAgent"


class PAFAgent(ROS1Agent):

    def setup(self, path_to_conf_file):
        self.track = Track.MAP

    def get_ros_entrypoint(self):
        return {
            "package": "agent",
            "launch_file": "agent.launch",
            "parameters": {
                "role_name": "hero",
            },
        }

    def sensors(self):
        sensors = [
            {
                "type": "sensor.camera.rgb",
                "id": "Center",
                "x": 0.0,
                "y": 0.0,
                "z": 1.70,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": 1280,
                "height": 720,
                "fov": 100,
            },
            {
                "type": "sensor.lidar.ray_cast",
                "id": "LIDAR",
                "x": 0.0,
                "y": 0.0,
                "z": 1.70,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
            {
                "type": "sensor.other.radar",
                "id": "RADAR0",
                "x": 2.0,
                "y": 1.5,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "horizontal_fov": 25,
                "vertical_fov": 0.1,
            },
            {
                "type": "sensor.other.radar",
                "id": "RADAR1",
                "x": 2.0,
                "y": -1.5,
                "z": 0.5,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "horizontal_fov": 25,
                "vertical_fov": 0.1,
            },
            {"type": "sensor.other.gnss", "id": "GPS", "x": 0.0, "y": 0.0, "z": 0.0},
            {
                "type": "sensor.other.imu",
                "id": "IMU",
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "noise_accel_stddev_x": 0.0,
                "noise_accel_stddev_y": 0.0,
                "noise_accel_stddev_z": 0.0,
                "noise_gyro_stddev_x": 0.0,
                "noise_gyro_stddev_y": 0.0,
                "noise_gyro_stddev_z": 0.0,
                "noise_gyro_bias_x": 0.0,
                "noise_gyro_bias_y": 0.0,
                "noise_gyro_bias_z": 0.0,
            },
            {"type": "sensor.opendrive_map", "id": "OpenDRIVE", "reading_frequency": 1},
            {"type": "sensor.speedometer", "id": "Speed"},
        ]
        return sensors

    def destroy(self):
        super(PAFAgent, self).destroy()
