#!/usr/bin/env python3
import os
from time import sleep
import ros_compatibility as roscomp
from ros_compatibility.node import CompatibleNode
import carla

# from carla import command
import rospy
import random


class TestRoute(CompatibleNode):

    def __init__(self):
        super(TestRoute, self).__init__("testRoute")

        self.control_loop_rate = self.get_param("control_loop_rate", 0.025)
        self.role_name = self.get_param("role_name", "ego_vehicle")
        self.follow_hero = self.get_param("follow_hero", True)
        self.vehicle_number = self.get_param("vehicle_number", 50)
        self.only_cars = self.get_param("only_cars", False)
        self.disable_vehicle_lane_change = self.get_param(
            "disable_vehicle_lane_change", False
        )

        host = os.environ.get("CARLA_SIM_HOST", "localhost")

        self.client = carla.Client(host, 2000)
        self.client.set_timeout(60.0)

        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_hybrid_physics_mode(True)
        self.traffic_manager.set_hybrid_physics_radius(70.0)
        self.traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        self.traffic_manager.set_random_device_seed(0)
        self.traffic_manager.set_respawn_dormant_vehicles(True)
        self.traffic_manager.set_boundaries_respawn_dormant_vehicles(25, 700)
        self.traffic_manager.global_percentage_speed_difference(10.0)

        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.tile_stream_distance = 650
        settings.actor_active_distance = 650
        settings.spectator_as_ego = False
        self.world.apply_settings(settings)

        self.wait_for_hero()

        self.spectator = self.world.get_spectator()

    def spawn_traffic(self):
        self.loginfo("Spawning traffic")

        spawn_points = self.world.get_map().get_spawn_points()
        hero_location = self.hero.get_location()
        spawn_points.sort(key=lambda x: x.location.distance(hero_location))

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        if self.only_cars:
            blueprints = [
                b for b in blueprints if int(b.get_attribute("number_of_wheels")) == 4
            ]
        vehicles = []
        max_vehicles = min([self.vehicle_number, len(spawn_points)])

        # SpawnActor = command.SpawnActor
        # SetAutopilot = command.SetAutopilot
        # FutureActor = command.FutureActor

        # batch = []
        for _, transform in enumerate(spawn_points[:max_vehicles]):
            blueprint = random.choice(blueprints)

            if blueprint.has_attribute("driver_id"):
                driver_id = random.choice(
                    blueprint.get_attribute("driver_id").recommended_values
                )
                blueprint.set_attribute("driver_id", driver_id)

            if blueprint.has_attribute("color"):
                color = random.choice(
                    blueprint.get_attribute("color").recommended_values
                )
                blueprint.set_attribute("color", color)

            blueprint.set_attribute("role_name", "autopilot")

            vehicle = self.world.try_spawn_actor(blueprint, transform)

            # batch.append(SpawnActor(blueprint, transform)
            #              .then(SetAutopilot(FutureActor, True)))

            if vehicle is not None:
                vehicles.append(vehicle)
                vehicle.set_autopilot(True)

                if self.disable_vehicle_lane_change:
                    self.traffic_manager.auto_lange_change(vehicle, False)

        # for response in self.client.apply_batch_sync(batch, False):
        #     if response.error:
        #         print(response.error)
        #     else:
        #         vehicles.append(response.actor_id)

        self.loginfo("Spawned {} vehicles".format(len(vehicles)))

    def wait_for_hero(self):
        while not rospy.is_shutdown():
            actors = self.world.get_actors()
            if not actors:
                continue

            self.hero = [
                a
                for a in actors
                if "role_name" in a.attributes
                and a.attributes.get("role_name") == self.role_name
            ]
            if self.hero:
                self.hero = self.hero[0]
                break

    def set_spectator(self, set_rotation=False):
        transform = self.hero.get_transform()
        location = carla.Location(
            x=transform.location.x, y=transform.location.y, z=transform.location.z + 2
        )

        if set_rotation:
            self.spectator.set_transform(
                carla.Transform(
                    location,
                    carla.Rotation(
                        pitch=transform.rotation.pitch - 15,
                        yaw=transform.rotation.yaw,
                        roll=transform.rotation.roll,
                    ),
                )
            )
        else:
            self.spectator.set_location(location)

    def run(self):
        self.loginfo("Test-Route node running")
        self.set_spectator(set_rotation=True)

        def loop(timer_event=None):
            self.set_spectator()

        if self.follow_hero:
            self.loginfo("Following hero")
            self.new_timer(self.control_loop_rate, loop)
        else:
            self.loginfo("Not following hero, setting spectator only once")
            self.set_spectator()

        sleep(5)
        self.spawn_traffic()

        self.spin()


def main(args=None):
    roscomp.init("testRoute", args=args)

    try:
        node = TestRoute()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()
