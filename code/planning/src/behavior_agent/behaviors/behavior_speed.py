from collections import namedtuple

import rospy


def convert_to_ms(speed):
    return speed / 3.6


# Speeds as Parameters:
intersection_enter_speed = rospy.get_param("/intersection_enter_speed", 50)
intersection_approach_speed = rospy.get_param("/intersection_approach_speed", 50)

# End Parameters

Behavior = namedtuple("Behavior", ("name", "speed"))

# Leave Parking space

parking = Behavior("parking", convert_to_ms(30.0))

# INTERSECTION BEHAVIORS

# Approach
int_app_init = Behavior("int_app_init", convert_to_ms(intersection_approach_speed))

# No Traffic Light or Sign -> stop dynamically at Stopline
int_app_to_stop = Behavior("int_app_to_stop", -2)

int_app_green = Behavior("int_app_green", convert_to_ms(intersection_enter_speed))

# Wait
int_wait = Behavior("int_wait", 0)

# Enter
int_enter = Behavior("int_enter", convert_to_ms(intersection_enter_speed))

# Exit
int_exit = Behavior("int_exit", -1)  # Get SpeedLimit dynamically


# LANE CHANGE

# Approach
lc_app_init = Behavior("lc_app_init", convert_to_ms(30.0))

# Stop dynamically at lane change point
lc_app_blocked = Behavior("lc_app_blocked", -2)

lc_app_free = Behavior("lc_app_free", convert_to_ms(30.0))

# Wait
lc_wait = Behavior("lc_wait", 0)

# Enter
lc_enter_init = Behavior("lc_enter_init", convert_to_ms(20.0))

# Exit
lc_exit = Behavior("lc_exit", -1)  # Get SpeedLimit dynamically


# Overtake: Speed is most of the time dynamic

# Approach

ot_app_blocked = Behavior("ot_app_blocked", -2)

ot_app_free = Behavior("ot_app_free", -1)

# Wait

ot_wait_stopped = Behavior("ot_wait_stopped", convert_to_ms(0.0))

ot_wait_free = Behavior("ot_wait_free", -1)

# Enter

ot_enter_init = Behavior("ot_enter_init", -1)

ot_enter_slow = Behavior("ot_enter_slow", -2)

# Exit

ot_leave = Behavior("ot_leave", -1)

# Cruise

cruise = Behavior("Cruise", -1)

# Unstuck Routine
us_unstuck = Behavior("us_unstuck", -3)
us_stop = Behavior("us_stop", 0)
us_overtake = Behavior("us_overtake", 0)
