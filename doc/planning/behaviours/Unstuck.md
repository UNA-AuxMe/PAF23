# Unstuck Behavior

**Summary:** This file explains the unstuck behavior used as a fallback to recover from stuck situations.

- [Unstuck Behavior](#unstuck-behavior)
  - [Explanation](#explanation)

## Explanation

The Unstuck Behavior is triggered whenever we find ourselves in a stuck situation for some specified amount of time. This Behavior can overwrite any other behavior, since we can also get stuck waiting inside other behaviors.

The Unstuck Behavior works with two Timers:

1. ```Wait Stuck Timer```
2. ```Stuck Timer```

Whenever we are driving at some velocity > 0.1 the timers are reset.

Also if we are in a planned waiting behaviors (currently bs.int_wait.name, bs.lc_wait.name, bs.ot_wait_stopped) or don't have set a behavior yet
(means we are in unparking but not yet started driving -> Wait for lane free) the stuck timer gets reset. This will prevent triggering the unstuck
routine when we still wait scheduled. As we still have the wait stuck timer there is no chance to get in a 'hard' stuck here when something went
wrong in the waiting behaviors.

If we get into a situation where the velocity <= 0.1 the timers start running.

We can then either trigger the Wait Stuck Timer condition (currently 60 secs)
after not moving for longer than that duration -> Wait Stuck

OR

we trigger the quicker Stuck Timer condition (currently 20 secs), after not moving for longer than that duration WHILE being told to move -> Stuck.

Once one of the Timers is triggered we do the following:

1. First time getting stuck in the same position:
   1. Start driving backwards without steering for 1.2 secs
   2. Stop
   3. Reset all timers and let other behaviors work again
2. Second time getting stuck in the same position:
   1. Start driving backwards without steering for 1.2 secs
   2. Stop
   3. Add ```Overtake``` trajectory from [motion_planning.py](/code/planning/src/local_planner/motion_planning.py)
   4. Reset all timers and let other behaviors work again
3. Every other time we only do the same as in **1.** to prevent us from overtaking twice or more

To check wether we got stuck at the same position we also save the last position we triggered the Unstuck Routine at.

Furthermore we publish an unstuck_flag so the motion.py knows not to trigger more than one overtake if we stay in the same position.

If True we won't overtake anymore, if False we are able to overtake again.

We also publish an ```unstuck_distance``` so the motion_planning knows where to start the overtake trajectory.

This behavior is ONLY implememted to improve route completion and make debugging easier. It is the last fallback behavior when every other planning component can not get us back on track.

Files influenced by this behavior are:

- Planning:
  - [motion_planning.py](/code/planning/src/local_planner/motion_planning.py), for the target_speed and overtake
  - [behavior_speed.py](/code/planning/src/behavior_agent/behaviours/behavior_speed.py), for the target_speed
- Acting:
  - [vehicle_controller.py](/doc/acting/vehicle_controller.md), because of driving backwards without steering
  - [velocity_controller.py](/doc/acting/velocity_controller.md), because of the sepcial -3 target_speed case
