# CSE5280: Robotic Arms being a PITA when the building's on fire!
## Author: Andrew Bastien

# Overview
Building on the previous assignment, adding a hostile/malicious/troublesome force in the form of a robot arm that defies physics and pokes around inside the building to specifically harass and block evacuees as efficiently as it possibly can. What a jerk.

## Setup
`python -m venv .venv`
Linux: `source .venv/bin/activate` - or Windows: `.\Scripts\activate.bat`
`pip install -r requirements.txt`

## Project Files:
- the `scenarios/` directory contain simulations to run.
- `simulation.py` is the engine.

# Development Process, Challenges, Solutions, Observations, Insights
1. Reorganized and refactored the simulation engine to combine floorplans and scenarios into a single json file under new directory.

2. Decoupled simulation environment from rendering and created a new `Actor` class which replaces and expands on the simple `Agents` of before. Actors project `Bodies` into the environment which other Actors can react to using various rules. Added support for multiple exits and ensured that they all emit attraction fields.

3. Improved the animation smoothness, made it interruptible by SIGINT from ctrl+c or clicking the GUI's X

4. To study the interactivity between `Evacuees` and the physics-defying robot arm that can phase through ceilings specifically to play goalie, I first created a `Harasser` actor with a repulsive field and the goal of chasing the nearest Evacuee. 

This went poorly at first. The Harasser would chase Evacuees until it got itself stuck underneath a ramp trying to get at the Evacuees who were on the topside of the ramp. Fixed that problem by adding a tangential repulsive force to make anything that gets under a ramp to slide off to either side.

Once the Harasser was no longer getting stuck under the stairs like the creepy stalker it is, I altered its targeting to be at the predicted paths of clusters of nearby Evacuees. Increased its repulsive effect so that Evacuees would run away if the Harasser was in their way even if they were close to the goal.

Then adjusted the Harasser so that it would add urgency to the predicted path of clusters. Added an indicator to show what exactly it was targeting at any given moment because it was hard to tell why it wanted to vibrate in place on occasion instead of playing goalie.

Added support for multiple Harassers, each using different strategies. It's proving difficult to study what the impact of the different strategies is, though.

5. Everything went to pot for a while once I adapted in the robot arms code. The rendering scale between the robot arm and my building are entirely off and it took some time to figure out exactly what was going wrong; the robot had fixed dimensions, I rewired it to have a dynamic scaling so it could be resized and down at the 0.01 to 0.02 range is where the robot arm's size began to be reasonable.

Several hours were spent tweaking and troubleshooting to understand the robot behavior and the limits of their reach. I now have a thorough headache, but my goal was to make the "Harassers" from before into RobotArm tips, and this plan has worked. The harassers are now limited by the reach of the robot arms, which is limited by their scale, but they are now succcessfully doing the basic task of being jerks to the evacuees.

# Algorithms:

The robot arm harassers use four different targeting strategies to locate and pursue evacuees:

## Targeting Strategies

### 1. Nearest
The simplest strategy: directly targets the closest active evacuee. Computes Euclidean distance in 2D from the end-effector position to each evacuee's position and selects the minimum.

### 2. Density
Divides the floor into a grid (default cell size 1.5m) and counts evacuees per cell. Selects the center point of the cell with the highest count. Useful for herding evacuees toward exits by occupying high-density areas.

### 3. Flow
Predicts future crowd movement using a weighted centroid approach. Each evacuee is weighted by proximity to the nearest exit (closer evacuees get higher weight). The target is the weighted centroid plus the average velocity vector projected `prediction_time` seconds into the future (default 2s). Effective for intercepting evacuees before they reach safety.

### 4. Intercept (Default)
Combines cluster detection with trajectory prediction. Uses breadth-first search to group evacuees within `CLUSTER_DISTANCE_THRESHOLD` (2.0m) of each other. Each cluster has a minimum size of `CLUSTER_MIN_SIZE` (2 evacuees). For each valid cluster:
- Computes the centroid (mean position)
- Computes the average velocity of cluster members
- Predicts cluster position 2 seconds ahead
- Scores clusters by: `size × urgency` where urgency = `1 / distance_to_nearest_goal`
- Selects the highest-scoring cluster's predicted position

## Inverse Kinematics

The robot arm uses Jacobian-based numerical IK to reach target positions. At each simulation step:
1. Target (x, y, z) is computed based on the selected strategy, where z corresponds to the target floor's surface + 0.5m
2. The IK solver iteratively adjusts joint angles using pseudoinverse Jacobian until:
   - End-effector is within `target_tolerance` (0.05m) of target, OR
   - `max_iterations` (200) reached

The end-effector position becomes the harasser body's position for collision detection with evacuees.

## Social Force Model

Evacuees experience forces from:
- **Attraction**: Toward nearest exit based on navigation field
- **Repulsion**: From other evacuees (social distancing)
- **Harasser Repulsion**: Strong radial repulsion from robot arm end-effectors when within range
- **Ramp Underside**: Tangential force to prevent getting stuck under ramps

# Scenarios:
There's only one for now:
- `python simulation.py --scenario basic`
More may come as I continue to refine this.