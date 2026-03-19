# Overview
In the previous assignment, you implemented a particle-based crowd simulation using cost minimization, obstacle avoidance, and social force interactions. Particles moved through the environment while avoiding walls, obstacles, and other particles, eventually reaching exits.

In this assignment, we introduce a robotic agent whose goal is to interfere with the evacuation process.

The robot is modeled as a robot arm with a controllable end-effector. The end-effector acts as a dynamic obstacle in the environment. As particles attempt to leave through the exits, the robot observes the crowd and moves its end-effector toward areas where particles are attempting to escape.

To detect these areas, the robot must analyze the particle distribution and identify clusters of particles approaching exits. The robot will then attempt to intercept the cluster by predicting its future position and moving toward that location.

The overall goal of the robot is not necessarily to stop particles completely, but to interfere with the evacuation flow by strategically placing its end-effector near emerging escape streams.

#Simulation Setup
- You will reuse the simulation environment from the previous assignment. Your implementation already includes:

- particle motion based on cost minimization

- social force interactions between particles

- obstacle avoidance for walls and obstacles

- particle movement toward exits

In this assignment, the following new elements are introduced:

1. A robot arm whose end-effector moves in the environment.

2. A dynamic obstacle corresponding to the robot’s end-effector.

3. A method to detect clusters of particles approaching exits.

4. A prediction mechanism to estimate future cluster locations.

# Particle Behavior
Particles behave exactly as in the previous assignment, following your existing motion model. However, the robot end-effector must now be treated as an additional obstacle in the environment.

Particles should therefore respond to the robot in the same way they respond to other obstacles. Your existing obstacle avoidance mechanism should be reused or adapted to include the robot end-effector as a dynamic obstacle.

You are free to choose how the end-effector influences particles. For example, it may generate repulsive forces similar to walls or other agents.

# Clustering of Escaping Particles
The robot must determine where evacuation is occurring.

Instead of analyzing the entire crowd, the robot should focus on particles that are close to exits. These particles represent individuals actively attempting to leave the building.

Using the positions of these particles, the robot should identify clusters that represent emerging streams of evacuation.

You may use any clustering approach you consider appropriate. One common method is k-means clustering, but other methods may also be explored if desired.

Your implementation should determine which cluster represents the largest or most significant evacuation flow.

# Predicting Cluster Motion
Simply reacting to the current cluster location may not be sufficient, since clusters of particles are constantly moving toward exits.

To improve interception, the robot should estimate the future position of the cluster.

You are encouraged to derive a prediction model based on your existing particle dynamics. One possible approach is to estimate cluster velocity and extrapolate its motion forward in time.

Another possible approach is to adapt the anisotropic social force formulation used in your simulation to bias predictions toward the direction of motion.

You are free to design your own prediction strategy.

# Robot Behavior
At each timestep, the robot should perform the following tasks:

1. Observe the particle distribution.

2. Identify particles that are approaching exits.

3. Detect clusters among those particles.

4. Select a cluster to intercept.

5. Predict the cluster’s future location.

6. Move the robot end-effector toward the predicted location.

The robot motion should be generated using the inverse kinematics framework provided in the starter code.

# Interaction Between Robot and Crowd
Once the robot moves its end-effector into the environment, particles should treat it as an obstacle.

This interaction may cause particles to:

- divert their path

- slow down

- form new clusters

- attempt alternate routes

Your simulation should allow these interactions to emerge naturally from the force and cost functions.

# Implementation Requirements
Your implementation should include:

- clustering of particles near exits

- prediction of cluster motion

- robot end-effector motion toward predicted cluster locations

- interaction between the robot and particle dynamics

The robot behavior should operate within the simulation loop, reacting continuously to the evolving crowd.

# Visualization
Your visualization should clearly show:

- particles moving through the environment

- exits

- clusters detected by the robot

- predicted cluster positions

- robot arm motion

- robot end-effector acting as an obstacle

Clear visualization will help illustrate the interaction between the robot and the crowd.

# Experiments and Analysis
You should conduct experiments to evaluate how the robot affects evacuation behavior.

Possible questions to investigate include:

- Does the robot successfully interfere with evacuation flows?

- How sensitive is the behavior to clustering parameters?

- How does prediction horizon affect robot effectiveness?

- What happens when particle density increases?

- Can the robot unintentionally create new clusters?

# Deliverables
Submit the following:

1. Your simulation code.

2. A short video demonstrating the simulation.

3. A brief report describing:

- your clustering approach

- your prediction model

- how the robot interacts with the crowd

- observations from your experiments.