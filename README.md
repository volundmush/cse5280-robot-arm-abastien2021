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