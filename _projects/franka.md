---
title: Vision-Guided Pick-and-Place with Franka Emika
tags: ROS 2, MoveIt 2, Perception, Manipulation
---

## Overview
This project implements a **vision-guided pick-and-place pipeline** on a **Franka Emika Panda**, enabling autonomous object manipulation in cluttered scenes.

The system integrates **RGB-D perception**, **deep-learning-based object detection**, and **collision-aware motion planning**, allowing the robot to detect objects, estimate their 3D pose, and execute safe pick-and-place motions.

---

## System Pipeline
1. RGB-D perception using **Intel RealSense**
2. Object detection using a YOLO-based model
3. 3D pose estimation from depth data
4. Grasp pose generation in the robot base frame
5. Motion planning and execution using **MoveIt 2**

---

## Simulation
The full pipeline was first validated in a simulation environment to:
- Debug perception-to-planning interfaces
- Validate grasp approach strategies
- Ensure collision safety before hardware execution

### Simulation Demo
<iframe width="100%" height="420"
  src="https://www.youtube.com/embed/SIMULATION_VIDEO_ID"
  frameborder="0"
  allowfullscreen>
</iframe>

---

## Real Robot Execution
The system was deployed on a **Franka Emika Panda** using live RGB-D perception and collision-aware motion planning.

### Real Robot Demo
<iframe width="100%" height="420"
  src="https://www.youtube.com/embed/R
