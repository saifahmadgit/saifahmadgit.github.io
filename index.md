---
layout: home
title: Portfolio
---

<!-- 1) Quadruped -->
<div class="project-row">
  <div class="project-media">
    <a href="{{ site.baseurl }}/projects/quadruped-locomotion-rl/">
      <img src="{{ '/assets/gifs/sim_to_real.gif' | relative_url }}" alt="Unitree Go2 RL Locomotion" loading="lazy">
    </a>
  </div>
  <div class="project-info">
    <p><a href="{{ site.baseurl }}/projects/quadruped-locomotion-rl/"><strong>Unitree Go2 — PPO Sim-to-Real Locomotion</strong></a></p>
    <p><em>Reinforcement Learning, Genesis Simulation, Sim-to-Real Transfer</em></p>
    <p>Trains a PPO-based locomotion policy in Genesis simulation and deploys it on a Unitree Go2, using curriculum learning and domain randomization to bridge real-world gaps like torque saturation and sensor noise.</p>
    <p><a href="https://github.com/saifahmadgit/quadruped_locomotion_UnitreeGo2_RL" target="_blank" rel="noopener">GitHub ↗</a></p>
  </div>
</div>

<!-- 2) Franka -->
<div class="project-row">
  <div class="project-media">
    <a href="{{ site.baseurl }}/projects/franka/">
      <img src="{{ '/assets/gifs/franka_demo.gif' | relative_url }}" alt="Franka Vision-Guided Pick & Place" loading="lazy">
    </a>
  </div>
  <div class="project-info">
    <p><a href="{{ site.baseurl }}/projects/franka/"><strong>Franka Vision-Guided Pick & Place</strong></a></p>
    <p><em>ROS 2, MoveIt 2, YOLOv8, Intel RealSense</em></p>
    <p>A complete vision-guided pick-and-place system integrating YOLOv8 object detection, 3D pose estimation, and MoveIt 2 trajectory planning to autonomously grasp and place objects on a Franka Panda robot.</p>
    <p><a href="https://github.com/saifahmadgit/franka-vision-guided-manipulation" target="_blank" rel="noopener">GitHub ↗</a></p>
  </div>
</div>

<!-- 3) Grasp -->
<div class="project-row">
  <div class="project-media">
    <a href="{{ site.baseurl }}/projects/grasp-pose-estimation/">
      <img src="{{ '/assets/gifs/grasp_demo.gif' | relative_url }}" alt="Prompt-to-Pose Grasp Estimation" loading="lazy">
    </a>
  </div>
  <div class="project-info">
    <p><a href="{{ site.baseurl }}/projects/grasp-pose-estimation/"><strong>Prompt-to-Pose Grasp Estimation</strong></a></p>
    <p><em>Grounding DINO, SAM 2, Contact-GraspNet, 6-DoF Grasping</em></p>
    <p>A natural language-guided grasp planning pipeline using Grounding DINO for object localization, SAM 2 for segmentation, and Contact-GraspNet to generate 6-DoF grasp poses from text descriptions.</p>
    <p><a href="https://github.com/saifahmadgit/prompt-guided-robotic-grasping" target="_blank" rel="noopener">GitHub ↗</a></p>
  </div>
</div>

<!-- 4) Jack -->
<div class="project-row">
  <div class="project-media">
    <a href="{{ site.baseurl }}/projects/jack-in-the-box/">
      <img src="{{ '/assets/gifs/jack_demo.gif' | relative_url }}" alt="Jack-in-the-Box Dynamics" loading="lazy">
    </a>
  </div>
  <div class="project-info">
    <p><a href="{{ site.baseurl }}/projects/jack-in-the-box/"><strong>Jack-in-the-Box Dynamics</strong></a></p>
    <p><em>Hybrid Dynamical Systems, SymPy, SciPy</em></p>
    <p>Physics-based simulation and analysis of a Jack-in-the-Box mechanism, modeling the hybrid dynamics of spring oscillation combined with discrete impact events.</p>
  </div>
</div>

<!-- 5) Pen -->
<div class="project-row">
  <div class="project-media">
    <a href="{{ site.baseurl }}/projects/pen-catcher/">
      <img src="{{ '/assets/gifs/pen_demo.gif' | relative_url }}" alt="Pen Catcher Robot" loading="lazy">
    </a>
  </div>
  <div class="project-info">
    <p><a href="{{ site.baseurl }}/projects/pen-catcher/"><strong>Pen Catcher Robot</strong></a></p>
    <p><em>ROS 2, OpenCV, Kalman Filter</em></p>
    <p>A real-time vision system that uses OpenCV tracking and Kalman filter prediction to intercept a falling pen with a robotic arm under tight latency constraints.</p>
    <p><a href="https://github.com/saifahmadgit/MS_Robotics_PenChallenge" target="_blank" rel="noopener">GitHub ↗</a></p>
  </div>
</div>
