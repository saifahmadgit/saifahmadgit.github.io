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
    <p class="project-tags"><span class="tag">Reinforcement Learning</span><span class="tag">Genesis Simulation</span><span class="tag">Sim-to-Real Transfer</span></p>
    <p>A PPO-based locomotion policy trained in Genesis simulation and deployed on a real Unitree Go2, using curriculum learning and domain randomization to close the sim-to-real gap across torque saturation, sensor noise, and terrain variability.</p>
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
    <p class="project-tags"><span class="tag">ROS 2</span><span class="tag">MoveIt 2</span><span class="tag">YOLOv8</span><span class="tag">Intel RealSense</span></p>
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
    <p class="project-tags"><span class="tag">Grounding DINO</span><span class="tag">SAM 2</span><span class="tag">Contact-GraspNet</span><span class="tag">6-DoF Grasping</span></p>
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
    <p class="project-tags"><span class="tag">Hybrid Dynamical Systems</span><span class="tag">SymPy</span><span class="tag">SciPy</span></p>
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
    <p class="project-tags"><span class="tag">ROS 2</span><span class="tag">OpenCV</span><span class="tag">Kalman Filter</span></p>
    <p>A real-time vision system that uses OpenCV tracking and Kalman filter prediction to intercept a falling pen with a robotic arm under tight latency constraints.</p>
    <p><a href="https://github.com/saifahmadgit/MS_Robotics_PenChallenge" target="_blank" rel="noopener">GitHub ↗</a></p>
  </div>
</div>
