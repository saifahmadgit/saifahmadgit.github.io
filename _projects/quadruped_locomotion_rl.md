---
layout: project
title: Unitree Go2 — PPO Sim-to-Real Locomotion
order: 1
tags: Reinforcement Learning, Genesis Simulation, Sim-to-Real, Domain Randomization, Curriculum Learning, Per-Leg Stiffness
gif: /assets/gifs/sim_to_real.gif
---

## Overview

This project trains **PPO locomotion policies** in the **Genesis** physics simulator and deploys them on a real **Unitree Go2** quadruped. In simulation, four behaviors were developed: **omnidirectional walking**, **stair climbing**, **crouching**, and **jumping**. Of these, walking and stair climbing have been successfully transferred to hardware using only proprioceptive sensing — no camera or LiDAR.

The central challenge is the **sim-to-real gap**: policies trained in simulation fail on hardware because of unmodeled actuator dynamics, sensing delays, contact uncertainty, and terrain variation. The work closes this gap through domain randomization, sensor noise and latency modeling, metric-gated curriculum learning, and per-leg adaptive stiffness. The overall approach is inspired by [Extreme Parkour](https://extreme-parkour.github.io/); the per-leg stiffness formulation follows [arXiv 2502.09436](https://arxiv.org/abs/2502.09436).

## Workflow

<img src="{{ '/assets/images/workflow.png' | relative_url }}" alt="Training and deployment workflow" style="width:65%;display:block;margin:16px auto;border-radius:8px;">

<img src="{{ '/assets/images/Actor_critic.png' | relative_url }}" alt="Actor-Critic observation asymmetry" style="width:100%;border-radius:8px;margin:16px 0;">

The pipeline runs in three stages:

**1. Genesis simulation** — 4096 parallel environments generate rollouts. The PPO **actor** receives only proprioceptive observations available on hardware. The **critic** additionally sees privileged ground-truth quantities (friction, true velocity, mass, push forces, terrain heights) that cannot be measured at deployment. This asymmetric actor–critic design lets the critic guide value estimation during training while keeping the actor deployable with sensors that actually exist on the robot.

**2. Qualitative evaluation in sim** — Before deploying, the policy is stress-tested at increasing difficulty: friction sweeps, observation and action noise, control latency, external push forces, dynamic payload, and stair heights. Convergence is monitored through TensorBoard — reward saturation, adaptive learning-rate decay, and entropy reduction all confirm the policy has converged.

**3. Hardware deployment** — The actor runs at **50 Hz**. Motor commands stream over the Go2 DDS bus at **500 Hz**. Real-robot trials provide qualitative feedback that informs the next training cycle — adjusting DR ranges, reward weights, or curriculum thresholds.

## Part I — Omnidirectional Walking

### Simulation Setup in Genesis

Training runs in **Genesis** with 4096 parallel environments. The PPO implementation is **rsl-rl-lib v2.2.4**.

**Simulation and training parameters**

| Parameter | Value |
|---|---|
| Control frequency | 50 Hz (dt = 0.02 s) |
| Physics substeps | 2 per control step |
| Episode length | 20 s |
| Command resampling | every 5 s |
| Parallel environments | 4,096 |
| Standing environments | 10 % (near-zero commands) |
| Architecture | MLP, ELU, hidden dims [512, 256, 128] |
| Learning rate | 1 × 10⁻³ (adaptive KL schedule) |
| Clip parameter | 0.2 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Desired KL | 0.01 |
| Entropy coefficient | 0.003 |
| Steps per env per update | 24 |
| Mini-batches | 4 |
| Epochs per update | 5 |
| Max iterations | 10,000 |

**Action space**

The policy outputs **16 actions**: 12 joint position targets (hip/thigh/calf × 4 legs) plus 4 per-leg stiffness scalars. Per-leg stiffness is described in the PLS section.

**Simulation videos**

<div style="background:#eaecf4;border-radius:8px;padding:32px;text-align:center;margin:24px 0;color:#666;font-style:italic;">
  [Video placeholder — Genesis simulation: omnidirectional walking]
</div>

### First Sim-to-Real Attempt

The baseline policy — trained with clean observations, zero latency, and fixed dynamics — learns a stable walk in simulation. On hardware it fails within seconds: the robot struggles to balance, overshoots joint targets, and falls.

The failure comes down to the fact that the simulator presents a fundamentally different world from the real robot. There is no sensor noise, no control delay, and dynamics are fixed across every episode. A policy that has only ever seen a clean, perfectly consistent environment cannot cope with the variability it encounters on hardware — noisy IMU readings, encoder quantization, bus latency between command and execution, and surface friction it has never been exposed to.

<div style="background:#eaecf4;border-radius:8px;padding:32px;text-align:center;margin:24px 0;color:#666;font-style:italic;">
  [Video placeholder — first deployment attempt: failure on real Go2]
</div>

### Closing the Gap

Three engineering additions close the gap, introduced sequentially.

#### 1. Domain Randomization (DR)

DR makes the real robot a member of the training distribution. Parameters are randomized from easy settings at curriculum level 0 up to hard ceilings at level 1.0.

| Parameter | Easy (level 0) | Hard (level 1.0) | Scope |
|---|---|---|---|
| Ground friction | [0.6, 0.8] | [0.3, 1.25] | Global |
| Kp factor | [0.95, 1.05] | [0.8, 1.2] | Per-env |
| Kd factor | [0.95, 1.05] | [0.8, 1.2] | Per-env |
| Motor strength | [0.97, 1.03] | [0.9, 1.1] | Per-env |
| Trunk mass shift | [−0.2, +0.5] kg | [−1.0, +3.0] kg | Global |
| CoM shift | ±0.005 m | ±0.03 m | Global |
| Per-leg hip mass | ±0.1 kg | ±0.5 kg | Global |
| Gravity offset | ±0.2 m/s² | ±1.0 m/s² | Per-env |
| Obs noise — ang_vel | — | ±0.2 rad/s | Per-step |
| Obs noise — dof_pos | — | ±0.01 rad | Per-step |
| Obs noise — dof_vel | — | ±1.5 rad/s | Per-step |
| Action noise | — | std = 0.1 | Per-step |
| External push | None | ±150 N, every 5 s, 0.05–0.2 s | External |
| Action delay | 0 steps | 0–1 steps (0–20 ms) | Per-episode |

**Non-stationarity control:** global parameters (friction, mass, CoM) are re-randomized only every 200 resets — not every episode — to prevent the simulator from behaving like a moving target during a PPO rollout.

#### 2. Sensor Noise and Control Latency

Gaussian noise is injected into every observation at each step. Control latency is modeled as a delay buffer of 0–1 steps (matching the Go2 bus delay at 50 Hz). Critically, the **observation fed back to the policy includes the delayed action that was actually applied** — not the most recently computed action. This keeps the MDP internally consistent and prevents the policy from reasoning about actions it has not yet executed.

#### 3. External Push Forces

Random impulses (±150 N, 0.05–0.2 s duration, every 5 s) are applied to the base. This builds disturbance rejection that is essential when the robot encounters unexpected contacts on real terrain.

### Convergence Issues → Metric-Gated Curriculum

Adding DR, noise, latency, and pushes simultaneously causes PPO to diverge: the training environment is too hard from the start, the policy never gains competence, and learning collapses. The fix is a **metric-gated curriculum** that increases difficulty only after demonstrating sustained performance.

Three EMA-tracked metrics gate progression:

| Metric | EMA smoothing | Level up (after 4 checks) | Level down (after 2 checks) |
|---|---|---|---|
| Timeout rate | α = 0.03 | ≥ 0.80 | — |
| Velocity tracking score | α = 0.03 | ≥ 0.75 | — |
| Fall rate | α = 0.03 | ≤ 0.15 | ≥ 0.25 |

Level changes are asymmetric: **+0.01 up, −0.03 down** — the policy retreats from difficulty faster than it advances, preventing catastrophic forgetting. A cooldown of 5 curriculum updates separates consecutive changes.

**Curriculum mixing** keeps training anchored: 80 % of environments sample the current difficulty level; the remaining 20 % sample from a lower band [0.0, 0.5]. This prevents the policy from forgetting easier behaviors as it pushes the frontier upward.

### Per-Leg Stiffness (PLS)

Inspired by [arXiv 2502.09436](https://arxiv.org/abs/2502.09436), the policy outputs a **stiffness scalar per leg** in addition to joint position targets. Damping is derived from stiffness by a fixed formula:

**Kd = 0.2 × √Kp** (Eq. 6 in the paper)

| PLS parameter | Value |
|---|---|
| Stiffness outputs | 4 (one per leg: FR, FL, RR, RL) |
| Kp range (training) | [10, 70] Nm/rad |
| Kp default | 40 Nm/rad |
| Action scale | 20 |
| Kp range (deployment) | [20, 60] Nm/rad |

**Why PLS helps:** during locomotion, the **stance leg** needs high stiffness to support body weight and resist perturbations; the **swing leg** benefits from lower stiffness to absorb contact impulses. Fixed Kp/Kd cannot express this timing-dependent compliance. With PLS the policy learns, emergently, to stiffen a leg during stance and soften it during swing. PLS also reduces brittle manual gain tuning — stiffness becomes a policy output, not a fixed hyperparameter, and it generalizes across surfaces.

### Reward Function (Walking)

| Term | Scale | Role |
|---|---|---|
| tracking_lin_vel | +1.5 | Track commanded linear velocity |
| tracking_ang_vel | +0.8 | Track commanded yaw rate |
| lin_vel_z | −2.0 | Penalize vertical base movement |
| base_height | −0.6 | Maintain nominal height (0.3 m target) |
| orientation_penalty | −5.0 | Penalize roll and pitch |
| ang_vel_xy | −0.05 | Penalize rolling/pitching rate |
| action_rate | −0.01 | Penalize rapid action changes |
| similar_to_default | −0.1 | Regularize toward default joint pose |
| dof_acc | −2.5 × 10⁻⁷ | Penalize joint accelerations |
| dof_vel | −5 × 10⁻⁴ | Penalize joint velocities |
| feet_air_time | +0.2 | Reward appropriate foot lift (0.1 s target) |
| foot_slip | −0.1 | Penalize foot slipping on contact |
| foot_clearance | −0.1 | Penalize insufficient foot height |
| joint_tracking | −0.1 | Penalize joint target error |
| stand_still | −0.5 | Penalize motion when commands ≈ 0 |
| stand_still_vel | −2.0 | Penalize velocity when commands ≈ 0 |
| feet_stance | −0.3 | Encourage proper stance timing |

### Results — Omnidirectional Walking

The policy achieves robust omnidirectional locomotion: forward/backward, lateral stepping, and yaw rotation. Transfer to hardware is successful — the robot walks reliably on indoor floors, transitions between surfaces, and recovers from external pushes.

<img src="{{ '/assets/gifs/sim_And_real_Omni.gif' | relative_url }}" alt="Omnidirectional walking: simulation and real" style="width:100%;border-radius:8px;margin:16px 0;">

<iframe class="video"
        src="https://www.youtube.com/embed/vyA5PE-hZUc"
        title="Unitree Go2 — Sim + Real Omnidirectional Walking"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>

## Part II — Stair Climbing

### Approach

Stair climbing introduces challenges absent on flat ground: the base must pitch forward during ascent, each leg needs higher foot clearance to clear step edges, and the terrain underfoot is discontinuous. The goal was a **blind (proprioceptive-only) policy** — no camera or LiDAR at deployment.

The actor uses the same **49-dimensional observation** as the walking policy: no height information. The **critic**, used only during training, additionally receives a **height scan** of 11 × 7 = 77 sample points in the body frame (±0.5 m forward/backward, ±0.3 m lateral). This gives the critic terrain context to guide value estimation, without making that information a dependency of the deployable actor.

The stair policy is **initialized from the pre-trained walking checkpoint** and fine-tuned on stair terrain at a lower learning rate (3 × 10⁻⁴), appropriate for transfer learning.

### Terrain Curriculum

The stair terrain is a heightfield with **13 difficulty rows**, each row containing 4 repeating up-down stair flights with flat approach and recovery sections. Row index corresponds directly to step height.

| Parameter | Value |
|---|---|
| Difficulty levels | 13 rows |
| Step height range | 2 cm (row 0) → 15 cm (row 12) |
| Step depth (tread) | 39 cm — matched to the real staircase |
| Steps per flight | 6 up + 6 down |
| Flights per row | 4 up-down cycles |
| Heightfield resolution | 5 cm horizontal, 0.5 cm vertical |
| Episode length | 25 s |

The curriculum starts at **level 0.65** (row ~8 of 13) because the walking policy already handles low stairs. Thresholds are relaxed relative to the walking curriculum — stair climbing slows the robot, so tracking scores are naturally lower:

| Metric | Threshold (advance) | Threshold (retreat) |
|---|---|---|
| Timeout rate | ≥ 0.60 | — |
| Tracking score | ≥ 0.45 | — |
| Fall rate | ≤ 0.35 | ≥ 0.40 |
| Streak needed | 5 checks | 2 checks |

**Spawn distribution** uses a 40 % frontier / 30 % near-frontier / 30 % easy split to balance exploration of harder rows with consolidation of learned skills.

### Reward Changes for Stairs

Several reward terms are modified from the walking policy to accommodate stair-specific behavior:

| Term | Walking | Stairs | Reason |
|---|---|---|---|
| forward_progress | — | +0.4 | Direct incentive for +x displacement on stairs |
| orientation | roll + pitch | **roll only** (−5.0) | Pitch is expected during ascent — do not penalize it |
| lin_vel_z | −2.0 | −1.0 + 0.15 m/s deadzone | Allow upward body velocity during step climbing |
| base_height | −0.6 | −0.1 | Robot needs freedom to bob on uneven steps |
| similar_to_default | −0.1 | −0.05 | Stair postures naturally deviate from default stance |
| foot_clearance | −0.1 | **−0.5** (terrain-relative) | Must clear step edges — the key stair reward |
| feet_height_target | 0.075 m | **0.17 m** | Higher foot lift required to clear step edges |
| tracking_lin_vel | +1.5 | +1.5 | Kept, but commands restricted to forward-only [0.3, 0.8] m/s |

Commands during stair training are **forward-only** (no lateral, no yaw) — the terrain is a corridor layout and the task is purely ascending/descending.

**Two-phase DR schedule:** stair learning and DR are decoupled. During Phase 1 (terrain curriculum level < 0.5), DR is capped at level 0.15 so the robot can focus on the novel task without simultaneous dynamics disturbance. Once the policy handles mid-difficulty stairs (Phase 2), DR ramps up to the full ceiling for robustness.

### Results — Stair Climbing

<img src="{{ '/assets/gifs/Stairs.gif' | relative_url }}" alt="Stair climbing in Genesis simulation" style="width:100%;border-radius:8px;margin:16px 0;">

<div style="background:#eaecf4;border-radius:8px;padding:32px;text-align:center;margin:24px 0;color:#666;font-style:italic;">
  [Video placeholder — stair climbing in Genesis simulation]
</div>

The blind policy transfers to the real staircase (39 cm tread, ~15 cm riser) using only proprioception.

<img src="{{ '/assets/gifs/stairs_final.gif' | relative_url }}" alt="Stair climbing on real Unitree Go2" style="width:100%;border-radius:8px;margin:16px 0;">

<div style="background:#eaecf4;border-radius:8px;padding:32px;text-align:center;margin:24px 0;color:#666;font-style:italic;">
  [Video placeholder — stair climbing on real Go2]
</div>

## Deployment Details

### Dual-Frequency Control Loop

Two threads run simultaneously during deployment:

| Thread | Frequency | Role |
|---|---|---|
| Policy thread | 50 Hz | Read IMU + joints, build obs, run actor, compute targets |
| LowCmd writer | 500 Hz | Stream joint position + Kp/Kd commands over DDS |

The policy runs at 50 Hz (matching the simulation control frequency); the 500 Hz writer saturates the Go2's native LowCmd protocol for smooth motor control.

### Safety and Smoothing

Before activating the policy:
- Sport mode is released via `MotionSwitcherClient`
- The robot ramps to a default stand pose over **4 seconds** (interpolated from current joint positions)
- A **slew limiter** caps per-step joint target changes at **0.1 rad** to prevent jerk

At shutdown, 200 stop packets are flushed to ensure a safe zero-torque state.

### PLS Deployment Knobs

The network-computed stiffness output can be scaled at runtime without retraining:

```
final_Kp = network_Kp × KP_FACTOR
final_Kd = network_Kd × KD_FACTOR
```

KP_FACTOR = 1.0 by default. Increasing it stiffens the robot; decreasing it softens it. This is particularly useful because the same Kp/Kd values produce different joint behavior in sim and on hardware — the runtime factor provides a one-knob adjustment without retraining.

## Conclusions and Future Work

The sim-to-real gap is inherent: no simulator fully captures actuator compliance, contact mechanics, or real-world sensor characteristics. Domain randomization addresses this by making the real robot a member of the training distribution — rather than matching sim to reality exactly, it makes reality a subset of the simulated variations.

In practice the gap persists in subtle ways. Successful policies achieve the goal (walking, climbing stairs), but the motion strategies differ between simulation and hardware. The same Kp/Kd values produce noticeably different joint behavior in the two settings — pointing to unmodeled actuator compliance and transmission elasticity.

**Future directions:**

- **System identification for actuators** — fitting simulator actuator parameters to real motor response (frequency response, current-to-torque mapping) would substantially reduce the PD gain mismatch that is currently compensated by the KP_FACTOR deployment knob
- **Exteroceptive sensing** — adding a depth camera or LiDAR to the actor observation would let the robot perceive terrain ahead of time, likely closing the remaining reliability gap in stair climbing
- **Online adaptation** — an RMA-style adaptation module could estimate real-world parameter shifts (payload, surface friction) at deployment and feed a compact latent into the actor, bridging the remaining gap without retraining

## References

- [Extreme Parkour with Legged Robots](https://extreme-parkour.github.io/) — inspiration for the overall sim-to-real training framework
- [Variable Stiffness for Robust Locomotion through RL (arXiv 2502.09436)](https://arxiv.org/abs/2502.09436) — per-leg stiffness formulation and Kd = 0.2 × √Kp formula
