---
layout: project
title: Unitree Go2 — Sim-to-Real Reinforcement Learning
order: 1
tags: Locomotion, Reinforcement Learning, Sim-to-Real, Domain Randomization
gif: /assets/gifs/sim_to_real.gif
---


## Overview

This project is my end-to-end attempt at **getting a Unitree Go2 to walk reliably on real hardware using an RL policy trained in simulation**.

I started from a Genesis quadruped locomotion example and quickly hit the classic wall: **a policy that looks great in sim often fails on the real robot** because real-world dynamics are messy—latency, imperfect sensing, motor saturation, contact uncertainty, and model mismatch.

So the project became less about “train PPO to walk” and more about **engineering a training setup that forces realism** while still converging:
- I progressively introduced **domain randomization**, **sensor/action noise**, and **action latency**.
- These realism knobs initially **broke convergence**.
- I stabilized training using a **metric-gated curriculum** (difficulty adapts to agent performance).
- Finally, I improved sim-to-real robustness by letting the policy output **variable stiffness per leg (PLS)**, not just joint targets.

The result is a PPO policy that tracks velocity commands in simulation under heavy perturbations and **transfers more reliably to the real Go2**.

---

## Media (Watch First)

These two clips are the anchor for everything below—simulation behavior and what transfer looks like on hardware.

### Simulation Demo (Genesis)
<iframe class="video"
        src="https://www.youtube.com/embed/Dnq2HbR6G24"
        title="Unitree Go2 — Genesis Simulation Demo"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>

### Sim-to-Real Demo (Unitree Go2)
<iframe class="video"
        src="https://www.youtube.com/embed/-mpwRv5wt9U"
        title="Unitree Go2 — Sim-to-Real Demo"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>

---

## What I Built (in one sentence)

A **PPO locomotion policy** trained in **Genesis** with a **sim-to-real training stack**: torque-safe control, observation consistency under latency, randomized dynamics, pushes/noise, and a **curriculum that ramps difficulty based on measurable performance**—plus **per-leg stiffness control** to improve robustness.

---

## Training Setup (what the policy actually experiences)

### Simulation (Genesis)
- **Time step:** `dt = 0.02s` (50 Hz), **substeps = 2**
- **Episode length:** `20s` (≈1000 steps)
- **Parallelization:** up to **4096 environments**
- **Robot:** Go2 URDF + ground plane; friction and dynamics randomized during training

### PPO (RSL-RL / `rsl-rl-lib==2.2.4`)
- **Actor/Critic:** MLPs (`ELU`) with hidden dims **[512, 256, 128]**
- **Rollout length:** `num_steps_per_env = 24`
- **Optimization:** `5` epochs, `4` minibatches
- **Core PPO params:** `lr=1e-3`, `clip=0.2`, `gamma=0.99`, `lam=0.95`, `desired_kl=0.01`, `entropy_coef=0.003`

---

## Action Space: Joint Targets + Per-Leg Stiffness (PLS)

A big part of this project is that the policy does **more than position tracking**.

Instead of outputting only joint position targets, the policy outputs:

- **12 position actions** → target joint angles for hip/thigh/calf across 4 legs  
- **+4 stiffness actions** → one value per leg controlling **Kp** (stiffness)

So the action vector is:
- **16 actions = 12 (pos) + 4 (stiffness)**

### Why stiffness helps in sim-to-real
Real contact is unpredictable: the same foot strike can behave differently depending on surface friction, compliance, timing, and motor saturation. With PLS, the policy can choose **soft legs when uncertain** and **stiff legs when it needs authority**, which improves robustness under perturbations.

Implementation details (as used in my code):
- Kp per leg is clamped: `Kp ∈ [10, 70]`  
- Default `Kp=40`, action scale `20`
- Damping is derived automatically:
  - `Kd = 0.2 * sqrt(Kp)`

---

## The Sim-to-Real Problem (what broke and how I fixed it)

When I first enabled sim-to-real realism knobs (randomization + noise + delay), training became unstable or collapsed. The core issue was **nonstationarity** and **inconsistent control/observation semantics**.

Below are the “non-negotiables” that made the setup both **realistic** and **trainable**.

---

## Sim-to-Real: Non-Negotiables

### 1) Torque limits + manual PD torque mode (sim2real-critical)

In simulation, a position controller can demand unrealistic torques without consequences. That leads to policies that “cheat” and fail immediately on hardware.

To fix this, I switched to a **manual PD torque controller** and enforced **per-joint torque limits**:

- PD torque:
  - `τ = Kp (q_target - q) - Kd qdot`
- Torque clamp:
  - hip/thigh: **23.7 Nm**
  - calf: **45.0 Nm**
- Final:
  - `τ = clamp(τ, -τ_max, τ_max)`

This one change forces the policy to learn strategies that respect real motor capabilities.

---

### 2) Latency modeling + observation consistency (the subtle killer)

Real robots have delay: sensing → compute → bus → actuator. If the environment applies delayed actions but the observation shows the *non-delayed* action, training becomes inconsistent and the policy learns the wrong timing.

I implemented:
- **Action history buffer** (delay line)
- Per-episode random delay: **0–1 steps** (≈0–20ms at 50 Hz)
- The observation contains the **applied (delayed) action**, not the raw policy output

This makes the MDP consistent and improves transfer because the policy learns to act under realistic timing.

---

### 3) Correct termination → reward → reset ordering

This is an easy bug to miss and it matters a lot once you add resets and curriculum.

I enforced the standard order:
1. check termination  
2. compute reward using the terminal/pre-reset state  
3. reset environments  
4. compute observations for the next step

This eliminated reward leakage and stabilized learning—especially under aggressive perturbations.

---

### 4) Throttling “global” randomization to reduce nonstationarity

Some Genesis parameters behave effectively **global** (e.g., friction set on the ground entity). If those parameters change too often, training becomes a moving target.

So I throttled global DR:
- Global updates happen only every **N resets** (`global_dr_update_interval = 200`)
- When updated, I ensure privileged buffers stay consistent with current physics

This retains realism while avoiding “training on sand”.

---

## Domain Randomization (what I randomize and why)

Once the above stability fixes were in place, I progressively enabled the full DR suite.

At full difficulty, the training environment includes:

### Contact + terrain interaction
- **Friction (global):** `μ ∈ [0.3, 1.25]`

### Actuation variability
- **Kp/Kd per-joint factors (per-env):** `[0.8, 1.2]`
- **Motor strength scaling (per-env):** `[0.9, 1.1]`

### Inertial / morphology shifts
- **Trunk mass shift (global):** `[-1.0, 3.0] kg`
- **CoM shift (global):** `[-0.03, 0.03] m`
- **Per-leg hip link mass shift (global):** `[-0.5, 0.5] kg`

### Sensor + action noise
- **Observation noise components:**
  - `ang_vel`, `gravity`, `dof_pos`, `dof_vel`
- **Action noise on target joints:** `std = 0.1`

### External disturbances
- Random pushes:
  - force `[-150, 150] N`
  - duration `[0.05, 0.2] s`
  - interval around every `5s` at full difficulty

### Initialization randomization
- Base height: `z ∈ [0.38, 0.45]`
- Base roll/pitch: `±5°`

---

## Curriculum Learning (metric-gated, not time-based)

Adding realism made the environment harder—but “hard from day one” caused PPO collapse.

So I implemented a **metric-gated curriculum manager** that adjusts difficulty based on actual competence.

### What it tracks (EMA)
- **timeout rate**
- **fall rate**
- **tracking reward per second** (normalized by episode duration)

### When it increases difficulty
Difficulty ramps up only after consistently meeting thresholds like:
- `timeout_rate ≥ 0.80`
- `tracking ≥ 0.75`
- `fall_rate ≤ 0.15`
- sustained for a streak (not just one lucky run)

### When it decreases difficulty
If the agent starts failing:
- `fall_rate ≥ 0.25` for a short streak → reduce level

### Why this matters
This curriculum directly addresses the real sim-to-real training dilemma:
> realism improves transfer but hurts convergence.

The curriculum lets the policy **earn** realism progressively.

---

## Observations: Actor vs Privileged Critic

### Actor observation (deployable)
The actor gets only signals that exist on the real robot:
- base angular velocity (scaled)
- projected gravity vector
- commanded velocities (x, y, yaw)
- joint position offsets from nominal pose
- joint velocities
- **applied (delayed) action**

### Privileged critic observation (training only)
The critic additionally sees hidden randomization state:
- true base linear velocity
- current friction
- kp/kd factors
- motor strength
- mass/CoM shifts
- per-leg mass shifts
- gravity offset
- push force
- normalized delay

This improves training stability/sample efficiency without leaking unrealistic info into the deployable policy.

---

## Reward Design (high-level)

The reward is shaped to balance **command tracking**, **stability**, and **gait quality** while discouraging jitter and energy-wasting behavior:

- **Tracking:** linear + angular velocity tracking
- **Stability:** orientation penalty, base height penalty, vertical velocity penalty
- **Smoothness:** action-rate penalty, joint accel/vel regularization
- **Gait quality:** feet air time, foot slip, foot clearance, stance stability (especially for near-zero commands)

The reward design is intentionally conservative: it pushes the policy toward behaviors that transfer and avoids fragile, overfit gaits.

---

## Results (what I’m proud of)

- A training setup that stays stable even after adding:
  **randomization + noise + pushes + latency**
- A policy that transfers better because it’s trained with:
  **torque realism**, **timing realism**, and **progressive difficulty**
- PLS stiffness control that improves robustness under contact uncertainty

---

## Links (for quick access)

- **Simulation:** https://www.youtube.com/watch?v=Dnq2HbR6G24  
- **Sim-to-Real:** https://www.youtube.com/watch?v=-mpwRv5wt9U  
