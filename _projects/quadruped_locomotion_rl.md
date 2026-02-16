---
layout: project
title: Unitree Go2 — PPO Sim-to-Real Locomotion in Genesis
tags: Locomotion, Reinforcement Learning, Sim-to-Real, Domain Randomization
gif: /assets/gifs/sim_to_real.gif
---

## Overview

This project trains a **Unitree Go2 quadruped walking policy** in **Genesis** and transfers it to the real robot using a **PPO** pipeline (RSL-RL / `rsl-rl-lib==2.2.4`). I started from a Genesis walking example and then incrementally built a sim-to-real stack:

- **Domain randomization** (contact + actuation + inertial + sensor)
- **Observation / action noise**
- **Action latency (delay)**
- **Curriculum learning** to recover convergence when the environment became too hard
- **Variable stiffness control (PLS)**: the policy outputs **per-leg stiffness** in addition to joint position targets

The final result is a policy that can track velocity commands robustly under randomized dynamics and delayed actuation, while staying within physically plausible motor limits.

---

## Media (Quick Demos)

Seeing the behavior first makes the rest of the technical details easier to follow:

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

## Training Setup (what the policy actually sees)

### Simulation (Genesis)
- **Time step:** `dt = 0.02s` (50 Hz), with **substeps = 2**
- **Episode length:** `20s` (≈1000 steps)
- **Parallelization:** up to **4096 environments**
- **Robot:** Go2 URDF in Genesis, plane ground entity with randomized friction

### Policy + PPO (RSL-RL)
- **Algorithm:** PPO (adaptive KL scheduling)
- **Actor/Critic:** MLPs with `ELU`, hidden dims **[512, 256, 128]**
- **Rollout:** `num_steps_per_env = 24`
- **Optimization:** `5` epochs, `4` minibatches
- **Key PPO params:** `lr=1e-3`, `clip=0.2`, `gamma=0.99`, `lam=0.95`, `desired_kl=0.01`, `entropy_coef=0.003`

---

## Action Space: Joint Targets + Per-Leg Stiffness (PLS)

A key sim2real improvement was extending the action space from *only* joint position targets to:

- **12 joint position actions** (FR/FL/RR/RL hip/thigh/calf)
- **+4 stiffness actions** (one **Kp per leg**)

So total actions:
- **16 actions = 12 (pos) + 4 (stiffness)**

### Variable stiffness mapping (PLS)
- Policy outputs a stiffness scalar per leg.
- Converted into **per-leg Kp**, clamped:
  - `Kp ∈ [10, 70]`, default `Kp=40`, action scale `20`
- **Kd is derived automatically**:
  - `Kd = 0.2 * sqrt(Kp)` (stable, smooth damping schedule)

This gives the policy an extra degree of freedom to adapt contact dynamics (soft vs stiff legs) under friction / mass / delay shifts — especially valuable during sim-to-real.

---

## Sim-to-Real: The “non-negotiables” I had to fix

Sim-to-real initially failed in predictable ways (over-aggressive torques, timing mismatch, and nonstationary DR). These are the main fixes that made transfer realistic and stable.

### What this looks like in practice

If you want to map the fixes below to behavior, these two clips are the reference points:

- **Simulation:** https://www.youtube.com/watch?v=Dnq2HbR6G24  
- **Sim-to-Real:** https://www.youtube.com/watch?v=-mpwRv5wt9U  

---

### 1) Torque limits + manual PD torque mode (sim2real critical)
A pure position controller can silently demand unrealistically large torques in simulation. I switched to a **manual PD torque** computation and **clamped torques** to motor limits:

- **Torque clamp (per joint):**
  - hip/thigh: **23.7 Nm**
  - calf: **45.0 Nm**
- PD torque:
  - `τ = Kp (q_target - q) - Kd qdot`
- Then:
  - `τ = clamp(τ, -τ_max, τ_max)`

This prevents the policy from learning “cheats” that cannot exist on hardware.

---

### 2) Action latency: delay buffer + correct observation of applied actions
Real robots have delay (sensing, compute, bus, actuator). I modeled this with an **action history buffer** and per-env random delay:

- Delay sampled per episode: **0–1 steps** (≈0–20ms at `dt=0.02`)
- Critically, the observation includes the **applied (delayed) action**, not the instantaneous policy output.

That one detail matters: if the actor observes the wrong action, training becomes inconsistent and sim2real transfer breaks due to a timing mismatch.

---

### 3) Reward/termination/reset ordering (correct RL semantics)
I enforced the standard “legged-gym style” step order:

1. check termination  
2. compute reward on the terminal/pre-reset state  
3. reset environments  
4. compute observations for the next step

This removed subtle reward leakage and stabilized learning under resets and curriculum gating.

---

### 4) Global parameters in Genesis: throttle to reduce nonstationarity
Some physics parameters are effectively **global** in the Genesis scene (e.g., friction, mass shift on a single entity). If those change too frequently, training becomes extremely nonstationary.

So I throttled global DR updates:
- friction/mass-type global updates only every **N resets** (`global_dr_update_interval = 200`)
- when updated, I also update the privileged obs buffers so the critic stays consistent with the current physics.

This was important once DR became aggressive.

---

## Domain Randomization (what I randomize and why)

I gradually ramp the environment from “easy” to “hard” (see curriculum section). At full difficulty, the DR suite includes:

### Contact + terrain interaction
- **Friction (global):** `μ ∈ [0.3, 1.25]`

### Actuation variability
- **Kp/Kd per-joint factors (per-env):** `[0.8, 1.2]`
- **Motor strength scaling (per-env):** `[0.9, 1.1]`

### Inertial and morphology shifts
- **Trunk mass shift (global):** `[-1.0, 3.0] kg`
- **CoM shift (global):** `[-0.03, 0.03] m`
- **Per-leg (hip link) mass shift (global):** `[-0.5, 0.5] kg`

### Sensor + command noise
- Observation noise with structured components:
  - `ang_vel`, `gravity`, `dof_pos`, `dof_vel`
- Noise is scaled by a curriculum-controlled `obs_noise_level`

### Action noise
- Target position noise: `std = 0.1` (applied to joint targets)

### External disturbances
- Random pushes:
  - force range `[-150, 150] N`
  - duration `[0.05, 0.2] s`
  - interval ~ every `5s` at full difficulty

### Initialization randomization
- Base height randomization: `z ∈ [0.38, 0.45]`
- Base roll/pitch randomization: `±5°`

---

## Curriculum Learning (metric-gated, not time-based)

Once I added DR + noise + delay, training **stopped converging**. The fix was a **metric-gated curriculum manager** that adjusts difficulty based on performance, not iteration count.

### What it tracks (EMA)
- **timeout rate**
- **fall rate**
- **tracking performance per second** (from tracking rewards normalized by episode time)

### When it increases difficulty
It steps the curriculum level up when the agent sustains:
- `timeout_rate ≥ 0.80`
- `tracking ≥ 0.75`
- `fall_rate ≤ 0.15`
- over a streak of multiple checks

### When it decreases difficulty
If the environment becomes too hard:
- `fall_rate ≥ 0.25` for a short streak → decrease level

### Why this helped
This curriculum prevented the classic failure mode:
> “I turned on sim2real realism knobs and PPO collapsed.”

Instead, the policy earns its way into harder physics (pushes, noise, delay, broader friction/mass ranges) only when it is ready.

---

## Observations: Actor vs Privileged Critic

### Actor observation (what transfers to the robot)
Concatenation of:
- base angular velocity (scaled)
- projected gravity vector
- commanded velocities (x, y, yaw)
- joint position error from default pose
- joint velocities
- **applied (delayed) action**

### Privileged critic observation (training only)
The critic additionally gets “hidden” DR state:
- true base linear velocity
- current friction
- kp/kd factors
- motor strength
- mass/CoM shifts
- per-leg mass shifts
- gravity offset
- push force
- normalized delay level

This improves sample efficiency and stabilizes PPO while still learning a deployable actor.

---

## Reward Design (high-level)

The reward is a weighted sum of:
- **command tracking:** linear + angular velocity tracking
- **stability penalties:** orientation, vertical velocity, base height
- **smoothness/regularization:** action rate, dof acceleration, dof velocity
- **gait quality:** feet air time, slip, foot clearance, stance quality
- **standing behavior:** special penalties when command ≈ 0 to encourage stable standing without jitter

---

## Notes / Replication

- Training code uses **RSL-RL** runner (`OnPolicyRunner`) with Genesis GPU backend.
- Sim-to-real stability relies on: **torque clamping**, **delay modeling**, **curriculum gating**, and **global DR throttling**.
- Variable stiffness (PLS) helps the policy adapt to contact uncertainty and actuation differences across sim and hardware.

---

## Links

- **Simulation:** https://www.youtube.com/watch?v=Dnq2HbR6G24  
- **Sim-to-Real:** https://www.youtube.com/watch?v=-mpwRv5wt9U  
