---
layout: project
title: Unitree Go2 — Sim-to-Real Reinforcement Learning (Genesis + PPO)
order: 1
tags: Locomotion, Reinforcement Learning, Sim-to-Real, Domain Randomization, Curriculum Learning, Torque Control, Latency
gif: /assets/gifs/sim_to_real.gif
---

## Media (sim + real in one clip)

<iframe class="video"
        src="https://www.youtube.com/embed/vyA5PE-hZUc"
        title="Unitree Go2 — Sim + Real (Sim-to-Real RL)"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>

---

## Overview

This project trains a **PPO locomotion policy** in **Genesis** and deploys it on a **Unitree Go2**.  
The emphasis is **sim-to-real reliability** under typical gaps: torque saturation, latency, noisy sensing, contact uncertainty, and dynamics mismatch.

The work started from the standard Genesis quadruped locomotion example. The main engineering effort was making the simulator-policy interface behave like hardware, then gradually increasing realism without breaking PPO convergence.

---

## Design evolution (what changed over time, and why)

This section is the “thread” that connects the technical pieces below.

### Step 0 — Starting point: Genesis locomotion baseline
The initial baseline can learn a walk in simulation, but it typically relies on:
- overly clean sensing
- effectively zero latency
- overly “helpful” actuation
- fixed dynamics

This usually looks stable in sim and fails on hardware.

### Step 1 — First improvement: Domain Randomization (DR)
The first attempt to bridge sim-to-real was **domain randomization** (friction, gains, mass/CoM, motor strength, etc.).  
This improved diversity, but on hardware the robot still struggled to balance and would fall — indicating the remaining gap was not just parameters, but **timing + sensing + actuation realism**.

### Step 2 — Add sensor noise + latency (the real gap)
Next, **sensor noise** and **action latency** were introduced to match the real control loop.  
This made the simulated experience closer to hardware, but it triggered a common problem: **PPO convergence became unstable or collapsed** because the environment difficulty jumped too quickly and the training became highly non-stationary.

### Step 3 — Fix convergence: metric-gated curriculum
To keep realism but avoid collapse, training switched to a **curriculum**:
- start with easier settings
- increase DR / noise / pushes / delay only after the agent shows competence
- reduce difficulty if failure spikes

This allowed the policy to “earn” realism instead of being overwhelmed early.

### Step 4 — Improve robustness: per-leg stiffness (PLS) as policy output
Finally, to improve robustness under contact uncertainty, the policy was extended to output **per-leg stiffness** (Kp) in addition to joint targets.

This matches an intuitive control idea: during walking, the **planted leg benefits from higher stiffness** while the swing leg can remain softer. In practice it also reduces the need for manual Kp/Kd tuning, because stiffness becomes part of the learned control strategy.

---

## Table of Contents

- [System overview](#system-overview)
- [Simulation + PPO configuration](#simulation--ppo-configuration)
- [Action space](#action-space-joint-targets--per-leg-stiffness-pls)
- [Control stack: manual PD torque + torque limits](#control-stack-manual-pd-torque--torque-limits)
- [Latency + observation consistency](#latency--observation-consistency)
- [Reset / reward correctness](#reset--reward-correctness)
- [Domain randomization](#domain-randomization-dr)
- [Metric-gated curriculum](#metric-gated-curriculum)
- [Observations: actor vs privileged critic](#observations-actor-vs-privileged-critic)
- [Rewards](#rewards-full-scales)
- [Deployment on Go2](#deployment-on-go2-policy-loop--500hz-lowcmd)
- [Repro](#repro-training--deployment-entrypoints)

---

## System overview

### Training / deployment pipeline

**Training (Genesis, PPO)**  
commands → actor obs → PPO policy → (pos targets + per-leg stiffness) → manual PD torque → torque clamp → sim step  
+ domain randomization + noise + pushes + latency + curriculum + privileged critic

**Deployment (Go2 hardware)**  
Go2 lowstate → build actor obs (same layout as training) → policy inference @ 50 Hz → target joint angles + PLS stiffness  
→ stream LowCmd @ 500 Hz with per-joint Kp/Kd (plus safety ramp + slew limiting)

---

## Simulation + PPO configuration

A note on intent: these numbers aren’t “the magic recipe”. They are the stable baseline used while progressively adding realism and curriculum.

### Simulation (Genesis)
- **dt:** 0.02 s (50 Hz)
- **substeps:** 2
- **episode length:** 20 s
- **command resampling:** every 5 s
- **parallel envs:** typically 4096

### PPO (RSL-RL / rsl-rl-lib==2.2.4)
- **Actor/Critic MLP:** ELU, hidden dims **[512, 256, 128]**
- **rollout:** `num_steps_per_env = 24`
- **opt:** 5 epochs, 4 minibatches
- **core params:**  
  `lr=1e-3, clip=0.2, gamma=0.99, lam=0.95, desired_kl=0.01, entropy_coef=0.003`

### Command distribution (training)
- `lin_vel_x ∈ [-1.0, 1.0]`
- `lin_vel_y ∈ [-0.3, 0.3]`
- `yaw_rate ∈ [-1.0, 1.0]`
- 10% of envs trained explicitly on near-zero commands for stable standing

---

## Action space: joint targets + per-leg stiffness (PLS)

This is the main “control interface” decision. A position-only action space works in sim, but becomes brittle across surfaces and timing mismatches on hardware.

The policy outputs **both**:
- **12 position actions:** target joint angles (hip/thigh/calf × 4 legs)
- **+4 stiffness actions (PLS):** one scalar per leg controlling stiffness (Kp)

So the action vector is:
- **16 actions = 12 (pos) + 4 (stiffness)**

### PLS mapping + stiffness law
- Leg mapping: **FR joints 0–2, FL 3–5, RR 6–8, RL 9–11**
- Training stiffness range: **Kp ∈ [10, 70]**
- Default stiffness: **Kp = 40**
- Action scale: **20**
- Damping derived from stiffness:
  - **Kd = 0.2 · sqrt(Kp)**

### Why PLS helps sim-to-real (intuition + practical effect)
During locomotion, one or more legs spend time in **stance** (planted) while others are in **swing**.
- Stance leg benefits from higher stiffness to support weight and resist slip/perturbations.
- Swing leg benefits from softer behavior to avoid harsh contacts and reduce jitter.

PLS lets the policy express that timing-dependent compliance. In practice, it also reduces brittle manual Kp/Kd tuning because stiffness becomes part of the policy’s output, not a fixed hyperparameter.

---

## Control stack: manual PD torque + torque limits

This section exists because it is one of the most common silent sim-to-real failure modes:
a simulation controller can generate unrealistic torques, and the policy learns to depend on that.

### Manual PD torque
\[
\tau = K_p (q_{target} - q) - K_d \dot{q}
\]

### Torque limits (per joint)
- hip: **23.7 Nm**
- thigh: **23.7 Nm**
- calf: **45.0 Nm**
(applied across 4 legs → 12 joints)

Final torque is clamped:
\[
\tau \leftarrow \mathrm{clip}(\tau, -\tau_{max}, +\tau_{max})
\]

This prevents “torque cheating” in simulation and forces strategies that remain feasible on hardware.

---

## Latency + observation consistency

After DR, one of the remaining gaps was timing. Real robots have sensing → compute → bus → actuator delay.

The subtle part is not just delaying actions, but making sure the observation/action semantics remain consistent:
if the env applies delayed actions but the observation references a different timing, training becomes incoherent.

### What’s implemented
- **action delay line / history buffer**
- random delay per episode: **0–1 steps** (0–20 ms @ 50 Hz)
- observation includes the **applied (delayed) action**, not the raw policy output

This makes the MDP consistent and reduces timing-related transfer failures.

---

## Reset / reward correctness

Once resets become frequent (because pushes/noise/curriculum increase failure early), ordering mistakes can inject reward leakage.

Step ordering is enforced as:
1) termination check  
2) reward computed from terminal / pre-reset state  
3) reset  
4) next observations computed

---

## Domain randomization (DR)

DR is introduced progressively, but the “ceiling” includes:

### DR maxima (hard ceiling)
**Contact**
- friction (global): **μ ∈ [0.3, 1.25]**

**Actuation**
- per-joint Kp factor: **[0.8, 1.2]**
- per-joint Kd factor: **[0.8, 1.2]**
- motor strength: **[0.9, 1.1]**

**Inertial / morphology**
- trunk mass shift (global): **[-1.0, 3.0] kg**
- CoM shift (global): **[-0.03, 0.03] m**
- per-leg hip link mass shift (global): **[-0.5, 0.5] kg**

**Noise**
- observation noise:
  - ang_vel: 0.2
  - gravity: 0.05
  - dof_pos: 0.01
  - dof_vel: 1.5
- action noise (targets): **std = 0.1**

**External disturbances**
- pushes every **5 s**
- force: **[-150, 150] N**
- duration: **[0.05, 0.2] s**

**Initialization**
- base z: **[0.38, 0.45]**
- roll/pitch: **±5°**

### Non-stationarity control: throttling global DR
Some parameters behave like global sim state (ground friction, mass/CoM).  
If they change too frequently, PPO trains on a moving target.

So global DR updates are throttled:
- update interval: **every 200 resets** (not every reset)

---

## Metric-gated curriculum

This is the convergence fix that allowed noise + latency + strong DR to be added without training collapse.

Difficulty is not increased by wall-clock time. It is increased only after competence is measured consistently.

### Metrics tracked (EMA)
- timeout rate
- fall rate
- tracking score (normalized)

### Thresholds
Increase difficulty only after sustained competence:
- **timeout_rate ≥ 0.80**
- **tracking ≥ 0.75**
- **fall_rate ≤ 0.15**
- must hold for **ready_streak = 4**

Decrease difficulty when failures persist:
- **fall_rate ≥ 0.25** for **hard_streak = 2**
- step sizes:
  - step_up = 0.01
  - step_down = 0.03
- cooldown_updates = 5

### Curriculum mixing (stability trick)
To reduce catastrophic forgetting / instability, sampling is mixed:
- **80% current level**
- remainder sampled from a lower-level band (0.0 → 0.5)

This keeps the policy anchored while pushing the ceiling upward.

---

## Observations: actor vs privileged critic

### Actor obs (deployable)
Signals available on hardware:
- ang_vel (3)
- gravity projection (3)
- command (3)
- dof_pos offsets (12)
- dof_vel (12)
- applied action (16 with PLS)

Total: **49 obs** (with PLS)

### Privileged critic obs (training-only)
Additional randomized “hidden state” for learning stability:
- true base linear velocity
- friction
- kp/kd factors
- motor strength
- mass shift + CoM shift
- per-leg mass shift (4)
- gravity offset (3)
- push force (3)
- delay (normalized)

Total: **49 + 55 = 104 obs** (with PLS)

---

## Rewards (full scales)

These are the exact reward scales used (pre-dt scaling).

### Tracking
- tracking_lin_vel: **+1.5**
- tracking_ang_vel: **+0.8**

### Regularization / stability
- lin_vel_z: **-2.0**
- base_height: **-0.6**
- action_rate: **-0.01**
- similar_to_default: **-0.1**
- orientation_penalty: **-5.0**
- dof_acc: **-2.5e-7**
- dof_vel: **-5e-4**
- ang_vel_xy: **-0.05**

### Gait quality
- feet_air_time: **+0.2**
- foot_slip: **-0.1**
- foot_clearance: **-0.1**
- joint_tracking: **-0.1**

### Energy / torque (disabled in this config)
- energy: **0.0**
- torque_load: **0.0**

### Standing (active when commands ≈ 0)
- stand_still: **-0.5**
- stand_still_vel: **-2.0**
- feet_stance: **-0.3**

Reward targets:
- tracking_sigma = 0.25
- base_height_target = 0.3
- feet_height_target = 0.075
- feet_air_time_target = 0.1

---

## Deployment on Go2: policy loop + 500 Hz LowCmd

Deployment supports:
- `dummy`: test inference from a saved lowstate (off-robot)
- `robot_print`: read-only loop printing actions
- `robot_run`: full control

### Timing
- policy inference: **50 Hz**
- LowCmd streaming: **500 Hz** (writer thread)

### Safety / smoothness
- release high-level/sport first
- ramp to stand pose over **4 s**
- slew limit on targets: **MAX_STEP_RAD = 0.1**
- safe stop sequence sends “stop packets” at shutdown

### PLS deployment knobs
Runtime multipliers allow softer/stiffer behavior without retraining:
- KP_FACTOR and KD_FACTOR scale the network stiffness output
- training clamps [10, 70]; deployment clamps [20, 60] before scaling

---

## Repro: training + deployment entrypoints

### Training
- script: `go2_train_test7.py`
- key args:
  - `-e / --exp_name` (default: go2-walking-v6)
  - `-B / --num_envs` (default: 4096)
  - `--max_iterations` (default: 10000)

### Environment
- env: `Go2Env` in `go2_env_test7.py`
- includes curriculum manager, DR, manual PD torque + clamp, delay semantics, pushes

### Deployment
- script: `go2_policy_3.py`
- loads checkpoint (e.g., `model_188000.pt`)
- builds actor obs matching training layout (49-dim with PLS)
- streams LowCmd at 500 Hz

---

## Quick evaluation guide (where the “sim-to-real” engineering lives)

If skimming:
1) manual PD torque + torque clamp  
2) latency semantics (applied delayed action in obs)  
3) metric-gated curriculum (increase realism only after competence)  
4) per-leg stiffness (PLS) to reduce brittle manual gain tuning
