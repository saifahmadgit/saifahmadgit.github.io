---
layout: project
title: Sim-to-Real Reinforcement Learning Walking Including Stairs
order: 1
tech_tags: "Reinforcement Learning (PPO), Sim-to-Real, Genesis"
gif: /assets/gifs/sim_to_real.gif
---

<iframe class="video"
        style="aspect-ratio: 16 / 9.95;"
        src="https://www.youtube.com/embed/nrwN8KrsD2c"
        title="Sim-to-Real Reinforcement Learning Walking Including Stairs"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
        allowfullscreen></iframe>

## Overview

**PPO locomotion policies** trained in the **Genesis** physics simulator deploy on a real **Unitree Go2** — no camera, no LiDAR, proprioceptive sensing only. Four behaviors were developed in simulation: **omnidirectional walking**, **stair climbing**, **crouching**, and **jumping**; walking and stair climbing transfer successfully to hardware.

The core problem is the **sim-to-real gap**: unmodeled actuator dynamics, sensing delays, contact uncertainty, and terrain variation cause simulation-trained policies to fail on hardware. This work closes the gap through domain randomization, sensor noise and latency injection, metric-gated curriculum learning, and per-leg adaptive stiffness.

## Workflow

<img src="{{ '/assets/images/BigPicture_RL_training.png' | relative_url }}" alt="Training and deployment workflow" style="width:100%;height:auto;display:block;margin:16px 0;border-radius:8px;">

The pipeline runs in four stages with explicit feedback loops (see diagram):

**1. Genesis simulation** — 4096 parallel environments generate rollouts using an **asymmetric actor-critic**: the actor sees only the 49 proprioceptive signals available on hardware; the critic additionally receives privileged ground-truth quantities (friction, velocity, mass, push forces, terrain heights). This lets the critic guide training without making the deployable actor dependent on information unavailable at runtime.

**2. Convergence check via TensorBoard** — reward curves and entropy are inspected after training. Convergence is not clean; as DR difficulty ramps up with noise, latency, and push forces added incrementally, reward oscillates rather than settling. Partially converged checkpoints carry forward if they show promising behavior.

<div style="display:flex;gap:32px;flex-wrap:wrap;margin:16px 0;justify-content:center;">
  <figure style="width:33%;margin:0;text-align:center;">
    <img src="{{ '/assets/images/walk_train.png' | relative_url }}" alt="Training reward curve" style="width:100%;border-radius:6px;">
    <figcaption style="font-size:0.85rem;color:#555;margin-top:6px;">Training reward — oscillations visible as DR difficulty increases</figcaption>
  </figure>
  <figure style="width:33%;margin:0;text-align:center;">
    <img src="{{ '/assets/images/walk_entropy.png' | relative_url }}" alt="Policy entropy curve" style="width:100%;border-radius:6px;">
    <figcaption style="font-size:0.85rem;color:#555;margin-top:6px;">Policy entropy — never settles to zero, reflecting the shifting training distribution</figcaption>
  </figure>
</div>

**3. Qualitative stress testing in sim** — gait naturalness, push recovery, and friction/payload extremes are visually inspected. Policies that don't pass don't move to hardware.

**4. Hardware deployment** — the actor runs at **50 Hz**; motor commands stream over DDS at **500 Hz**. Real-robot trials feed back into the next training cycle, adjusting DR ranges, reward weights, or curriculum thresholds.

## Key Technical Contributions

**Metric-gated curriculum** — adding DR, noise, latency, and pushes simultaneously causes PPO to diverge. A metric-gated curriculum increases difficulty only after the policy demonstrates sustained performance across timeout rate, velocity tracking, and fall rate. Difficulty retreats three times faster than it advances, preventing catastrophic forgetting.

**Per-leg adaptive stiffness** — beyond 12 joint position targets, the policy outputs a stiffness scalar per leg. The stance leg stiffens to support body weight; the swing leg softens to absorb contact impulses. Fixed Kp/Kd cannot express this timing-dependent compliance — with learned stiffness the policy adapts emergently, and the behavior generalizes across surfaces without manual gain tuning.

**Stair climbing (blind)** — the stair policy fine-tunes from the walking checkpoint on a 13-level heightfield curriculum (step heights 2–15 cm). The deployable actor uses the same 49 proprioceptive signals; only the training critic receives a height scan. Key reward modifications: pitch is not penalized (expected during ascent), foot clearance is computed terrain-relative, and target foot height increases from 7.5 cm to 17 cm to clear step edges.

## Conclusions and Future Work

The sim-to-real gap is inherent: no simulator fully captures actuator compliance, contact mechanics, or real-world sensor characteristics. Domain randomization addresses this by making the real robot a member of the training distribution; rather than matching sim to reality exactly, it makes reality a subset of the simulated variations.

In practice the gap persists in subtle ways. Successful policies achieve the goal (walking, climbing stairs), but the motion strategies differ between simulation and hardware. The same Kp/Kd values produce noticeably different joint behavior in the two settings, pointing to unmodeled actuator compliance and transmission elasticity.

**Future directions:**

- **System identification for actuators**: fitting simulator actuator parameters to real motor response (frequency response, current-to-torque mapping) would substantially reduce the PD gain mismatch that is currently compensated by the KP_FACTOR deployment knob
- **Exteroceptive sensing**: adding a depth camera or LiDAR to the actor observation would let the robot perceive terrain ahead of time, likely closing the remaining reliability gap in stair climbing
- **High-level policy**: a hierarchical controller that uses LiDAR or camera perception to select between low-level locomotion policies (e.g. switching from flat-ground walking to stair-climbing mode upon detecting stairs)
- **Sample efficiency**: the current pipeline trains with 4096 parallel environments for up to 10,000 iterations, which works but is not particularly efficient. Improving sample efficiency — through better reward shaping, off-policy methods, or more principled curriculum design — was not a focus of this work and remains an open direction

## Inspirations

- **[Extreme Parkour with Legged Robots](https://extreme-parkour.github.io/)**: overall sim-to-real reinforcement learning framework: asymmetric actor-critic, privileged critic observations, and domain randomization strategy
- **[Variable Stiffness for Robust Locomotion through Reinforcement Learning (arXiv 2502.09436)](https://arxiv.org/abs/2502.09436)**: per-leg adaptive stiffness formulation (Kd = 0.2 × √Kp), and domain randomization parameter ranges for friction, mass, motor gains, and external push forces

## Code

Both repositories are forks; this work builds on top of existing infrastructure with significant modifications for the training pipeline and deployment stack.

<div style="display:flex;gap:20px;flex-wrap:wrap;margin:20px 0;width:100%;">
  <a href="https://github.com/saifahmadgit/go2-sim2real-locomotion-rl" target="_blank" rel="noopener"
     style="flex:1;min-width:280px;display:flex;align-items:flex-start;gap:18px;background:#eaecf4;border-radius:8px;padding:24px 28px;text-decoration:none;color:#111;">
    <svg height="36" width="36" viewBox="0 0 16 16" fill="#111" style="flex-shrink:0;margin-top:3px;"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
    <div>
      <div style="font-weight:700;font-size:1.1rem;margin-bottom:6px;word-break:break-all;">saifahmadgit / go2-sim2real-locomotion-rl</div>
      <div style="font-size:0.95rem;color:#555;">Reinforcement Learning Training — Genesis + PPO</div>
    </div>
  </a>
  <a href="https://github.com/saifahmadgit/go2-sim2real-deploy" target="_blank" rel="noopener"
     style="flex:1;min-width:280px;display:flex;align-items:flex-start;gap:18px;background:#eaecf4;border-radius:8px;padding:24px 28px;text-decoration:none;color:#111;">
    <svg height="36" width="36" viewBox="0 0 16 16" fill="#111" style="flex-shrink:0;margin-top:3px;"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>
    <div>
      <div style="font-weight:700;font-size:1.1rem;margin-bottom:6px;">saifahmadgit / go2-sim2real-deploy</div>
      <div style="font-size:0.95rem;color:#555;">Hardware Deployment — Unitree Python SDK</div>
    </div>
  </a>
</div>

## Slides

<a href="{{ '/assets/quaduped_locomotion_Rl.odp' | relative_url }}" download style="display:inline-flex;align-items:center;gap:10px;background:#eaecf4;border-radius:8px;padding:16px 24px;text-decoration:none;color:#111;font-weight:600;">
  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/></svg>
  Download Slides (.odp)
</a>

