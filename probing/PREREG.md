# Probe-capacity sweep — pre-registered expectations
Written 2026-10-25 ~02:10 (training-box time), BEFORE any full-run result of the sweep was seen.
Only known numbers at writing time: 5-batch smoke metrics and baseline epoch 1-2 losses.

## Setup
- Frozen H-JEPA FourRoom L3 (official ckpt, seed42), official probing config, 60 epochs.
- Only variable: probe hidden_dim in {0 (linear), 128, 512 (official baseline), 2048}; one hidden layer, LN+GELU.
- Code: branch probe-capacity (commit c8a0095), default 512 == original code path.
- Control: true agent xy + closed-door one-hot -> same MLP512 head (runs/xy_only_control).

## Hypotheses (eval NMSE, per level L1/L2/L3)
H1 xy: decodable at all capacities; linear clearly worse than MLPs; 128 ~ 512 ~ 2048 (saturated).
H2 distractor0_xy: stays ~1.0 at every capacity incl. 2048 -> "distractor position absent from latent".
   Falsified if 2048 eval NMSE < ~0.7 at any level.
H3 min_distractor_dist: latent NMSE close to the xy-only control -> its decodability is mostly explained
   by agent position, not by distractor information. Falsified if latent is clearly below control (gap > ~0.1).
H4 Overfitting check: if 2048 improves train but not eval, count it as "not decodable".
