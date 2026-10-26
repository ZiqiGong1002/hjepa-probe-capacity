# Probe Capacity and Selective Abstraction in H-JEPA

[中文版](README.zh-CN.md) · [Original upstream README](UPSTREAM_README.md)

This repository extends the official implementation of [H-JEPA](https://arxiv.org/abs/2610.06805) with a probing study on the FourRoom environment. It (i) reproduces the paper's level-wise probing results and (ii) tests whether the paper's "selective abstraction" conclusion is robust to the capacity of the probe used to measure it.

| | |
|---|---|
| Upstream | [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA), commit `97054ec`, MIT License |
| Checkpoints | Official FourRoom checkpoints (2, 3 and 4 levels) from Hugging Face, SHA-256 verified |
| Modifications | Three commits under `h_jepa/` (`git log 97054ec..HEAD -- h_jepa`), experiment code and results under `probing/` |
| Not included | Datasets and checkpoints; see [`UPSTREAM_README.md`](UPSTREAM_README.md) for download instructions |

## Summary of findings

1. **Reproduction of Fig. 4.** For the 4-level model (3 model seeds), distractor-position NMSE per level is 0.053 / 0.167 / 0.340 / 0.553, consistent with the values reported in the paper (≈0.06 / 0.19 / 0.37 / 0.58).
2. **Level-wise trend across probes.** For the 2-, 3- and 4-level models, distractor-position NMSE increases monotonically with level for every probe tested (linear, and MLP of width 128, 512 and 2048). Beyond width 128, results are largely saturated for the 2- and 3-level models (at most 0.03 change at the top level). For the 4-level model, higher-level NMSE decreases slightly with width (by 0.07 at L3 and 0.06 at L4 from width 128 to 2048), a change comparable to the variation across model seeds; the capacity sweeps use a single model seed.
3. **Sensitivity to probe class.** At every depth, a linear probe yields substantially higher distractor-position NMSE than any MLP probe (e.g. 0.446 / 0.672 / 0.903 for the 3-level model; 0.978 at L4 of the 4-level model). The amount of information that appears to be discarded therefore depends strongly on whether the probe is linear or nonlinear.
4. **Random-encoder baseline.** An untrained encoder of identical architecture yields distractor-position NMSE of 0.77–0.80 and agent-position NMSE of 0.90–0.92 at all levels, confirming that the decodable information is acquired through training.
5. **Unresolved discrepancy.** For the 2-level model (Fig. 12, 3 model seeds), agent-position results agree with the paper, whereas distractor-position NMSE is higher at both levels (L1: 0.288 vs ≈0.01; L2: 0.503 vs ≈0.33). The gap is not explained by probe capacity: with an MLP-2048 probe, level-1 NMSE is still 0.144.

## Background and research question

H-JEPA assesses what each level of its hierarchy encodes by training a probe on frozen latent representations. In the paper, higher levels retain the agent's position while progressively discarding information about distractors, which is interpreted as selective abstraction. All reported probing uses a single probe architecture (a one-hidden-layer MLP with 512 units).

A probe measures what is *decodable* by that probe, not what is *present* in the representation. Information that a weak probe cannot recover may still be encoded, and a sufficiently strong probe may recover information that the downstream model does not use. This study therefore asks:

> Does the conclusion that higher levels discard distractor information hold across probes of different capacity?

## Method

**Models.** Official H-JEPA FourRoom checkpoints with 2, 3 and 4 levels; all encoder parameters are frozen.

**Probes.** The official probing configuration is used throughout (AdamW, learning rate 1e-3, weight decay 1e-4, batch size 128, 60 epochs, probe seed 42). Each probe has one hidden layer with LayerNorm and GELU; only the hidden width is varied:

| Probe | Hidden width | Parameters |
|---|---|---|
| Linear | 0 | 386 |
| MLP | 128 | 25k |
| MLP (official) | 512 | 101k |
| MLP | 2048 | 403k |

**Targets.** Agent position (`xy`), position of a distractor (`distractor0_xy`), and geodesic distance from the agent to the nearest distractor (`min_distractor_dist`).

**Metric.** Normalised mean squared error (NMSE) on the held-out evaluation split. NMSE = 1 corresponds to predicting the target mean; lower is better.

**Controls.**
- *Position-only control:* the official MLP-512 probe is trained on the ground-truth agent position and closed-door index instead of learned features. It measures how much of each target is predictable from the agent's position alone.
- *Random-encoder baseline:* the 3-level architecture is instantiated with the official constructor but left untrained, and probed in the same way.

**Implementation.** The following changes were made to `h_jepa/`:

| Commit | Description |
|---|---|
| Make probe hidden_dim configurable | Adds `probe_hidden_dim` (0 = linear). The default of 512 preserves the original code path. |
| Add cached-embedding probing runner | `main_probing_cached.py` evaluates the frozen encoder once, caches the embeddings, and trains the same probe heads with the same random-number sequence and metrics. |
| Cached probing: optional random_init baseline | `+random_init=true` instantiates the same architecture without loading trained weights. |

The cached runner was validated against the official pipeline: on the 3-level model (seed 42, MLP-512 probe), all 18 reported NMSE values are identical (maximum absolute difference 0; `probing/scripts/compare_metrics.py`).

**Pre-registration.** Hypotheses for the capacity sweep were recorded in [`probing/PREREG.md`](probing/PREREG.md) before any full result was available.

## Results

### Reproduction of Fig. 4 (4-level model, distractor position)

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Paper (read from figure) | ≈0.06 ± 0.02 | ≈0.19 ± 0.05 | ≈0.37 ± 0.08 | ≈0.58 ± 0.07 |
| This work (3 model seeds, mean ± SE) | 0.053 ± 0.020 | 0.167 ± 0.050 | 0.340 ± 0.073 | 0.553 ± 0.072 |

Agent-position NMSE is ≈1e-4 at all levels, as in the paper.

### Reproduction of Fig. 12 (2-level model, official MLP-512 probe)

| | Agent L1 | Agent L2 | Distractor L1 | Distractor L2 |
|---|---|---|---|---|
| Paper (3 model seeds) | ≈0 | ≈0 | ≈0.01 | ≈0.33 ± 0.01 |
| This work (3 model seeds, mean ± SE) | 5e-5 | 8e-5 | **0.288 ± 0.067** | **0.503 ± 0.082** |
| Seed 42 / 43 / 44 | 4e-5 / 5e-5 / 5e-5 | 7e-5 / 7e-5 / 9e-5 | 0.156 / 0.340 / 0.369 | 0.341 / 0.557 / 0.610 |

Agent-position results agree with the paper; distractor-position results are higher at both levels for every model seed. Probe capacity does not account for the difference (see the 2-level capacity sweep below). The caption of Fig. 12 states that probe heads differ across environments, whereas the released code uses a single MLP-512 probe for all environments, so the probe or model configuration used for this figure may differ from the released one. The cause has not been identified.

### Probe capacity (2-level model, seed 42)

| Probe | Distractor position L1 / L2 | Geodesic distance L1 / L2 | Agent position | Train–evaluation gap at L2 |
|---|---|---|---|---|
| Linear | 0.499 / 0.772 | 0.824 / 0.862 | 0.004–0.011 | 0.005 |
| MLP-128 | 0.166 / 0.369 | 0.197 / 0.414 | ≈5e-5 | 0.027 |
| MLP-512 (official) | 0.156 / 0.341 | 0.176 / 0.390 | ≈6e-5 | 0.033 |
| MLP-2048 | 0.144 / 0.312 | 0.173 / 0.370 | ≈8e-5 | 0.031 |
| Paper, Fig. 12 | ≈0.01 / ≈0.33 | — | ≈0 | — |

- **Level ordering.** NMSE increases from L1 to L2 for every probe.
- **Capacity.** The linear-to-MLP step dominates; from MLP-128 to MLP-2048, NMSE decreases by 0.02 (L1) and 0.06 (L2).
- **Fig. 12.** Level-1 distractor NMSE remains an order of magnitude above the paper's value even with MLP-2048, so the discrepancy is not a matter of probe capacity.

### Probe capacity (3-level model, seed 42)

The paper does not report probing results for the 3-level model.

| Probe | Distractor position L1 / L2 / L3 | Geodesic distance L1 / L2 / L3 | Agent position |
|---|---|---|---|
| Linear | 0.446 / 0.672 / 0.903 | 0.816 / 0.854 / 0.863 | 0.002–0.007 |
| MLP-128 | 0.117 / 0.313 / 0.507 | 0.326 / 0.506 / 0.631 | ≈1e-4 |
| MLP-512 (official) | 0.105 / 0.275 / 0.492 | 0.307 / 0.489 / 0.619 | ≈1e-4 |
| MLP-2048 | 0.098 / 0.257 / 0.487 | 0.301 / 0.467 / 0.615 | ≈1e-4 |
| Position-only control (MLP-512) | 0.981 | 0.873 | — |

- **Capacity.** The transition from a linear probe to MLP-128 reduces level-3 distractor NMSE by 0.40. Beyond width 128, level-3 NMSE changes by at most 0.02, which is smaller than the variation across model seeds observed in Fig. 4 (≈0.07).
- **Level ordering.** NMSE increases from L1 to L3 for every probe, including the linear one.
- **Overfitting.** The train–evaluation gap of MLP-2048 (≈0.03) matches that of MLP-512, indicating no additional overfitting.
- **Control.** Learned representations achieve geodesic-distance NMSE of 0.30–0.63, well below the position-only control (0.873). They therefore encode distractor information beyond what follows from the agent's position.

### Probe capacity (4-level model, seed 42)

| Probe | Distractor position L1 / L2 / L3 / L4 | Train–evaluation gap at L4 |
|---|---|---|
| Linear | 0.367 / 0.614 / 0.881 / 0.978 | 0.021 |
| MLP-128 | 0.051 / 0.188 / 0.411 / 0.643 | 0.027 |
| MLP-512 (official) | 0.046 / 0.168 / 0.377 / 0.617 | 0.049 |
| MLP-2048 | 0.043 / 0.152 / 0.339 / 0.579 | 0.042 |

- **Level ordering.** NMSE increases from L1 to L4 for every probe, including the linear one.
- **Probe class.** With the linear probe, L4 distractor NMSE is 0.978, i.e. distractor position is essentially not linearly decodable at the top level, whereas MLP probes recover part of it (0.58–0.64). Agent position remains linearly decodable at all levels (NMSE 0.003–0.009).
- **Capacity.** From MLP-128 to MLP-2048, NMSE decreases by 0.07 at L3 and 0.06 at L4, compared with at most 0.02 for the 3-level model. The magnitude is comparable to the variation across model seeds in Fig. 4 (≈0.07); with a single model seed, it cannot be distinguished from seed variation.
- **Overfitting.** The train–evaluation gap of MLP-2048 does not exceed that of MLP-512, indicating no additional overfitting.

### Random-encoder baseline (3-level architecture, MLP-512 probe)

| Evaluation NMSE, L1 / L2 / L3 | Agent position | Distractor position | Geodesic distance |
|---|---|---|---|
| Untrained encoder | 0.904 / 0.919 / 0.906 | 0.773 / 0.777 / 0.797 | 0.993 / 0.995 / 0.996 |
| Trained H-JEPA | 5e-5 / 7e-5 / 7e-5 | 0.105 / 0.275 / 0.492 | 0.307 / 0.489 / 0.619 |

Untrained features are only weakly informative and show no level dependence, indicating that stacking levels does not by itself remove information. After training, agent position is decodable at every level, and distractor information is largely decodable at L1. At higher levels distractor NMSE approaches, but remains below, the untrained baseline, so the reduction is partial rather than complete.

### Outcome of pre-registered hypotheses

| | Hypothesis | Outcome |
|---|---|---|
| H1 | Agent position is decodable at all capacities; the linear probe is clearly worse; MLP probes saturate | Supported |
| H2 | Distractor position remains at NMSE ≈ 1.0 at all capacities | Not supported (0.10–0.49 for MLP probes) |
| H3 | Geodesic distance is explained mainly by agent position (latent ≈ control) | Not supported (0.30–0.62 vs 0.873) |
| H4 | If MLP-2048 improves training but not evaluation NMSE, the target is treated as not decodable | Not triggered |

The representations contain more distractor information than was anticipated in H2 and H3.

## Discussion

Within the MLP family, the qualitative conclusion of the paper holds: distractor information becomes progressively less decodable at higher levels for every probe width tested, in the 2-, 3- and 4-level models. The absolute values at higher levels of the 4-level model decrease slightly with probe width, so the amount of information recoverable at those levels is somewhat sensitive to probe capacity. The effect of probe class is considerably larger. A linear probe would suggest that even the first level discards most distractor information, whereas nonlinear probes show that it remains largely recoverable. Statements about what a level "discards" should therefore specify the probe used.

## Limitations

- All probes use a single probe seed (42), following the official configuration.
- The capacity sweeps (2-, 3- and 4-level) use a single model seed (42).
- The Fig. 12 distractor discrepancy is unresolved; probe capacity has been ruled out as its cause.
- FourRoom contains no fast-changing proprioceptive variables. The broader claim that higher levels discard fast variables (e.g. leg pose) while retaining position requires evaluation in an embodied environment such as Ant, which H-JEPA also provides.

## Reproducibility

```bash
# Environment and data: follow UPSTREAM_README.md (uv, Python 3.11)
export HJEPA_HOME=/path/to/data

# The released checkpoints do not include the training config required by the probing script;
# recreate it for each checkpoint, e.g.
python probing/scripts/make_train_config.py h_jepa/config/train fourroom_hjepa_l3 42 \
    $HJEPA_HOME/ckpts/fourroom/fourroom_hjepa_l3/seed42/config.yaml

# Single run: <model> <model seed> <probe hidden width>
bash probing/scripts/run_probe.sh l3 42 512

# All runs reported above, executed sequentially
bash probing/scripts/sweep.sh

# Position-only control
python probing/scripts/xy_only_control.py
```

`probing/results/` contains the `metrics.yaml` of every reported run. The linear-probe results for the 3-level model were produced with the official pipeline (`probing/results/C_h0`); all other capacity results were produced with the cached runner.

**Environment.** NVIDIA RTX 4080 Laptop GPU (12 GB) under WSL2; Python 3.11; PyTorch 2.7.1 (CUDA 12.6 build, as the CUDA 12.8 index named upstream was unreachable). Concurrent CUDA processes caused driver faults on this system, so all runs were executed sequentially; `run_probe.sh` retries a failed run up to three times.

## Citation

The model and training code are by the H-JEPA authors. Please cite the original work:

```bibtex
@article{zhang2026hjepa,
  title   = {{H-JEPA}: End-to-End Learning of Hierarchical World Models for Visual Planning},
  author  = {Zhang, Wancong and Terver, Basile and Rabbat, Mike and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2610.06805},
  year    = {2026}
}
```

## License

MIT, inherited from the upstream repository. See [LICENSE](LICENSE).
