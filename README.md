# H-JEPA FourRoom probing: reproduction and probe-capacity study

[中文版](README.zh-CN.md) · [Original upstream README](UPSTREAM_README.md)

[H-JEPA](https://arxiv.org/abs/2610.06805) reads off what each level of its hierarchy remembers by training a small probe network on frozen features. Its Fig. 4 concludes that higher levels keep the agent's position but progressively drop information about distractors ("selective abstraction"). The paper uses a single probe architecture (an MLP with 512 hidden units). This study asks whether that conclusion depends on how strong the probe is.

- **Upstream:** [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA) at commit `97054ec` (MIT). Official FourRoom checkpoints from Hugging Face, sha256 verified.
- **My changes:** three commits on top of upstream (`git log 97054ec..HEAD -- h_jepa`), plus `probing/` and this README (the original is kept as [`UPSTREAM_README.md`](UPSTREAM_README.md)).
- **Not included:** datasets and checkpoints (download them as described in [`UPSTREAM_README.md`](UPSTREAM_README.md)).

## TL;DR

1. **Fig. 4 reproduces.** On the 4-level model over 3 model seeds, distractor-position error per level is 0.053 / 0.167 / 0.340 / 0.553, against roughly 0.06 / 0.19 / 0.37 / 0.58 read from the paper's plot.
2. **The layer-wise trend holds for MLP probes of any tested width.** On the 3-level model, distractor error rises from level 1 to level 3 for MLP probes with 128, 512 and 2048 hidden units, and widths above 128 change the numbers by at most 0.02.
3. **A linear probe gives a much weaker picture.** With a linear probe, distractor error is 0.45 / 0.67 / 0.90, so how much information appears "dropped" depends strongly on the probe class.
4. **The information comes from training.** A randomly initialised encoder with the same architecture gives distractor error around 0.78 at every level and agent-position error around 0.91.
5. **One mismatch remains.** For the 2-level model (Fig. 12), level 2 matches the paper but level 1 does not (0.156 vs about 0.01). The cause is unknown.

## Setup

- **Model:** official H-JEPA FourRoom checkpoints (2, 3 and 4 levels), frozen.
- **Probe:** official probing config: AdamW, lr 1e-3, weight decay 1e-4, batch 128, 60 epochs, probe seed 42. One hidden layer with LayerNorm and GELU; only the hidden width changes (0 = linear).
- **Targets:** agent position (`xy`), distractor position (`distractor0_xy`), geodesic distance to the nearest distractor (`min_distractor_dist`).
- **Metric:** normalised MSE (NMSE) on the evaluation split. 1.0 means no better than predicting the mean; lower is better.
- **Hardware:** RTX 4080 Laptop GPU 12 GB, WSL2. Python 3.11, torch 2.7.1+cu126.

### Code changes (in `h_jepa/`)

| Commit | Change |
|---|---|
| Make probe hidden_dim configurable | `probe_hidden_dim` option; `0` = linear; default 512 keeps the original code path |
| Add cached-embedding probing runner | `main_probing_cached.py`: runs the frozen encoder once and caches its outputs, then trains the same probe heads with the same RNG sequence and metrics. |
| Cached probing: optional random_init baseline | `+random_init=true` builds the same architecture untrained, via the official constructor |

**Check that the cached runner is faithful:** on the 3-level model (seed 42, MLP512 probe), all 18 NMSE values from the cached runner equal the official pipeline's exactly (max difference 0). See `probing/scripts/compare_metrics.py`.

## Results

### 1. Paper Fig. 4: 4-level model, distractor position

| | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| paper (read from plot) | ≈0.06 ± 0.02 | ≈0.19 ± 0.05 | ≈0.37 ± 0.08 | ≈0.58 ± 0.07 |
| this repo, 3 model seeds, mean ± SE | 0.053 ± 0.020 | 0.167 ± 0.050 | 0.340 ± 0.073 | 0.553 ± 0.072 |

Agent position is near-perfect at every level (NMSE ≈ 1e-4), as in the paper.

### 2. Paper Fig. 12: 2-level model (seed 42)

| | agent L1 | agent L2 | distractor L1 | distractor L2 |
|---|---|---|---|---|
| paper | ≈0 | ≈0 | ≈0.01 | ≈0.33 ± 0.01 |
| this repo | 4e-5 | 7e-5 | **0.156** | 0.341 |

Level 2 matches; level-1 distractor error does not, and I have not found why. The Fig. 12 caption says probe heads differ between environments, so the probe used there may not be the one in the released config.

### 3. Probe capacity: 3-level model (seed 42)

The paper reports no probing for the 3-level model.

| probe (parameters) | distractor xy L1 / L2 / L3 | geodesic distance L1 / L2 / L3 | agent xy |
|---|---|---|---|
| linear (386) | 0.446 / 0.672 / 0.903 | 0.816 / 0.854 / 0.863 | 0.002–0.007 |
| MLP 128 (25k) | 0.117 / 0.313 / 0.507 | 0.326 / 0.506 / 0.631 | ≈1e-4 |
| MLP 512, official (101k) | 0.105 / 0.275 / 0.492 | 0.307 / 0.489 / 0.619 | ≈1e-4 |
| MLP 2048 (403k) | 0.098 / 0.257 / 0.487 | 0.301 / 0.467 / 0.615 | ≈1e-4 |

- Going from linear to 128 hidden units is the big step (level-3 distractor error falls by 0.40). Beyond 128 the probe saturates: level-3 values move by at most 0.02, less than the spread across model seeds (about 0.07 in Fig. 4).
- The ordering L1 < L2 < L3 holds for every probe, so the Fig. 4 conclusion is robust to MLP width. With a linear probe, though, even level 1 looks like it has lost much of the distractor information.
- The widest probe does not overfit: its train–eval gap (about 0.03) is the same as for the 512 probe.
- This sweep uses a single model seed.

**Control: what can be predicted from the agent's position alone?** The same MLP512 probe trained on the true agent position plus the closed-door index (no learned features) gives distractor error 0.981 and geodesic-distance error 0.873. The learned features beat this by a wide margin (geodesic distance 0.31–0.62), so they carry real distractor information, not just a proxy through the agent's position.

### 4. Random-encoder baseline (3-level architecture, untrained, MLP512 probe)

| eval NMSE L1 / L2 / L3 | agent xy | distractor xy | geodesic distance |
|---|---|---|---|
| random encoder | 0.904 / 0.919 / 0.906 | 0.773 / 0.777 / 0.797 | 0.993 / 0.995 / 0.996 |
| trained H-JEPA | 5e-5 / 7e-5 / 7e-5 | 0.105 / 0.275 / 0.492 | 0.307 / 0.489 / 0.619 |

Random features carry little information, and it does not change across levels, so stacking levels does not by itself remove information. Training makes the agent position near-perfect at every level and the distractor readable at level 1. At higher levels distractor error moves back toward the random-encoder value but stays below it, so the drop is partial.

## Pre-registration

Before seeing any full result of the capacity sweep, I wrote down expectations in [`probing/PREREG.md`](probing/PREREG.md). How they turned out:

| | Expectation | Outcome |
|---|---|---|
| H1 | agent position decodable at every capacity; linear clearly worse; MLPs saturated | **confirmed** |
| H2 | distractor position stays ≈1.0 at every capacity ("absent from the latent") | **rejected**: 0.10–0.49 with MLP probes |
| H3 | geodesic distance is mostly explained by agent position (latent ≈ control) | **rejected**: latent 0.30–0.62 vs control 0.873 |
| H4 | if the widest probe improves train but not eval, count it as "not decodable" | not triggered: train–eval gap unchanged |

H2 and H3 were wrong: the features hold more distractor information than I expected.

## Reproduce

```bash
# 1. environment and data: follow UPSTREAM_README.md (uv, Python 3.11); then
export HJEPA_HOME=/path/to/data

# 2. the HF checkpoints ship without the training config the probe script needs; recreate it, e.g.
python probing/scripts/make_train_config.py h_jepa/config/train fourroom_hjepa_l3 42 \
    $HJEPA_HOME/ckpts/fourroom/fourroom_hjepa_l3/seed42/config.yaml

# 3. one probing run: <model> <model seed> <probe hidden width>
bash probing/scripts/run_probe.sh l3 42 512

# all runs in this document, sequentially
bash probing/scripts/sweep.sh

# 4. the agent-position-only control
python probing/scripts/xy_only_control.py
```

`probing/results/` holds the `metrics.yaml` of every run reported here. The linear-probe row for the 3-level model comes from the official pipeline (`results/C_h0`); the other capacity rows come from the cached runner.

## Notes

- On the test machine (laptop GPU under WSL2), running two CUDA processes at once crashed reliably (Windows logged `nvlddmkm` "Graphics FECS Exception" events at the crash times). All runs here were sequential, and `run_probe.sh` retries a failed run up to 3 times.
- The torch build is cu126 rather than the cu128 the upstream README names, because the official wheel index was unreachable from the test machine.

## Next steps

- FourRoom has no fast-changing body variables such as leg pose. Testing the claim "higher levels drop fast variables like leg pose but keep position" needs an embodied environment (H-JEPA also ships Ant).

## Acknowledgements

All model and training code is by the H-JEPA authors. If you use it, please cite:

```bibtex
@article{zhang2026hjepa,
  title   = {{H-JEPA}: End-to-End Learning of Hierarchical World Models for Visual Planning},
  author  = {Zhang, Wancong and Terver, Basile and Rabbat, Mike and LeCun, Yann and Balestriero, Randall},
  journal = {arXiv preprint arXiv:2610.06805},
  year    = {2026}
}
```

## License

MIT, inherited from upstream. See [LICENSE](LICENSE).
