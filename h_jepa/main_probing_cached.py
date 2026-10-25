"""Final probing eval with cached frozen-encoder embeddings.

Same heads, optimizer, seeding, batch order and metrics as
final_probing_decoding_eval.run_final_probing_decoding_eval, but every level's
embedding is computed once instead of once per epoch. Valid only because the
probing datasets apply no random augmentation and the encoder is frozen; decoders
are not supported (they would need cached pixels).
"""

import time
from pathlib import Path

import hydra
import lightning as pl
import torch
from loguru import logger as logging
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import final_probing_decoding_eval as fpde


@torch.no_grad()
def _cache_split(heads, dataset, cfg, device):
    """Encode every sample once, in dataset order. Uses its own generator so the
    global RNG (and therefore probe init / batch order) is untouched."""
    loader_cfg = fpde._loader_cfg(cfg, train=False)
    loader = DataLoader(dataset, **loader_cfg, generator=torch.Generator().manual_seed(0))
    target_keys = [f"{col}_level1" for col in heads.target_columns]
    embeds, targets = {}, {key: [] for key in target_keys}
    for batch in loader:
        batch = fpde._move_batch(batch, device)
        fpde._normalize_uint8_images_on_device(batch)
        encoded = fpde._dense_encode_no_stride(heads.model, batch)
        for key, value in encoded.items():
            embeds.setdefault(key, []).append(value)
        for key in target_keys:
            targets[key].append(batch[key])
    embeds = {key: torch.cat(values) for key, values in embeds.items()}
    targets = {key: torch.cat(values) for key, values in targets.items()}
    return embeds, targets


def _random_init_model(trained, train_cfg, seed):
    """Same architecture as the checkpoint, freshly initialized by the official constructor."""
    import stable_pretraining  # noqa: F401  registers the ${eval:...} resolver used by train configs
    from hjepa_utils import create_world_model

    torch.manual_seed(seed)
    model, _ = create_world_model(train_cfg)
    # Compare parameters only: buffers such as latent_action_queue are training-time caches.
    trained_state, random_state = dict(trained.named_parameters()), dict(model.named_parameters())
    if trained_state.keys() != random_state.keys() or any(
        trained_state[k].shape != random_state[k].shape for k in trained_state
    ):
        raise ValueError("Random model does not match the checkpoint architecture.")
    return model


def _cached_batches(embeds, targets, order, batch_size, drop_last, limit=None):
    n_batches = len(order) // batch_size if drop_last else -(-len(order) // batch_size)
    for batch_idx in range(n_batches):
        if limit is not None and batch_idx >= limit:
            break
        idx = order[batch_idx * batch_size:(batch_idx + 1) * batch_size]
        batch = {key: value[idx] for key, value in targets.items()}
        batch["encoded"] = {key: value[idx] for key, value in embeds.items()}
        # _batch_losses always reads pixels_level1; it is only used by decoders.
        batch["pixels_level1"] = torch.zeros(len(idx), 1, 1, device=order.device)
        yield batch


def run(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cache_dir = fpde._cache_dir(cfg)
    model, policy_path, train_cfg = fpde._load_policy(cfg.policy, cache_dir)
    normalizer_artifact = fpde.load_normalizer_artifact(fpde._normalizer_path_for_policy(policy_path))
    if bool(cfg.get("random_init", False)):
        model = _random_init_model(model, train_cfg, int(cfg.get("seed", 42)))
        logging.info("Using a randomly initialized (untrained) world model as the probing baseline")
    model = fpde._freeze_model(model)
    fpde._validate_dense_assumptions(model)
    output_dir = Path(cfg.output_dir).expanduser()

    # Mirror the official call order so the RNG streams line up.
    seed = int(cfg.get("seed", 42))
    pl.seed_everything(seed, workers=True)
    train_dataset, eval_dataset = fpde._build_datasets(cfg, normalizer_artifact)
    generator = torch.Generator().manual_seed(seed)
    train_loader_cfg = fpde._loader_cfg(cfg, train=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heads = fpde.FinalProbeDecodeHeads(model, train_dataset, cfg, train_cfg, normalizer_artifact).to(device)
    if len(heads.decoders):
        raise ValueError("Cached probing does not support decoders.")
    optimizer = fpde._optimizer(heads, cfg)
    train_probe_variances = fpde._probe_variances(heads, train_dataset, device)
    eval_probe_variances = fpde._probe_variances(heads, eval_dataset, device)

    start = time.time()
    train_embeds, train_targets = _cache_split(heads, train_dataset, cfg, device)
    eval_embeds, eval_targets = _cache_split(heads, eval_dataset, cfg, device)
    logging.info(f"Cached embeddings for {len(train_dataset)} train / {len(eval_dataset)} eval samples in {time.time() - start:.0f}s")

    fpde._dense_encode_no_stride = lambda model, batch: batch["encoded"]

    batch_size = int(train_loader_cfg["batch_size"])
    eval_order = torch.arange(len(eval_dataset), device=device)
    train_limit = fpde._limit_batches(cfg, "limit_train_batches")
    eval_limit = fpde._limit_batches(cfg, "limit_eval_batches")
    epochs = int(cfg.epochs)
    final_metrics = {}
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            # The official persistent DataLoader draws its worker base seed once.
            torch.empty((), dtype=torch.int64).random_(generator=generator)
        train_order = torch.randperm(len(train_dataset), generator=generator).to(device)
        # RandomSampler also evaluates a trailing randperm(n)[:0] when it is exhausted.
        torch.randperm(len(train_dataset), generator=generator)
        train_metrics = fpde._run_epoch(
            heads,
            _cached_batches(train_embeds, train_targets, train_order, batch_size, True, train_limit),
            device,
            train_probe_variances,
            optimizer=optimizer,
        )
        eval_metrics = fpde._run_epoch(
            heads,
            _cached_batches(eval_embeds, eval_targets, eval_order, batch_size, False, eval_limit),
            device,
            eval_probe_variances,
        )
        final_metrics = fpde._format_epoch_metrics(
            train_metrics, eval_metrics, fpde.FINAL_PROBING_DECODING_EVAL_PREFIX
        )
        logging.info(
            f"cached probing epoch {epoch}/{epochs}: train_loss={train_metrics.get('loss'):.6f}, "
            f"eval_loss={eval_metrics.get('loss'):.6f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fpde._save_artifacts(heads, output_dir, cfg, final_metrics)
    logging.info(f"Saved cached probing artifacts to {output_dir}")


@hydra.main(version_base=None, config_path="config/probing", config_name=None)
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
