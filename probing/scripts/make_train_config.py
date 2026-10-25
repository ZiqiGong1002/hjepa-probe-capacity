# Recreate the training config.yaml that main_hjepa.py would have saved (unresolved hydra cfg).
# The HF checkpoints ship without it, but the probing script needs it next to the checkpoint.
# Usage: python make_train_config.py <repo>/h_jepa/config/train fourroom_hjepa_l3 42 \
#            $HJEPA_HOME/ckpts/fourroom/fourroom_hjepa_l3/seed42/config.yaml
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
cfg_dir, name, seed, out = sys.argv[1:5]
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    cfg = compose(config_name=name, overrides=[f"seed={seed}"])
OmegaConf.save(cfg, out)
print("wrote", out, "num_levels =", cfg.num_levels)
