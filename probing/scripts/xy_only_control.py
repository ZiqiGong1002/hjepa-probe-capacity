# Control: how well can each probe target be predicted from the TRUE agent xy (+ closed door) alone?
# Same probe head as the baseline (MLP 512, LN, GELU), same normalization and NMSE definition.
import os, sys, h5py, numpy as np, torch, torch.nn as nn
torch.set_num_threads(2); torch.manual_seed(42)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "h_jepa"))
from models.probers import build_prober
H = os.environ["HJEPA_HOME"]
norm = torch.load(f"{H}/ckpts/fourroom/fourroom_hjepa_l3/seed42/normalizer.pt", weights_only=False)["stats"]
def load(name):
    with h5py.File(f"{H}/{name}.h5", "r") as f:
        xy = f["xy"][:].astype(np.float32); door = f["closed_door_idx"][:]
        T = {c: f[c][:].reshape(len(xy), -1).astype(np.float32) for c in ["xy", "distractor0_xy", "min_distractor_dist"]}
    oh = np.eye(5, dtype=np.float32)[door + 1]  # -1..3 -> one-hot
    x = np.concatenate([(xy - np.array(norm["xy"]["mean"][0])) / np.array(norm["xy"]["std"][0]), oh], 1)
    return torch.tensor(x), {c: torch.tensor(v) for c, v in T.items()}
xtr, Ttr = load("fourroom_tp35_d1_probing"); xva, Tva = load("fourroom_tp35_d1_probing_val")
res = {}
for c in Ttr:
    mu = torch.tensor(norm[c]["mean"]).reshape(1, -1); sd = torch.tensor(norm[c]["std"]).reshape(1, -1)
    ytr = (Ttr[c] - mu) / sd
    m = build_prober(input_dim=xtr.shape[1], output_dim=ytr.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(15):
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr) - 127, 128):
            b = perm[i:i + 128]; loss = nn.functional.mse_loss(m(xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = {}
        for split, x, T in (("train", xtr, Ttr), ("eval", xva, Tva)):
            pred = m(x) * sd + mu; y = T[c]
            out[split] = float((((pred - y) ** 2).mean(0) / y.var(0)).mean())
    res[c] = out; print(c, out, flush=True)
print("XYCTRL_DONE", res)
