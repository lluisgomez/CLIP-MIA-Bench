#!/usr/bin/env python3
"""
PyTorch implementation of the **Weakly‑Supervised Attack (WSA)** from Ko et al.
"Practical Membership Inference Attacks Against Large‑Scale Multi‑Modal Models"
(ICCV 2023).  The script reproduces the full WSA pipeline on *any* set of CLIP
(or CLIP‑like) image–text pairs stored in **tar shards** where each sample is a
triplet:
   ▸ *.jpg   – RGB image file
   ▸ *.txt   – UTF‑8 caption (single line)
   ▸ *.json  – metadata (ignored by default)

Main steps
===========
1.  **Feature extraction** – obtain CLIP image & text embeddings and normalised
    cosine similarity for *non‑member* shards and *candidate* (D_all) shards.
2.  **Statistical modelling of non‑members** – compute μ and σ of cosine
    similarities of guaranteed non‑members (D_no).
3.  **Pseudo‑member selection** – label samples whose similarity ≥ μ+λ·σ as
    pseudo‑members (Eq. 2 in the paper).
4.  **Attack model training** – train a lightweight logistic‑regression
    classifier f_attack on concatenated [img_emb, txt_emb, cos] vectors.
5.  **Evaluation** – predict on held‑out evaluation shards and report ROC‑AUC
    and TPR@1 % FPR.

The script is modular: any step can be skipped if its artefacts (.npz feature
files, trained attack model) already exist.

Example usage
-------------
```bash
python wsa_attack.py \
  --model ViT-B-32 --pretrained laion2b_aesthetic_32 \
  --non_member_shards "data/non_members/{0000..0004}.tar" \
  --all_shards        "data/all/{0000..0020}.tar" \
  --eval_member_shards     "data/eval/members/{0000..0001}.tar" \
  --eval_nonmember_shards  "data/eval/non_members/{0000..0001}.tar" \
  --lambda_thr 0.5 --batch_size 64 --out_dir outputs
```
Dependencies
------------
* torch ≥2.0          * open_clip_torch ≥2.0
* webdataset ≥0.2     * scikit‑learn ≥1.4
* pillow, numpy, tqdm

Install with:
```bash
pip install torch open_clip_torch webdataset scikit-learn pillow tqdm
```
"""
from __future__ import annotations
import argparse, os, glob, tarfile, io, json, random
from pathlib import Path

import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    import webdataset as wds
except ImportError as e:
    raise RuntimeError("webdataset library is required.  Install with pip install webdataset")

try:
    import open_clip
except ImportError:
    raise RuntimeError("open_clip_torch library is required.  Install with pip install open_clip_torch")

from sklearn.metrics import roc_auc_score, roc_curve

def parse_args():
    p = argparse.ArgumentParser(description="WSA membership‑inference attack (Ko et al., 2023)")
    # feature extraction
    p.add_argument("--model", default="ViT-B-16", help="CLIP vision backbone, e.g. ViT-B-16")
    p.add_argument("--pretrained", required=True, help="path to model weights or open_clip pretrained tag")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    # data shards
    p.add_argument("--non_member_shards", required=True, nargs='+', help="glob for guaranteed non‑member shards (D_no)")
    p.add_argument("--all_shards", required=True, nargs='+', help="one or more globs for candidate shards (D_all) – may include members")
    p.add_argument("--eval_member_shards", required=True, nargs='+', help="glob for held‑out member shards for evaluation")
    p.add_argument("--eval_nonmember_shards", required=True, nargs='+', help="glob for held‑out non‑member shards for evaluation")
    # WSA hyper‑parameters
    p.add_argument("--lambda_thr", type=float, default=0.5, help="threshold λ (Eq. 2): μ+λ·σ")
    p.add_argument("--epochs", type=int, default=5, help="attack model training epochs")
    # misc
    p.add_argument("--out_dir", default="outputs/wsa", help="directory to store intermediate features & models")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

# ---------------------------------------------------------------------------
#  Data loading helpers
# ---------------------------------------------------------------------------

def _expand_shards(patterns):
    """Turn one or many glob patterns into a sorted unique list of tar paths."""
    if isinstance(patterns, (list, tuple)):
        shard_list = sum([glob.glob(p) for p in patterns], [])
    else:
        shard_list = glob.glob(patterns)
    shard_list = sorted(set(shard_list))
    if not shard_list:
        raise FileNotFoundError(f"No shards match pattern(s): {patterns}")
    return shard_list


def make_dataset(patterns, tokenizer, preprocess):
    """
    WebDataset pipeline → yields (image_tensor, caption_tokens).

    Each tar sample must have *.jpg *.txt *.json ;  we ignore the JSON.
    """
    shards = _expand_shards(patterns)

    ds = (
        wds.WebDataset(shards, handler=wds.warn_and_continue)
        .decode("pil")                       # decode images only
        .to_tuple("jpg", "txt")      # grab all three; we'll drop json
        .map_tuple(
            lambda img: preprocess(img),     # image → tensor
            lambda txt: tokenizer(txt).squeeze(0),      # txt → token tensor
        )
        .map(lambda x: (x[0], x[1])) # return just (img, txt)
    )
    return ds

# ---------------------------------------------------------------------------
#  Feature extraction
# ---------------------------------------------------------------------------

def extract_features(patterns, model, tokenizer, preprocess, args, tag: str):
    """Extract and save CLIP features for all samples in shards_glob.
    Saves two files:
      out_dir/{tag}_features.npz  with keys img, txt (float32) & cos (float32)
      out_dir/{tag}_meta.jsonl    one JSON per line with sample metadata (index, shard, sample_key)
    Returns (img_feats, txt_feats, cos_sim)."""
    out_npz = Path(args.out_dir) / f"{tag}_features.npz"
    if out_npz.exists():
        data = np.load(out_npz)
        return data["img"], data["txt"], data["cos"]

    dataset = make_dataset(patterns, tokenizer, preprocess)
    loader  = DataLoader(dataset, batch_size=args.batch_size, num_workers=4)

    img_feats, txt_feats, cos_sims = [], [], []
    model.eval()
    with torch.no_grad():
        for img, txt in tqdm(loader, desc=f"extract {tag}"):
            img = img.to(args.device)
            txt = txt.to(args.device)
            img_f = model.encode_image(img)
            txt_f = model.encode_text(txt)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            cos = (img_f * txt_f).sum(dim=-1).cpu()
            img_feats.append(img_f.cpu())
            txt_feats.append(txt_f.cpu())
            cos_sims.append(cos)

    img_feats = torch.cat(img_feats).numpy().astype("float32")
    txt_feats = torch.cat(txt_feats).numpy().astype("float32")
    cos_sims  = torch.cat(cos_sims ).numpy().astype("float32")

    np.savez_compressed(out_npz, img=img_feats, txt=txt_feats, cos=cos_sims)
    return img_feats, txt_feats, cos_sims

# ---------------------------------------------------------------------------
#  Attack model (logistic regression on concatenated embeddings)
# ---------------------------------------------------------------------------

class AttackNet(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(1)

# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------

def tpr_at_fpr(probs: np.ndarray, labels: np.ndarray, fpr_threshold: float = 0.01):
    fpr, tpr, _ = roc_curve(labels, probs)
    idx = np.searchsorted(fpr, fpr_threshold, side="right") - 1
    idx = np.clip(idx, 0, len(tpr)-1)
    return tpr[idx]

# ---------------------------------------------------------------------------
#  Main WSA pipeline
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Load CLIP backbone
    print("Loading CLIP backbone…")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # 2. Feature extraction
    img_no, txt_no, cos_no = extract_features(args.non_member_shards, model, tokenizer, preprocess, args, "nonmember")
    img_all, txt_all, cos_all = extract_features(args.all_shards,        model, tokenizer, preprocess, args, "all")

    # 3. Estimate μ and σ of non‑member cosine similarities
    mu_no, sigma_no = cos_no.mean(), cos_no.std()
    print(f"µ_nonmember={mu_no:.4f},  σ_nonmember={sigma_no:.4f}")

    # 4. Select pseudo‑members (Eq. 2)
    thr = mu_no + args.lambda_thr * sigma_no
    pseudo_mask = cos_all >= thr
    print(f"λ = {args.lambda_thr} ⇒ threshold={thr:.4f}.  Pseudo‑members: {pseudo_mask.sum()}/{len(cos_all)}")

    # 5. Build attack dataset (concatenate img|txt|cos)
    X_no   = np.concatenate([img_no, txt_no, cos_no[:,None]], axis=1)
    y_no   = np.zeros(len(X_no), dtype=np.float32)
    X_pmem = np.concatenate([img_all[pseudo_mask], txt_all[pseudo_mask], cos_all[pseudo_mask][:,None]], axis=1)
    y_pmem = np.ones(len(X_pmem), dtype=np.float32)

    X_train = np.vstack([X_no, X_pmem])
    y_train = np.concatenate([y_no, y_pmem])

    print(f"Training attack model on {len(X_train)} samples …")

    # 6. Train logistic regression (implemented as single‑layer NN)
    attack = AttackNet(X_train.shape[1]).to(args.device)
    opt = torch.optim.AdamW(attack.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader  = DataLoader(dataset, batch_size=1024, shuffle=True)
    attack.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad()
            logits = attack(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        print(f"  epoch {epoch+1}/{args.epochs}  avg loss {epoch_loss/len(dataset):.4f}")

    torch.save(attack.state_dict(), Path(args.out_dir)/"attack_model.pt")

    # 7. Evaluation on held‑out shards
    for split_name, shards in [("eval_members", args.eval_member_shards), ("eval_nonmembers", args.eval_nonmember_shards)]:
        extract_features(shards, model, tokenizer, preprocess, args, split_name)

    eval_img_mem, eval_txt_mem, eval_cos_mem = extract_features(args.eval_member_shards,        model, tokenizer, preprocess, args, "eval_members")
    eval_img_non, eval_txt_non, eval_cos_non = extract_features(args.eval_nonmember_shards, model, tokenizer, preprocess, args, "eval_nonmembers")

    X_eval = np.concatenate([
        np.concatenate([eval_img_mem,  eval_txt_mem,  eval_cos_mem[:,None]], axis=1),
        np.concatenate([eval_img_non,  eval_txt_non,  eval_cos_non[:,None]], axis=1)
    ])
    y_eval = np.concatenate([
        np.ones(len(eval_img_mem), dtype=np.float32),
        np.zeros(len(eval_img_non), dtype=np.float32)
    ])

    attack.eval()
    with torch.no_grad():
        logits = []
        for i in range(0, len(X_eval), 4096):
            batch = torch.from_numpy(X_eval[i:i+4096]).to(args.device)
            logits.append(attack(batch).cpu())
    logits = torch.cat(logits).numpy()
    probs = 1/(1+np.exp(-logits))

    auc = roc_auc_score(y_eval, probs)
    tpr = tpr_at_fpr(probs, y_eval, 0.01)
    print("\n=== Evaluation ===")
    print(f"ROC‑AUC    : {auc*100:5.2f} %")
    print(f"TPR@1% FPR : {tpr*100:5.2f} %")

if __name__ == "__main__":
    main()

