#!/usr/bin/env python3
"""
Cosine-Similarity Attack (CSA) baseline from

    Ko et al., "Practical Membership Inference Attacks Against
    Large-Scale Multi-Modal Models" (ICCV 2023).

Works with CLIP / OpenCLIP models and tar-sharded datasets of triplets:
   *.jpg   – RGB image
   *.txt   – caption (single‑line UTF‑8)
   *.json  – metadata (ignored)

Pipeline
========
1.  Extract cosine similarities for a calibration set of guaranteed
    non‑members (D_no).
2.  Choose τ so that the false‑positive rate on D_no equals --fpr_target
    (default 1 %).
3.  Apply τ to held‑out evaluation shards, report ROC‑AUC and TPR@FPR.

No training or shadow models needed.
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

try:
    import webdataset as wds
except ImportError:
    raise RuntimeError("webdataset is required – pip install webdataset")

try:
    import open_clip
except ImportError:
    raise RuntimeError("open_clip_torch is required – pip install open_clip_torch")


def parse_args():
    p = argparse.ArgumentParser("Cosine‑Similarity Attack (CSA) for CLIP")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", required=True,
                   help="open_clip tag *or* path to weights")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)

    p.add_argument("--non_member_shards", required=True, nargs='+',
                   help="glob(s) for guaranteed non‑member shards (D_no)")
    p.add_argument("--eval_member_shards", required=True, nargs='+',
                   help="glob(s) for member eval shards")
    p.add_argument("--eval_nonmember_shards", required=True, nargs='+',
                   help="glob(s) for non‑member eval shards")

    p.add_argument("--fpr_target", type=float, default=0.01,
                   help="target FPR when selecting τ (default 1 %)")

    p.add_argument("--out_dir", default="outputs/csa",
                   help="directory for cached features (.npz)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _expand(patterns):
    if isinstance(patterns, (list, tuple)):
        shards = sum((glob.glob(p) for p in patterns), [])
    else:
        shards = glob.glob(patterns)
    shards = sorted(set(shards))
    if not shards:
        raise FileNotFoundError(f"No shards match pattern(s): {patterns}")
    return shards


def make_dataset(patterns, tokenizer, preprocess):
    shards = _expand(patterns)
    return (
        wds.WebDataset(shards, handler=wds.warn_and_continue)
        .decode("pil")
        .to_tuple("jpg", "txt")
        .map_tuple(
            lambda img: preprocess(img),
            lambda txt: tokenizer(txt).squeeze(0)
        )
        .map(lambda tpl: (tpl[0], tpl[1]))
    )


def extract_cos(patterns, model, tokenizer, preprocess, args, tag: str):
    out = Path(args.out_dir) / f"{tag}_cos.npz"
    if out.exists():
        return np.load(out)["cos"]
    ds = make_dataset(patterns, tokenizer, preprocess)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=4)
    cos_all = []
    model.eval()
    with torch.no_grad():
        for imgs, txts in tqdm(loader, desc=f"extract {tag}"):
            imgs, txts = imgs.to(args.device), txts.to(args.device)
            i_f = model.encode_image(imgs)
            t_f = model.encode_text(txts)
            i_f = i_f / i_f.norm(dim=-1, keepdim=True)
            t_f = t_f / t_f.norm(dim=-1, keepdim=True)
            cos = (i_f * t_f).sum(dim=-1).cpu().numpy()
            cos_all.append(cos.astype("float32"))
    cos = np.concatenate(cos_all, axis=0)
    np.savez_compressed(out, cos=cos)
    return cos


def tpr_at_fpr(scores, labels, fpr_target):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = np.clip(idx, 0, len(tpr) - 1)
    return tpr[idx]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading CLIP backbone …")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=args.device
    )
    tokenizer = open_clip.get_tokenizer(args.model)

    cos_no  = extract_cos(args.non_member_shards,   model, tokenizer, preprocess, args, "non_members")
    tau = np.quantile(cos_no, 1.0 - args.fpr_target)
    print(f"Selected τ = {tau:.4f} for target FPR {args.fpr_target*100:.2f}%")

    cos_mem  = extract_cos(args.eval_member_shards,    model, tokenizer, preprocess, args, "eval_members")
    cos_non  = extract_cos(args.eval_nonmember_shards, model, tokenizer, preprocess, args, "eval_nonmembers")

    scores = np.concatenate([cos_mem, cos_non])
    labels = np.concatenate([np.ones_like(cos_mem), np.zeros_like(cos_non)])

    auc = roc_auc_score(labels, scores)
    tpr = tpr_at_fpr(scores, labels, args.fpr_target)

    preds = scores >= tau
    fp = (preds & (labels == 0)).sum() / (labels == 0).sum()
    tp = (preds & (labels == 1)).sum() / (labels == 1).sum()

    print("\n=== CSA Results ===")
    print(f"ROC‑AUC        : {auc*100:5.2f}%")
    print(f"TPR@{args.fpr_target*100:.0f}% FPR : {tpr*100:5.2f}%")
    print(f"Achieved FPR   : {fp*100:5.2f}%")
    print(f"TPR at τ       : {tp*100:5.2f}%")


if __name__ == "__main__":
    main()

