#!/usr/bin/env python3
"""
plot_sim_gap.py — Empirical scaling curve for similarity-gap & AUC
==================================================================
Adds: bootstrap error bands, slope annotation, tidy log ticks.
"""
from __future__ import annotations

import argparse, glob, json, math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch, webdataset as wds
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ----------------------------------------------------------------------
# tiny helpers ----------------------------------------------------------
# ----------------------------------------------------------------------
def _expand(patterns) -> List[str]:
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
            lambda txt: tokenizer(txt).squeeze(0),
        )
    )


def extract_cos(
    patterns, model, tokenizer, preprocess, device, batch_size, cache_dir, tag: str
) -> np.ndarray:
    """Cache-aware extractor → returns 1-D np.float32 array of cosine scores."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / f"{tag}_cos.npz"
    if fname.exists():
        return np.load(fname)["cos"]

    ds      = make_dataset(patterns, tokenizer, preprocess)
    loader  = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=4)
    cos_all = []
    model.eval()
    with torch.no_grad():
        for imgs, txts in tqdm(loader, desc=f"extract {tag}"):
            imgs, txts = imgs.to(device), txts.to(device)
            i_feat = model.encode_image(imgs)
            t_feat = model.encode_text(txts)
            # ℓ2-normalise
            i_feat /= i_feat.norm(dim=-1, keepdim=True)
            t_feat /= t_feat.norm(dim=-1, keepdim=True)
            cos = (i_feat * t_feat).sum(dim=-1).cpu().numpy()
            cos_all.append(cos.astype("float32"))

    cos = np.concatenate(cos_all, axis=0)
    np.savez_compressed(fname, cos=cos)
    return cos


def load_model_and_transforms(model_path: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=model_path, device=device, jit=False
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(device)
    return model, tokenizer, preprocess


def bootstrap_stats(
    cos_mem: np.ndarray,
    cos_non: np.ndarray,
    n_boot: int = 250,
    rng: np.random.Generator = np.random.default_rng(0),
) -> Tuple[float, float, float, float]:
    """Return (gap_mean, gap_std, auc_mean, auc_std) via bootstrap."""
    m, n = len(cos_mem), len(cos_non)
    gaps, aucs = [], []
    for _ in range(n_boot):
        mem_samp = cos_mem[rng.integers(0, m, m)]
        non_samp = cos_non[rng.integers(0, n, n)]
        gaps.append(float(mem_samp.mean() - non_samp.mean()))
        aucs.append(
            roc_auc_score(
                np.concatenate([np.ones_like(mem_samp), np.zeros_like(non_samp)]),
                np.concatenate([mem_samp, non_samp]),
            )
        )
    return (
        float(np.mean(gaps)),
        float(np.std(gaps, ddof=1)),
        float(np.mean(aucs)),
        float(np.std(aucs, ddof=1)),
    )


# ----------------------------------------------------------------------
# main -----------------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("Similarity-gap scaling plot (CSA)")
    p.add_argument("--model4M", required=True)
    p.add_argument("--model40M", required=True)
    p.add_argument("--model400M", required=True)
    p.add_argument("--eval_member_shards", required=True)
    p.add_argument("--eval_nonmember_shards", required=True)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--no_gaussian_curve", action="store_true")
    args = p.parse_args()

    sizes = {"4M": 4_000_000, "40M": 40_000_000, "400M": 400_000_000}
    ckpts = {"4M": args.model4M, "40M": args.model40M, "400M": args.model400M}

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, float]] = {}

    # -------------------------------------------------- extraction + stats
    for tag in ["4M", "40M", "400M"]:
        print(f"\n=== {tag} model ===")
        model, tokenizer, preprocess = load_model_and_transforms(ckpts[tag], args.device)

        cos_mem = extract_cos(
            args.eval_member_shards, model, tokenizer, preprocess,
            args.device, args.batch_size, out_dir, f"{tag}_mem"
        )
        cos_non = extract_cos(
            args.eval_nonmember_shards, model, tokenizer, preprocess,
            args.device, args.batch_size, out_dir, f"{tag}_non"
        )

        # pooled within-class stdev (σ) = average of the two class stds
        sigma = 0.5 * (cos_mem.std() + cos_non.std())
        
        gap_mu, gap_sd, auc_mu, auc_sd = bootstrap_stats(cos_mem, cos_non)
        results[tag] = dict(
            gap=gap_mu,  gap_sd=gap_sd,
            auc=auc_mu,  auc_sd=auc_sd,
            sigma=sigma, 
        )


        print(f"Δ = {gap_mu:.4f} ± {gap_sd:.4f}   CSA-AUC = {auc_mu*100:.2f} ± {auc_sd*100:.2f}%")

    # ------------------------------------------------------------------ plotting
    xs       = np.array([1 / sizes[t] for t in ("4M", "40M", "400M")])
    gaps     = np.array([results[t]["gap"] for t in ("4M", "40M", "400M")])
    gap_err  = np.array([results[t]["gap_sd"] for t in ("4M", "40M", "400M")])
    aucs     = np.array([results[t]["auc"] for t in ("4M", "40M", "400M")])
    auc_err  = np.array([results[t]["auc_sd"] for t in ("4M", "40M", "400M")])

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # ----- left: Δ vs 1/N ------------------------------------------------
    ax[0].loglog(xs, gaps, "o-", lw=2, label="empirical Δ")
    ax[0].fill_between(xs, gaps - gap_err, gaps + gap_err, alpha=0.2)

    # power-law fit Δ ≈ C/N  (slope ≈ 1)
    slope, logC = np.polyfit(np.log10(xs), np.log10(gaps), 1)
    C_fit       = 10 ** logC
    xs_dense    = np.logspace(np.log10(xs.min()), np.log10(xs.max()), 200)
    ax[0].loglog(xs_dense, C_fit * xs_dense, "--", label="fit Δ∝1/N")

    # slope annotation
    x_annot, y_annot = xs_dense[len(xs_dense)//2], C_fit * xs_dense[len(xs_dense)//2]
    ax[0].annotate(f"slope ≈ {abs(slope):.2f}",
                   xy=(x_annot, y_annot),
                   xytext=(1.8*x_annot, 1.5*y_annot),
                   arrowprops=dict(arrowstyle="->", lw=0.8),
                   fontsize=9)

    ax[0].set_xlabel("1 / training-set size (1/N)")
    ax[0].set_ylabel("mean similarity gap Δ")
    ax[0].set_title("Δ vs. 1/N (log-log)")
    # consistent ticks
    ax[0].set_xticks([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7])
    ax[0].get_xaxis().set_major_formatter(plt.FormatStrFormatter("%1.0e"))


    
    
    # ----- right: AUC vs 1/N ----------------------------------------------------
    ax[1].semilogx(xs, aucs, "o-", lw=2, label="CSA (empirical)")
    ax[1].fill_between(xs, aucs - auc_err, aucs + auc_err, alpha=0.2)

    if not args.no_gaussian_curve:
        try:
            from scipy.stats import norm
            #sigma_hat = np.mean([results[t]["gap_sd"] for t in results])
            sigma_hat = np.mean([results[t]["sigma"]   for t in results])
            aucs_fit  = norm.cdf((C_fit * xs_dense) / (sigma_hat * np.sqrt(2)))
        except ModuleNotFoundError:
            vec_erf   = np.vectorize(math.erf, otypes=[float])
            sigma_hat = np.mean([results[t]["sigma"]   for t in results])
            aucs_fit  = 0.5 * (1 + vec_erf((C_fit * xs_dense)/(np.sqrt(2)*sigma_hat)))

        aucs_fit = np.clip(aucs_fit, None, 0.999)          # cosmetic: avoid line at y=1
        ax[1].semilogx(xs_dense, aucs_fit, "--", label="analytic Φ(Δ/√2σ)")

    ax[1].set_xlabel("1 / training-set size (1/N)")
    ax[1].set_ylabel("AUC")
    ax[1].set_ylim(0.45, 1.0)
    ax[1].set_title("CSA AUC vs. 1/N")
    ax[1].set_xticks([1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7])
    ax[1].get_xaxis().set_major_formatter(plt.FormatStrFormatter("%1.0e"))
    ax[1].legend(frameon=False)

    fig.tight_layout()
    out_png = out_dir / "sim_gap_scaling.png"
    fig.savefig(out_png, dpi=300)
    print(f"\nPlot saved → {out_png}")

    with open(out_dir / "sim_gap_results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Raw numbers dumped → {out_dir / 'sim_gap_results.json'}")


if __name__ == "__main__":
    main()

