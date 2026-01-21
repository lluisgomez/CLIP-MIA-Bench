#!/usr/bin/env python3
"""
Blind Bag‑of‑Words Baseline (Das et al., SPW 2025)
=================================================

A *model‑agnostic* membership inference “attack” that completely ignores the
target model.  It trains a simple text‑only classifier on a **labeled** subset
of members (M) and non‑members (N) and evaluates on held‑out folds, reproducing
the baseline in:

    Das, D.; Zhang, J.; Tranter, F. 2025.
    *Blind baselines beat membership inference attacks for foundation models*.
    IEEE S&P Workshops (SPW) 2025.

**Important:** This is *not* a realistic attacker—labels for some members are
assumed to be available purely to expose distribution‑shift artefacts in
benchmark datasets.

Dataset
-------
Exactly the same tar‑shard layout as earlier scripts, but you provide **two
lists of shards**:

* `--member_shards`      : shards known to contain *member* samples
* `--nonmember_shards`   : shards known to contain *non‑member* samples

Each sample in a shard must include:
   *.txt   – caption (single‑line UTF‑8)
Images / JSON files are ignored.

Pipeline
--------
1. Load all captions, assign label 1 (member) or 0 (non‑member).
2. Vectorise texts with `TfidfVectorizer` over uni+bi‑grams.
3. Perform k‑fold CV (default k=5).  For each fold:
     * Train LogisticRegression on (k‑1) folds,
     * Evaluate ROC‑AUC and TPR@1 % FPR on the held‑out fold.
4. Report mean±std over folds.

Example
-------
```bash
python blind_bow_attack.py \
  --member_shards      "data/members/*.tar" \
  --nonmember_shards   "data/non_members/*.tar" \
  --cv_folds 5 --fpr_target 0.01
```
"""
from __future__ import annotations
import argparse, glob, os, random
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

try:
    import webdataset as wds
except ImportError:
    raise RuntimeError("webdataset required – pip install webdataset")


def parse_args():
    p = argparse.ArgumentParser("Blind Bag‑of‑Words baseline")
    p.add_argument("--member_shards", required=True, nargs='+')
    p.add_argument("--nonmember_shards", required=True, nargs='+')
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--fpr_target", type=float, default=0.01)
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--max_features", type=int, default=100000,
                   help="Max vocabulary size for TF‑IDF vectoriser")
    p.add_argument("--out_txt", default="outputs/blind_bow_results.txt")
    return p.parse_args()

# ---------------------------------------------------------------------------
def _expand(patterns):
    if isinstance(patterns, (list, tuple)):
        shards = sum((glob.glob(p) for p in patterns), [])
    else:
        shards = glob.glob(patterns)
    shards = sorted(set(shards))
    if not shards:
        raise FileNotFoundError(f"No shards match {patterns}")
    return shards

def load_captions(shards_patterns: List[str], label: int) -> (List[str], List[int]):
    shards = _expand(shards_patterns)
    texts, labels = [], []
    ds = (
        wds.WebDataset(shards, handler=wds.warn_and_continue)
        .to_tuple("txt")   # grab only caption file as bytes
    )
    for (txt_bytes,) in tqdm(ds, desc=f"load {'members' if label else 'non‑members'}", unit="sample"):
        try:
            text = txt_bytes.decode("utf-8", errors="ignore").strip()
        except AttributeError:
            text = str(txt_bytes).strip()
        texts.append(text)
        labels.append(label)
    return texts, labels

def tpr_at_fpr(scores, labels, fpr_target):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, fpr_target, side="right") - 1
    idx = np.clip(idx, 0, len(tpr)-1)
    return tpr[idx]

# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    print("Loading captions …")
    mem_txts, mem_lbls = load_captions(args.member_shards, 1)
    non_txts, non_lbls = load_captions(args.nonmember_shards, 0)

    texts  = np.array(mem_txts + non_txts)
    labels = np.array(mem_lbls + non_lbls)

    print(f"Total samples: {len(texts)}  (members {labels.sum()}, non‑members {(1-labels).sum()})")

    vectoriser = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=args.max_features,
        lowercase=True,
        min_df=2
    )

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.random_state)

    aucs, tprs = [], []
    fold = 0
    for train_idx, test_idx in skf.split(texts, labels):
        fold += 1
        X_train = vectoriser.fit_transform(texts[train_idx])
        X_test  = vectoriser.transform(texts[test_idx])

        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)[:,1]
        auc  = roc_auc_score(y_test, probs)
        tpr  = tpr_at_fpr(probs, y_test, args.fpr_target)

        aucs.append(auc)
        tprs.append(tpr)

        print(f"Fold {fold}:  AUC={auc*100:5.2f}%  TPR@{args.fpr_target*100:.0f}%FPR={tpr*100:5.2f}%")

    print("\n=== Blind Bag‑of‑Words Summary ===")
    print(f"Mean AUC        : {np.mean(aucs)*100:5.2f}% ± {np.std(aucs)*100:4.2f}")
    print(f"Mean TPR@{args.fpr_target*100:.0f}%FPR : {np.mean(tprs)*100:5.2f}% ± {np.std(tprs)*100:4.2f}")

    with open(args.out_txt, "w") as fp:
        fp.write(f"AUCs  : {aucs}\\n")
        fp.write(f"TPRs  : {tprs}\\n")
        fp.write(f"meanAUC={np.mean(aucs)}  stdAUC={np.std(aucs)}\\n")
        fp.write(f"meanTPR={np.mean(tprs)}  stdTPR={np.std(tprs)}\\n")

    print(f"\nSaved per‑fold scores to {args.out_txt}")

if __name__ == "__main__":
    main()

