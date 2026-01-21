#!/usr/bin/env python3
"""
White-box loss- & gradient-based MIA for CLIP
=============================================
Adapts the 'Loss' and 'GradNorm' attacks from classic literature to
contrastive vision-language models.

Inputs & CLI **identical** to your WSA/consistency scripts:
  --model            OpenCLIP backbone
  --pretrained       checkpoint (weights)
  --non_member_shards   glob(s)   # guaranteed non-members, D_no  (225 k in your case)
  --all_shards          glob(s)   # candidate mix,  D_all
  --eval_member_shards
  --eval_nonmember_shards
Outputs:
  roc-auc / tpr@1 % fpr for both Loss-only, GradNorm-only and combined.
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import torch, numpy as np
from torch import nn
from torch.utils.data import DataLoader
import webdataset as wds, open_clip
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve

# ---------- CLI ----------------------------------------------------------- #
def parse():
    p = argparse.ArgumentParser("White-box loss+grad MIA")
    p.add_argument("--model", default="ViT-B-16")
    p.add_argument("--pretrained", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--non_member_shards", required=True, nargs='+')
    p.add_argument("--all_shards", required=True, nargs='+')
    p.add_argument("--eval_member_shards", required=True, nargs='+')
    p.add_argument("--eval_nonmember_shards", required=True, nargs='+')
    p.add_argument("--lambda_thr", type=float, default=0.0,    # λ=0 == μ  (loss is already >0)
                   help="threshold = μ + λσ (loss) / μ−λσ (grad)")
    p.add_argument("--out_dir", default="outputs/wb_mia")
    return p.parse_args()

# ---------- Data ---------------------------------------------------------- #
def _expand(globs):
    return sorted(set(sum([glob.glob(g) for g in globs], [])))

def make_ds(globs, tok, pre):
    return ( wds.WebDataset(_expand(globs), handler=wds.warn_and_continue)
             .decode("pil")
             .to_tuple("jpg", "txt")
             .map_tuple(lambda im: pre(im),
                        lambda txt: tok(txt).squeeze(0)) )

# ---------- Per-sample InfoNCE loss -------------------------------------- #
def clip_contrastive_loss(img_f, txt_f, t=1/0.07):
    """Return CLIP InfoNCE loss for N pairs (no negative mining; full NxN)."""
    logits = (img_f @ txt_f.T) * t
    labels = torch.arange(len(img_f), device=img_f.device)
    loss_i = nn.functional.cross_entropy(logits, labels, reduction='none')
    loss_t = nn.functional.cross_entropy(logits.T, labels, reduction='none')
    return (loss_i + loss_t)/2           # per-sample scalar

def features_losses(grads=False):
    return None

# ---------- Main --------------------------------------------------------- #
def run_split(name, globs, model, tok, pre, args, need_grad=False):
    ds = make_ds(globs, tok, pre)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=4)
    losses, gnorms = [], []
    model.eval()
    for img, txt in tqdm(dl, desc=f"{name}"):
        img, txt = img.to(args.device), txt.to(args.device)
        with torch.no_grad():
            img_f = model.encode_image(img)
            txt_f = model.encode_text(txt)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)

        # break the graph → no huge activations kept
        img_f = img_f.detach().requires_grad_()
        txt_f = txt_f.detach().requires_grad_()

        t = model.logit_scale.exp()
        loss_vec = clip_contrastive_loss(img_f, txt_f, t)

        if need_grad:
            # total loss so we can back-prop once
            loss_sum = loss_vec.sum()
            g_img, g_txt = torch.autograd.grad(
            loss_sum, [img_f, txt_f], retain_graph=False, create_graph=False)

            # flatten per sample and concatenate
            g = torch.cat([g_img.flatten(1), g_txt.flatten(1)], dim=1)   # (B, D_img+D_txt)
            gnorms.append(g.norm(dim=1).detach().cpu())

        losses.append(loss_vec.detach().cpu())

    losses = torch.cat(losses).numpy()
    gnorms = torch.cat(gnorms).numpy() if need_grad else None
    np.savez_compressed(Path(args.out_dir)/f"{name}.npz",
                        loss=losses, gnorm=gnorms)
    return losses, gnorms

def metric(probs, y, fpr=0.01):
    auc = roc_auc_score(y, probs)
    fpr_, tpr_, _ = roc_curve(y, probs)
    idx = np.searchsorted(fpr_, fpr, 'right')-1
    return auc, tpr_[max(idx,0)]

def main():
    a = parse();  os.makedirs(a.out_dir, exist_ok=True)
    model, _, pre = open_clip.create_model_and_transforms(
        a.model, pretrained=a.pretrained, device=a.device)
    tok = open_clip.get_tokenizer(a.model)

    # ---------- STEP 1: baseline distributions on guaranteed non-members --- #
    nm_loss, nm_g = run_split("nonmember", a.non_member_shards,
                              model, tok, pre, a, need_grad=True)

    mu_L,  sig_L  = nm_loss.mean(), nm_loss.std()
    mu_G,  sig_G  = nm_g.mean(),    nm_g.std()

    def score(loss, g):     # higher ⇒ more likely member
        s_loss = -(loss - mu_L)/sig_L           # lower loss ⇒ higher score
        s_grad = -(g    - mu_G)/sig_G           # lower grad norm ⇒ higher
        return 0.5*s_loss + 0.5*s_grad          # simple average

    # ---------- STEP 2: pseudo members from D_all -------------------------- #
    all_loss, all_g = run_split("allset", a.all_shards,
                                model, tok, pre, a, need_grad=True)
    pm_mask = score(all_loss, all_g) > 0        # weak supervision: keep top half
    X_train = np.vstack([ np.column_stack([nm_loss, nm_g]),
                          np.column_stack([all_loss[pm_mask],
                                           all_g[pm_mask]]) ])
    y_train = np.hstack([ np.zeros(len(nm_loss)),
                          np.ones( pm_mask.sum() ) ])

    # Fit a simple logistic regressor on (loss, gnorm)
    from sklearn.linear_model import LogisticRegression
    reg = LogisticRegression(solver='lbfgs', max_iter=200).fit(X_train, y_train)

    # ---------- STEP 3: evaluation ----------------------------------------- #
    m_loss,  m_g  = run_split("eval_members", a.eval_member_shards,
                              model, tok, pre, a, need_grad=True)
    n_loss,  n_g  = run_split("eval_nonmembers", a.eval_nonmember_shards,
                              model, tok, pre, a, need_grad=True)

    X_test = np.vstack([ np.column_stack([m_loss, m_g]),
                         np.column_stack([n_loss, n_g]) ])
    y_test = np.hstack([ np.ones(len(m_loss)),
                         np.zeros(len(n_loss)) ])
    probs = reg.predict_proba(X_test)[:,1]

    auc, tpr = metric(probs, y_test)
    print(f"\n=== White-box Loss+Grad attack ===")
    print(f"ROC-AUC     : {auc*100:6.2f} %")
    print(f"TPR @1% FPR : {tpr*100:6.2f} %")

if __name__ == "__main__":
    main()

