# CLIP-MIA-Bench

[![Paper](https://img.shields.io/badge/paper-PDF-red.svg)](https://ojs.aaai.org/index.php/AAAI/article/view/39276/43237)
[![Dataset](https://img.shields.io/badge/data-HuggingFace-yellow.svg)](https://huggingface.co/datasets/CLIP-MIA-Bench/clip-mia-bench-data)
[![Models](https://img.shields.io/badge/models-HuggingFace-yellow.svg)](https://huggingface.co/CLIP-MIA-Bench/clip-mia-bench-models)


CLIP-MIA-Bench is a benchmarking suite for evaluating **membership inference attacks (MIA)** against CLIP-style vision–language models.
It provides standardized baselines, scripts, and reproduction code accompanying the paper:

**Rethinking Membership Inference Attacks for CLIP**  
Lluis Gomez, AAAI 2026

The goal of this repository is to facilitate **reproducible, fair, and extensible evaluation** of MIA methods under different CLIP model variants, datasets, and attack assumptions.

---

## Installation & Setup

### Clone the Repository

```bash
git clone https://github.com/lluisgomez/CLIP-MIA-Bench.git
cd CLIP-MIA-Bench
```

### Download Assets

Make the download script executable and run it:

```bash
chmod +x download_assets.sh
./download_assets.sh
```

This will automatically download and place all required data and models.

---

## Data shards and splits

The dataset is organized into WebDataset tar shards containing image–text pairs.
For benchmarking, shards are split into two main sets.

### Evaluation (hold-out) set

A fixed held-out evaluation set is used by all baselines and is never used for
attack construction or training.

It consists of:
- Member shards:
  clip-mia-bench-data/data/member/{00000001..00000003}.tar
- Non-member shards:
  clip-mia-bench-data/data/nonmember/{00043001..00043003}.tar

These shards contain approximately 27,000 evaluation member samples and 27,000 evaluation non-member samples.

### Other samples

All remaining shards are grouped as other samples, consisting of:
- 23 member shards
- 23 non-member shards
(~400,000 samples in total)

These shards can be used differently depending on the attack:

- CSA / one-sided attacks:
  Known non-member shards are used as one-sided information to model the
  non-member distribution.

- WSA (Weakly-Supervised Attack):
  Known non-members are used to estimate non-member statistics, while a larger
  mixed pool of member and non-member shards is used to generate pseudo-members
  and train the attack classifier.

Exact shard usage is specified via command-line arguments for each baseline.

---

## Running Baselines

The following commands reproduce the baseline membership inference attacks evaluated in the paper.

```bash
# Blind attack
python baselines/blind.py \
  --member_shards clip-mia-bench-data/data/member/{00000001..00000003}.tar \
  --nonmember_shards clip-mia-bench-data/data/nonmember/{00043001..00043003}.tar

# CSA (one-sided attack)
python baselines/csa.py \
  --model ViT-B-16 \
  --pretrained clip-mia-bench-models/model/400M/checkpoints/epoch_latest.pt \
  --non_member_shards clip-mia-bench-data/data/nonmember/{00043004..00043026}.tar \
  --eval_member_shards clip-mia-bench-data/data/member/{00000001..00000003}.tar \
  --eval_nonmember_shards clip-mia-bench-data/data/nonmember/{00043001..00043003}.tar

# MCD
python baselines/mcd.py \
  --model ViT-B-16 \
  --pretrained clip-mia-bench-models/model/400M/checkpoints/epoch_latest.pt \
  --eval_member_shards clip-mia-bench-data/data/member/{00000001..00000003}.tar \
  --eval_nonmember_shards clip-mia-bench-data/data/nonmember/{00043001..00043003}.tar

# WSA
python baselines/wsa.py \
  --model ViT-B-16 \
  --pretrained clip-mia-bench-models/model/400M/checkpoints/epoch_latest.pt \
  --non_member_shards clip-mia-bench-data/data/nonmember/{00043004..00043026}.tar \
  --all_shards \
    clip-mia-bench-data/data/nonmember/{00043004..00043026}.tar \
    clip-mia-bench-data/data/member/{00000004..00000026}.tar \
  --eval_member_shards clip-mia-bench-data/data/member/{00000001..00000003}.tar \
  --eval_nonmember_shards clip-mia-bench-data/data/nonmember/{00043001..00043003}.tar

# WSA (grad-loss variant)
python baselines/wsa_grad_loss.py \
  --model ViT-B-16 \
  --pretrained clip-mia-bench-models/model/400M/checkpoints/epoch_latest.pt \
  --non_member_shards clip-mia-bench-data/data/nonmember/{00043004..00043026}.tar \
  --all_shards \
    clip-mia-bench-data/data/nonmember/{00043004..00043026}.tar \
    clip-mia-bench-data/data/member/{00000004..00000026}.tar \
  --eval_member_shards clip-mia-bench-data/data/member/{00000001..00000003}.tar \
  --eval_nonmember_shards clip-mia-bench-data/data/nonmember/{00043001..00043003}.tar
```

---

## Reproducing Figure 2: Similarity-Gap Scaling

To reproduce **Figure 2 (Similarity-gap scaling)** from the paper, run:

```bash
python plot_sim_gap.py --model4M clip-mia-bench-models/model/4M/checkpoints/epoch_latest.pt --model40M clip-mia-bench-models/model/40M/checkpoints/epoch_latest.pt --model400M clip-mia-bench-models/model/400M/checkpoints/epoch_latest.pt --eval_member_shards clip-mia-bench-data/data/member/00000001.tar --eval_nonmember_shards clip-mia-bench-data/data/nonmember/00043001.tar 
```

---

## Citation

If you use this repository or build upon this work, please cite:

```bibtex
@inproceedings{gomez2026rethinking,
  title     = {Rethinking Membership Inference Attacks for CLIP},
  author    = {Gomez, Lluis},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

