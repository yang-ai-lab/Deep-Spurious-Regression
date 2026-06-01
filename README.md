# Deep Spurious Regression

[![Paper](https://img.shields.io/badge/paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Deep--Spurious--Regression-FFD21E)](https://huggingface.co/yang-ai-lab/Deep-Spurious-Regression)
[![Webpage](https://img.shields.io/badge/%F0%9F%8C%90%20website-demo-blue)](https://yang-ai-lab.github.io/Deep-Spurious-Regression/)
[![License](https://img.shields.io/badge/%F0%9F%93%9C%20license-MIT-green)](LICENSE)

Real-world regression often exhibits shortcuts: attributes spuriously correlated with continuous targets during training that become unreliable under deployment shifts. Existing work on spurious correlations focuses primarily on classification, where labels are categorical and groups are naturally defined. However, many real-world tasks require **continuous prediction**, where hard label boundaries or discrete group-label pairs do not exist.

We define **Deep Spurious Regression (DSR)** as learning from regression data with attribute-label confounding, addressing continuous spurious correlations, and generalizing to all attribute-label combinations at test time. Motivated by the intrinsic difference between classification and regression shortcuts, we propose to exploit the similarity among spurious attributes in both label and feature spaces — accounting for nearby targets and related groups while calibrating both label and learned feature distributions across attributes. Extensive experiments spanning computer vision, environmental sensing, and LLM regression verify the superior performance of our strategies.

![DSR Transferability](teaser/DSR_new.gif)

## 📰 News
- **[2026-XX-XX]** Paper released on [arXiv](https://arxiv.org/abs/XXXX.XXXXX)!
- **[2026-05-27]** [Project website](https://yang-ai-lab.github.io/Deep-Spurious-Regression/) is live!
- **[2026-05-26]** Code released on GitHub, and model released on [HuggingFace](https://huggingface.co/yang-ai-lab/Deep-Spurious-Regression)!

## 💿 Installation

```bash
git clone https://github.com/yang-ai-lab/Deep-Spurious-Regression.git
pip install -r requirements.txt
```

### Dependencies

- torch
- torchvision
- numpy
- pandas
- Pillow
- huggingface_hub

## 📂 Data Preparation

Once installed, prepare your dataset as follows.

**UTKFace** — download the images from [susanqq.github.io/UTKFace](https://susanqq.github.io/UTKFace/) and place them under:
```
data/UTKFace/images/*.jpg
```

The train/val/test split CSVs are already included in `data/` — no additional setup needed.

## 🔍 Evaluation

### Download Checkpoints

Checkpoints are hosted on HuggingFace at [`yang-ai-lab/Deep-Spurious-Regression`](https://huggingface.co/yang-ai-lab/Deep-Spurious-Regression).

| Dataset | Method | File |
|---------|--------|------|
| UTKFace | LMDS | `UTKFace/LMDS.pth` |
| UTKFace | FMDS | `UTKFace/FMDS.pth` |
| UTKFace | LMDS+FMDS | `UTKFace/LMDS_FMDS.pth` |

More checkpoints for other datasets will be released soon.

Download a checkpoint by specifying the dataset and method file:

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="yang-ai-lab/Deep-Spurious-Regression",
    filename="<DATASET>/<METHOD_FILE>"  # e.g. "UTKFace/FMDS.pth"
)
```

Or via CLI:
```bash
huggingface-cli download yang-ai-lab/Deep-Spurious-Regression <DATASET>/<METHOD_FILE>
```

### Run Evaluation

To reproduce the results in the original paper, use `download_and_evaluate.py` to automatically download and evaluate without manually specifying checkpoint paths:

```bash
# evaluate all methods on UTKFace
python download_and_evaluate.py --dataset UTKFace --data_folder ./data

# evaluate one specific method on UTKFace
python download_and_evaluate.py --dataset UTKFace --method FMDS.pth --data_folder ./data
```

Alternatively, after downloading a checkpoint manually (see [Download Checkpoints](#download-checkpoints)), run:

```bash
python evaluate.py --dataset <DATASET> --ckpt <CKPT_PATH> --data_folder <DATA_ROOT>
```

For example:
```bash
python evaluate.py --dataset UTKFace --ckpt UTKFace/FMDS.pth --data_folder ./data
```

## 📊 Results

Test L1 errors (↓ lower is better) for our proposed methods — **LMDS**, **FMDS**, and **LMDS+FMDS** — on UTKFace.

| Dataset | LMDS | FMDS | LMDS+FMDS |
|---------|------|------|-----------|
| UTKFace | 7.039 | 6.961 | 7.032 |

## 🔁 Reproducibility Notes

This repo is intentionally lightweight and focuses on **inference** for one dataset (UTKFace). Full training code and evaluation on additional datasets will be released upon the acceptance of the paper.

## 📝 Citation

If you use this work in your research, please cite the paper:

```bibtex
@article{xu2026shortcut,
  title   = {Shortcut to Nowhere: Demystifying Deep Spurious Regression},
  author  = {Xu, Guanrong and Li, Jessica and Wang, Hao and Yang, Yuzhe},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
