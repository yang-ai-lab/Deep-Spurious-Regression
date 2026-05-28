# Deep Spurious Regression

Inference code for **Deep Spurious Regression (DSR)**. Load a dataset and a pre-trained checkpoint to reproduce test results. Currently supports the UTKFace dataset — more datasets coming soon.

## Install

```bash
git clone https://github.com/yang-ai-lab/Deep-Spurious-Regression.git
pip install -r requirements.txt
```

## Prepare Data

Place images under your local data directory as follows:

**UTKFace** — download from [susanqq.github.io/UTKFace](https://susanqq.github.io/UTKFace/)
```
data/UTKFace/images/*.jpg
```

The train/val/test split CSVs are already included in `data/`.

## Download Checkpoints

Checkpoints are hosted on HuggingFace at [`yang-ai-lab/Deep-Spurious-Regression`](https://huggingface.co/yang-ai-lab/Deep-Spurious-Regression).

Currently available checkpoints (more datasets coming soon):

| Dataset | Method | File |
|---------|--------|------|
| UTKFace | LMDS | `UTKFace/LMDS.pth` |
| UTKFace | FMDS | `UTKFace/FMDS.pth` |
| UTKFace | LMDS+FMDS | `UTKFace/LMDS_FMDS.pth` |

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

## Evaluate

To reproduce the results in the original paper, follow the steps below.

**Step 1.** Download a checkpoint from HuggingFace (see [Download Checkpoints](#download-checkpoints)):

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="yang-ai-lab/Deep-Spurious-Regression",
    filename="<DATASET>/<METHOD_FILE>"  # e.g. "UTKFace/FMDS.pth"
)
```

**Step 2.** Run evaluation:

```bash
python evaluate.py --dataset <DATASET> --ckpt <CKPT_PATH> --data_folder <DATA_ROOT>
```

For example:
```bash
python evaluate.py --dataset UTKFace --ckpt UTKFace/FMDS.pth --data_folder ./data
```

**Or use `download_and_evaluate.py` to automatically download and evaluate** without manually specifying checkpoint paths:

```bash
# evaluate all methods on UTKFace
python download_and_evaluate.py --dataset UTKFace --data_folder ./data

# evaluate one specific method on UTKFace
python download_and_evaluate.py --dataset UTKFace --method FMDS.pth --data_folder ./data
```

## Results

Test L1 errors (↓ lower is better) for our proposed methods — **LMDS**, **FMDS**, and **LMDS+FMDS** — on UTKFace. Results on additional datasets coming soon.

| Dataset | LMDS | FMDS | LMDS+FMDS |
|---------|------|------|-----------|
| UTKFace | 7.039 | 6.961 | 7.032 |
