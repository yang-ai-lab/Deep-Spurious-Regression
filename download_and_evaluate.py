from huggingface_hub import hf_hub_download
import subprocess
import argparse

REPO_ID = "yang-ai-lab/Deep-Spurious-Regression"

CHECKPOINTS = {
    "UTKFace": ["LMDS.pth", "FMDS.pth", "LMDS_FMDS.pth"],
}


def run(dataset, ckpt_path, data_folder, split, batch_size):
    cmd = [
        "python", "evaluate.py",
        "--dataset", dataset,
        "--ckpt", ckpt_path,
        "--data_folder", data_folder,
        "--split", split,
        "--batch_size", str(batch_size),
    ]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["UTKFace"],
                        help="dataset to evaluate (default: all)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["LMDS.pth", "FMDS.pth", "LMDS_FMDS.pth"],
                        help="checkpoint to evaluate (default: all)")
    parser.add_argument("--data_folder", type=str, default="./data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(CHECKPOINTS.keys())

    for dataset in datasets:
        methods = [args.method] if args.method else CHECKPOINTS[dataset]
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset} | Method: {method}")
            print(f"{'='*60}")
            ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=f"{dataset}/{method}")
            run(dataset, ckpt_path, args.data_folder, args.split, args.batch_size)


if __name__ == "__main__":
    main()
