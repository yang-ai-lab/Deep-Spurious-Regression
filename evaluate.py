import argparse
import torch
from torch.utils.data import DataLoader
from model import SupResNet
from dataset import DATASETS
from utils import get_transforms, AverageMeter


def parse_args():
    parser = argparse.ArgumentParser(description='DSR Evaluation')
    parser.add_argument('--dataset', type=str, required=True, choices=['UTKFace'])
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint (.pth)')
    parser.add_argument('--data_folder', type=str, default='./data', help='path to data root')
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def evaluate(loader, model, device):
    model.eval()
    meter = AverageMeter()
    criterion = torch.nn.L1Loss()
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            meter.update(loss.item(), labels.size(0))
    return meter.avg


def main():
    args = parse_args()

    import dataset as ds_module
    ds_module.DATA_ROOT = args.data_folder

    transform = get_transforms(split=args.split)
    dataset = DATASETS[args.dataset](split=args.split, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SupResNet(name=args.model, num_classes=1).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    print(f'Loaded checkpoint: {args.ckpt}')

    l1 = evaluate(loader, model, device)
    print(f'{args.split.capitalize()} L1 error: {l1:.3f}')


if __name__ == '__main__':
    main()
