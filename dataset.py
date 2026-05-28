import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data

DATA_ROOT = './data'


def _load_image(dataset, *path_parts):
    return Image.open(os.path.join(DATA_ROOT, dataset, 'images', *path_parts)).convert('RGB')


class UTKFace(data.Dataset):
    def __init__(self, split='test', transform=None):
        df = pd.read_csv('./data/UTKFace.csv')
        self.df = df[df['split'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = np.asarray([row['age']], dtype=np.float32)
        attr = np.asarray([row['race']], dtype=np.int64)
        img = _load_image('UTKFace', row['filename'])
        if self.transform is not None:
            img = self.transform(img)
        return img, label, attr


DATASETS = {
    'UTKFace': UTKFace,
}
