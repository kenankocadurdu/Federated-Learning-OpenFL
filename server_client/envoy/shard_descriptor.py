import logging
import os
from pathlib import Path
from typing import Tuple
from urllib.request import urlretrieve
from zipfile import ZipFile
import numpy as np
from PIL import Image
from openfl.interface.interactive_api.shard_descriptor import ShardDataset
from openfl.interface.interactive_api.shard_descriptor import ShardDescriptor
from openfl.utilities import tqdm_report_hook
from openfl.utilities import validate_file_hash
from openfl.utilities.data_splitters.numpy import LogNormalNumPyDataSplitter

logger = logging.getLogger(__name__)

class MyShardDataset(ShardDataset):
    def __init__(self, data_folder: Path, data_type='train', rank=1, worldsize=1):
        self.data_type = data_type
        self.samples = []
        root = Path(data_folder)
        classes = [d.name for d in root.iterdir() if d.is_dir()]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        root = root.absolute()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = root / target_class
            for path in sorted(target_dir.glob('*')):
                item = path, class_index
                self.samples.append(item)
        np.random.seed(0)
        np.random.shuffle(self.samples)
        idx_range = list(range(len(self.samples)))
        idx_sep = int(len(idx_range) * 0.8)
        train_idx, test_idx = np.split(idx_range, [idx_sep])       
        if data_type == 'train':
            self.idx = train_idx
        else:
            self.idx = test_idx

    def __len__(self) -> int:
        return len(self.idx)

    def load_pil(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def __getitem__(self, index: int) -> Tuple['Image', int]:
        path, target = self.samples[self.idx[index]]
        sample = self.load_pil(path)
        return sample, target


class MyShardDescriptor(ShardDescriptor):
    DEFAULT_PATH = Path('.') / 'data'
    
    def __init__(self, data_folder: Path = DEFAULT_PATH, rank_worldsize: str = '1,1', **kwargs):

        self.data_folder = Path.cwd() / data_folder
        self.rank, self.worldsize = tuple(int(num) for num in rank_worldsize.split(','))

    def get_dataset(self, dataset_type):
        return MyShardDataset(
            data_folder=self.data_folder,
            data_type=dataset_type,
            rank=self.rank,
            worldsize=self.worldsize
        )

    @property
    def sample_shape(self):
        shape = self.get_dataset('train')[0][0].size
        return [str(dim) for dim in shape]

    @property
    def target_shape(self):
        target = self.get_dataset('train')[0][1]
        shape = np.array([target]).shape
        return [str(dim) for dim in shape]

    @property
    def dataset_description(self) -> str:
        return (f'Dataset, shard number {self.rank}'
                f' out of {self.worldsize}')
