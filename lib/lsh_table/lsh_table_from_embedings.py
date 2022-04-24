import numpy as np
from typing import Tuple
from datasketch import MinHash, MinHashLSH, MinHashLSHForest
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class HashBuilderFromModel:
    def __init__(self,
                 model: nn.Module,
                 device: str,
                 dataset: Dataset,
                 num_permutations: int,
                 num_prefix_trees: int,
                 lsh_threshold: float):
        model.to(device)
        self.model = model
        self.device = device
        self.dataset = dataset
        self.num_permutations = num_permutations
        self.tree = MinHashLSHForest(num_perm=num_permutations, l=num_prefix_trees)
        self.hash_lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_permutations)

    def build_forest(self,
                     full_loader: DataLoader) -> Tuple[MinHashLSHForest, MinHashLSH]:

        self.model.eval()
        iter_num = 0
        with torch.no_grad():
            for (train_img, target_img) in tqdm(full_loader):
                train_img = train_img.to(self.device)
                model_output = self.model(train_img).cpu()
                for image in model_output:
                    image = image.squeeze(0).detach().ceil().int().numpy()
                    image_hash = self.get_min_embeding(image, self.num_permutations)
                    self.tree.add(self.dataset.get_id(iter_num), image_hash)
                    self.hash_lsh.insert(self.dataset.get_id(iter_num), image_hash)
                    iter_num += 1
        return self.tree, self.hash_lsh

    @staticmethod
    def get_min_embeding(embeging: np.ndarray, permutations: int) -> np.ndarray:
        value = MinHash(num_perm=permutations)

        value.update(embeging)
        return value
