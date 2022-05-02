import numpy as np
from typing import Tuple
from datasketch import MinHash, MinHashLSH, MinHashLSHForest, WeightedMinHashGenerator
from multiprocessing import Pool
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
        self.hash_gen = WeightedMinHashGenerator(dim=2048, sample_size=num_permutations, seed=42)

    def build_forest(self,
                     full_loader: DataLoader) -> Tuple[MinHashLSHForest, MinHashLSH]:

        self.model.eval()
        iter_num = 0
        with torch.no_grad():
            for (train_img, target_img) in tqdm(full_loader):
                train_img = train_img.to(self.device)
                model_output = self.model(train_img).cpu()
                self.batch_query(iter_num, model_output)
                iter_num += len(model_output)
        return self.tree, self.hash_lsh, self.hash_gen

    def query_img_weighted(self,
                           iter_num: int,
                           feature: np.ndarray) -> None:
        image_hash = self.hash_gen.minhash(feature)
        self.tree.add(self.dataset.get_id(iter_num), image_hash)
        self.hash_lsh.insert(self.dataset.get_id(iter_num), image_hash)

    def query_img_simple(self,
                         iter_num: int,
                         feature: np.ndarray) -> None:
        image_hash = MinHash(num_perm=self.num_permutations)
        image_hash.update(feature)
        self.tree.add(self.dataset.get_id(iter_num), image_hash)
        self.hash_lsh.insert(self.dataset.get_id(iter_num), image_hash)

    def batch_query(self, iter_num: int, features: torch.Tensor) -> None:

        def to_numpy(batch_elem: torch.Tensor) -> np.ndarray:
            return batch_elem.squeeze().detach().ceil().int().numpy()

        batch_features = np.array(list(map(to_numpy, features)))

        with Pool() as pool:
            pool_data = [(iter_num + idx, feature) for idx, feature in enumerate(batch_features)]
            pool.starmap(self.query_img_simple, pool_data)
