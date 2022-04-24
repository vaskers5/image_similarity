from tqdm import tqdm
from typing import List
import pandas as pd
from datasketch import MinHash, MinHashLSHForest
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from lib.dataset import FolderDataset
from lib.utils import TRANSFORMS_LIST


class ForestInferer:
    def __init__(self,
                 df: pd.DataFrame,
                 model: nn.Module,
                 device: str,
                 batch_size: int,
                 tree: MinHashLSHForest,
                 num_permutations: int,
                 ):
        self.df = df
        self.model = model
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self.tree = tree
        self.num_perm = num_permutations
        all_imgs = df.local_path.to_list()
        ids = df.id.to_list()
        self.dataset = FolderDataset(all_imgs, ids, TRANSFORMS_LIST)
        self.full_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size)

    def collections_check_filter(self,
                                 query_sample_id: int,
                                 result_ids: List[int]) -> List[int]:
        collection_id = self.df[self.df['id'] == query_sample_id].iloc[0].collectionId
        if pd.isna(collection_id):
            return result_ids
        df_ids_filter = self.df[self.df['id'].isin(result_ids)]
        filtred_results = df_ids_filter.loc[df_ids_filter['collectionId'] != collection_id].id.to_list()
        return filtred_results

    def plot_img(self, img_ids: List[int]):
        img_paths = self.df[self.df['id'].isin(img_ids)].local_path.to_list()
        pic_box = plt.figure(figsize=(20, 20))
        for i, img_path in enumerate(img_paths):
            pic_box.add_subplot(2, 5, i + 1)
            plt.imshow(img_path.permute(1, 2, 0))
            plt.axis('off')
        plt.show()

    def start_infer(self, top_k: int) -> dict:
        duplicates_search_res = {}
        iter_num = 0
        for batch in tqdm(self.full_loader):
            batch = batch[0].to(self.device)
            batch_features = self.model(batch).cpu()
            for features in batch_features:
                features = features.squeeze(0).detach().ceil().int().numpy()
                m_hash = MinHash(self.num_perm)
                m_hash.update(features)
                result = self.tree.query(m_hash, top_k)
                query_sample_id = self.df.iloc[iter_num].id
                filtred_tree_results = self.collections_check_filter(query_sample_id, result)
                iter_num += 1
                duplicates_search_res[str(query_sample_id)] = [str(key) for key in filtred_tree_results]
        return duplicates_search_res
